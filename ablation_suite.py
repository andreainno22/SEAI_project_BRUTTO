"""
ablation_suite.py — Orchestration script for the QCNN ablation test suite.

For each test defined in test_suite_config.ABLATION_TESTS:
  1. Builds a DynamicQCNN from the merged config (baseline + overrides).
  2. Trains it on a stratified subset of Fashion-MNIST.
  3. Saves per-epoch metrics to ablation_results/<test_name>/metrics.csv.
  4. Evaluates the best checkpoint on the held-out test set.

At the end, writes ablation_results/final_comparison.csv with the
test accuracy for every variant, ready for comparison.

Usage:
    python ablation_suite.py                     # runs all tests
    python ablation_suite.py baseline_amplitude  # runs one specific test
"""

import os
import sys
import time
import json
import random

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset

from test_suite_config import BASELINE_CONFIG, ABLATION_TESTS, get_config
from qcnn_builder import DynamicQCNN

# ---------------------------------------------------------------------------
# Environment (set before any torch/pennylane import resolves threads)
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
torch.set_num_threads(1)

EPS = 1e-8


# ============================================================
# Feature extraction — unified for all encoding variants
# ============================================================

def image_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert a (1, H, W) float tensor to a (H, W) float32 numpy array."""
    return img_tensor.squeeze(0).cpu().numpy().astype(np.float32)


def split_patches_16x16(X: np.ndarray):
    """Split a 32×32 image into four 16×16 patches (TL, TR, BL, BR)."""
    return [
        X[0:16,  0:16],
        X[0:16,  16:32],
        X[16:32, 0:16],
        X[16:32, 16:32],
    ]


def extract_features(X: np.ndarray) -> dict:
    """Pre-compute all features needed by any encoding variant.

    Returns a dict with:
      amp4x256    (4, 256) float32 — unit-norm amplitude vectors per patch
      norm_angles (4,)    float32 — relative L2 norm angles in [0, pi] (C9)
      quad_means  (4, 8)  float32 — 8 horizontal strip means per patch (E1/E4)
      gA4         (4,)    float32 — global image statistics [mean, var, grad, asym]
    """
    patches = split_patches_16x16(X)

    # --- Amplitude features ---
    amps, raw_norms = [], []
    for patch in patches:
        amp  = patch.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(amp))
        raw_norms.append(norm)
        if norm < EPS:
            amp    = np.zeros(256, dtype=np.float32)
            amp[0] = 1.0
        else:
            amp = amp / norm
        amps.append(amp)

    total       = sum(raw_norms) + EPS
    norm_angles = np.array([np.pi * n / total for n in raw_norms], dtype=np.float32)
    amp4x256    = np.stack(amps, axis=0)    # (4, 256)

    # --- Strip means per patch (E1 encoding only) ---
    # Each 16×16 patch is split into 8 horizontal strips of 2×16 pixels.
    # The 8 strip means feed the 8 local-wire RY rotations of the E1 ansatz.
    quad_means_list = []
    for patch in patches:
        strips = [patch[r * 2:(r + 1) * 2, :].mean() for r in range(8)]
        quad_means_list.append(np.array(strips, dtype=np.float32))
    quad_means = np.stack(quad_means_list, axis=0)   # (4, 8)

    # --- Global statistics gA4 (over the whole 32×32 image) ---
    # g3 is the mean squared gradient magnitude on the overlap region where
    # both dx and dy are defined (kept consistent with the legacy formula).
    g1 = float(X.mean())
    g2 = float(((X - g1) ** 2).mean())
    dx = X[:, 1:] - X[:, :-1]    # (32, 31)
    dy = X[1:, :] - X[:-1, :]    # (31, 32)
    dx_o = dx[0:31, 0:31]
    dy_o = dy[0:31, 0:31]
    g3 = float((dx_o ** 2 + dy_o ** 2).mean())
    H  = float((dx ** 2).sum())
    V  = float((dy ** 2).sum())
    g4 = float((V - H) / (V + H + EPS))
    gA4 = np.array([g1, g2, g3, g4], dtype=np.float32)

    return {
        "amp4x256":    torch.tensor(amp4x256,    dtype=torch.float32),
        "norm_angles": torch.tensor(norm_angles, dtype=torch.float32),
        "quad_means":  torch.tensor(quad_means,  dtype=torch.float32),
        "gA4":         torch.tensor(gA4,         dtype=torch.float32),
    }


class UnifiedDataset(Dataset):
    """Fashion-MNIST wrapper that pre-extracts and caches all features.

    The cache is populated lazily on first access. Because all features
    (amplitude, strip means, gA4) are computed in one pass, the same
    dataset object can serve every encoding variant without re-loading.
    """

    def __init__(self, base_dataset):
        self.dataset = base_dataset
        self._cache: dict = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        if idx not in self._cache:
            x, y   = self.dataset[idx]
            X      = image_to_numpy(x)
            sample = extract_features(X)
            sample["y"] = int(y)
            self._cache[idx] = sample
        return self._cache[idx]


# ============================================================
# Stratified subset sampling (class-balanced, reproducible)
# ============================================================

def stratified_indices(dataset: UnifiedDataset, n_samples: int, seed: int,
                       n_classes: int = 10) -> list:
    """Return class-balanced indices from `dataset`."""
    rng     = random.Random(seed)
    buckets = {c: [] for c in range(n_classes)}
    for i in range(len(dataset)):
        y = int(dataset.dataset[i][1])
        buckets[y].append(i)

    base = n_samples // n_classes
    rem  = n_samples % n_classes
    idxs = []
    for c in range(n_classes):
        take = min(base + (1 if c < rem else 0), len(buckets[c]))
        idxs.extend(rng.sample(buckets[c], take))

    rng.shuffle(idxs)
    return idxs


def stratified_split(dataset: UnifiedDataset, n_train: int, n_val: int,
                     seed: int, n_classes: int = 10):
    """Return non-overlapping class-balanced train and val index lists."""
    rng     = random.Random(seed)
    buckets = {c: [] for c in range(n_classes)}
    for i in range(len(dataset)):
        y = int(dataset.dataset[i][1])
        buckets[y].append(i)

    base_tr = n_train // n_classes
    rem_tr  = n_train % n_classes
    base_vl = n_val   // n_classes
    rem_vl  = n_val   % n_classes

    train_idx, val_idx = [], []
    for c in range(n_classes):
        pool = buckets[c][:]
        rng.shuffle(pool)
        qt = base_tr + (1 if c < rem_tr else 0)
        qv = base_vl + (1 if c < rem_vl else 0)
        train_idx.extend(pool[:qt])
        val_idx.extend(pool[qt:qt + qv])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# ============================================================
# Optimizer & scheduler factory
# ============================================================

def build_optimizer(model: DynamicQCNN, config: dict) -> torch.optim.Optimizer:
    """Create AdamW with per-group learning rates."""
    groups = []

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    if head_params:
        groups.append({
            "params":       head_params,
            "lr":           config["LR_HEAD"],
            "weight_decay": config["WEIGHT_DECAY_HEAD"],
        })

    if model.theta_q.requires_grad:
        groups.append({
            "params":       [model.theta_q],
            "lr":           config["LR_QKERNEL"],
            "weight_decay": config["WEIGHT_DECAY_QKERNEL"],
        })

    if model.a_embed is not None and model.a_embed.requires_grad:
        groups.append({
            "params":       [model.a_embed, model.c_embed],
            "lr":           config["LR_EMBED"],
            "weight_decay": config["WEIGHT_DECAY_EMBED"],
        })

    if not groups:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(groups)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model: DynamicQCNN, loader: DataLoader,
             loss_fn: torch.nn.Module) -> tuple:
    """Return (mean_loss, accuracy) on the given loader.

    `loss_fn` should be a plain CrossEntropyLoss (no label smoothing) so that
    val/test losses stay comparable across ablation runs and to the literature.
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch in loader:
        logits, _ = model(batch)
        y         = batch["y"]
        loss      = loss_fn(logits, y)
        preds     = logits.argmax(dim=1)

        total_loss += loss.item() * len(y)
        correct    += (preds == y).sum().item()
        total      += len(y)

    mean_loss = total_loss / total if total else 0.0
    accuracy  = correct   / total if total else 0.0
    return mean_loss, accuracy


# ============================================================
# Single test run
# ============================================================

def run_test(test_name: str, config: dict,
             train_dataset: UnifiedDataset,
             test_dataset:  UnifiedDataset,
             seed: int = 42) -> dict:
    """Train and evaluate one ablation configuration.

    Returns a result dict with test_name, final_test_acc, final_test_loss,
    best_val_loss, n_trainable_params, and the full config snapshot.
    """
    # Print header with only the overriding keys
    overrides = ABLATION_TESTS.get(test_name, {})
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"  Encoding   : {config['ENCODING']}")
    print(f"  Ansatz     : {config['ANSATZ_TYPE']}")
    print(f"  Readout    : {config['READOUT']}  (wires={config.get('READOUT_WIRES', 'kept')})")
    print(f"  Trainable  : {config['QCNN_TRAINABLE']}")
    print(f"  InjectNorm : {config['INJECT_NORM']}")
    if overrides:
        print(f"  Overrides  : {overrides}")

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Stratified data split
    train_idx, val_idx = stratified_split(
        train_dataset,
        n_train=config["TRAIN_SAMPLES"],
        n_val=config["VAL_SAMPLES"],
        seed=seed + 1,
    )
    test_idx = stratified_indices(
        test_dataset,
        n_samples=config["TEST_SAMPLES"],
        seed=seed + 2,
    )

    train_loader = DataLoader(Subset(train_dataset, train_idx),
                              batch_size=config["BATCH_SIZE"], shuffle=True)
    val_loader   = DataLoader(Subset(train_dataset, val_idx),
                              batch_size=config["BATCH_SIZE"], shuffle=False)
    test_loader  = DataLoader(Subset(test_dataset, test_idx),
                              batch_size=config["BATCH_SIZE"], shuffle=False)

    # Model, optimizer, scheduler, loss
    model     = DynamicQCNN(config)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params}")

    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, config["EPOCHS"] // 3), T_mult=1, eta_min=1e-6
    )
    # Training loss uses label smoothing as a regulariser; evaluation uses the
    # plain cross-entropy so val/test losses stay comparable across ablation
    # runs and to external baselines.
    train_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config["LABEL_SMOOTHING"])
    eval_loss_fn  = torch.nn.CrossEntropyLoss()

    # Output directory
    out_dir = os.path.join("ablation_results", test_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save config snapshot for reproducibility
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Training loop
    best_val_loss = float("inf")
    patience      = 0
    best_ckpt     = os.path.join(out_dir, "best_model.pt")
    rows          = []

    # Remove a stale checkpoint from a previous run: if the architecture changed
    # (e.g. READOUT Z -> Z_ZZ, or ANSATZ_TYPE), the old state_dict shapes won't
    # match and load_state_dict at the end of training would fail or — worse —
    # partially load incompatible weights.
    if os.path.exists(best_ckpt):
        os.remove(best_ckpt)

    for epoch in range(1, config["EPOCHS"] + 1):
        t0           = time.time()
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model(batch)
            y         = batch["y"]
            loss      = train_loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config["CLIP_NORM"],
            )
            optimizer.step()
            running_loss += loss.item() * len(y)

        train_loss          = running_loss / max(1, len(train_idx))
        val_loss, val_acc   = evaluate(model, val_loader, eval_loss_fn)
        epoch_sec           = time.time() - t0
        scheduler.step()

        rows.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "time_sec":   epoch_sec,
        })

        if epoch % config["PRINT_EVERY"] == 0:
            print(f"  Ep {epoch:03d} | "
                  f"train={train_loss:.4f} | "
                  f"val={val_loss:.4f} | "
                  f"acc={val_acc:.3f} | "
                  f"{epoch_sec:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience      = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience += 1

        if patience >= config["EARLY_STOP_PATIENCE"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best checkpoint and evaluate on test set
    if os.path.exists(best_ckpt):
        model.load_state_dict(
            torch.load(best_ckpt, weights_only=True)
        )

    test_loss, test_acc = evaluate(model, test_loader, eval_loss_fn)
    print(f"  → test_acc={test_acc:.4f}  test_loss={test_loss:.4f}")

    # Save per-epoch metrics
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    return {
        "test_name":        test_name,
        "final_test_acc":   round(test_acc,  4),
        "final_test_loss":  round(test_loss, 4),
        "best_val_loss":    round(best_val_loss, 4),
        "n_trainable_params": n_params,
        "encoding":         config["ENCODING"],
        "ansatz":           config["ANSATZ_TYPE"],
        "readout":          config["READOUT"],
        "trainable":        config["QCNN_TRAINABLE"],
        "inject_norm":      config["INJECT_NORM"],
    }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # Optionally run a single test passed as CLI argument
    tests_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(ABLATION_TESTS.keys())

    # Validate requested test names early
    for t in tests_to_run:
        if t not in ABLATION_TESTS:
            print(f"Error: '{t}' is not a valid test name. "
                  f"Available: {list(ABLATION_TESTS.keys())}")
            sys.exit(1)

    # Build datasets (one pass, shared across all tests via the cache)
    os.makedirs("./data", exist_ok=True)
    _transform = transforms.Compose([
        transforms.Pad(2),        # 28×28 -> 32×32, no interpolation
        transforms.ToTensor(),
    ])
    raw_train = torchvision.datasets.FashionMNIST(
        "./data", train=True,  download=True, transform=_transform
    )
    raw_test = torchvision.datasets.FashionMNIST(
        "./data", train=False, download=True, transform=_transform
    )

    # Note: RandomHorizontalFlip is intentionally omitted here.
    # Augmentation is applied at image level, but UnifiedDataset caches
    # the features computed from the (already transformed) image tensor.
    # Since caching is index-based and each epoch re-uses cached values,
    # per-epoch stochastic augmentation is not compatible with a pre-cache
    # strategy. For strong augmentation, disable the cache or pre-generate
    # multiple augmented copies per sample (out of scope for ablation speed).
    train_dataset = UnifiedDataset(raw_train)
    test_dataset  = UnifiedDataset(raw_test)

    os.makedirs("ablation_results", exist_ok=True)
    out_path = os.path.join("ablation_results", "final_comparison.csv")

    results = []

    for tname in tests_to_run:
        cfg    = get_config(tname)
        result = run_test(tname, cfg, train_dataset, test_dataset, seed=42)
        results.append(result)

        # Write after every test so a crash doesn't lose completed runs.
        res_df = pd.DataFrame(results).sort_values("final_test_acc", ascending=False)
        res_df.to_csv(out_path, index=False)
        print(f"[checkpoint] {tname} done — results saved to {out_path}")

    print(f"\n{'='*60}")
    print("FINAL ABLATION RESULTS (sorted by test accuracy)")
    print("="*60)
    print(res_df.to_string(index=False))
    print(f"\nSaved to: {out_path}")
