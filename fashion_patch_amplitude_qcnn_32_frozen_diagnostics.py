"""
Fashion-MNIST QCNN with patch-wise amplitude encoding


Pipeline:
    28x28 image
    -> black padding to 32x32
    -> split into four 16x16 patches
    -> amplitude encoding of each 16x16 patch on 8 qubits
    -> same 8-qubit convolution + hierarchical pooling 8 -> 4 for each patch
    -> 4 quantum features per patch
    -> concatenate 4 x 4 = 16 quantum features
    -> simplified linear classical head
    -> 10 Fashion-MNIST logits

This version compares two settings with the same seed and data split:
    1) trainable QCNN convolution/pooling parameters;
    2) frozen random QCNN convolution/pooling parameters.

It also logs gradient norms and feature separability diagnostics, so it is
easier to check whether the quantum feature extractor is learning useful
class-separating representations rather than leaving all the work to the
linear classifier.
"""

import os
import json
import time
import random
from typing import Dict, List

import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
import pennylane as qml


torch.set_num_threads(1)


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    "BASE_DIR": "./fashion_patch_amplitude_qcnn_32_frozen_comparison",
    # One seed only: the script runs exactly two experiments, trainable vs frozen QCNN.
    "SEED": 0,
    "RUNS": [
        {"run_name": "trainable_qcnn", "freeze_qcnn": False},
        {"run_name": "frozen_qcnn", "freeze_qcnn": True},
    ],
    "TRAIN_SAMPLES": 1000,
    "VAL_SAMPLES": 400,
    "TEST_SAMPLES": 200,
    "EPOCHS": 7,
    "ACC_STEPS": 8,
    "EARLY_STOP_PATIENCE": 5,
    "PRINT_EVERY": 1,
    "LR_HEAD": 1e-3,
    "LR_QKERNEL": 3e-4,
    "WEIGHT_DECAY_HEAD": 1e-3,
    "WEIGHT_DECAY_QKERNEL": 1e-5,
    "CLIP_NORM": 1.0,
}

N_CLASSES = 10
N_QUBITS = 8
N_PATCHES = 4
PATCH_SIZE = 16
PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE  # 256 = 2^8
KEEP_WIRES = [0, 2, 4, 6]
QFEATURES_PER_PATCH = 4  # 4 local Z readouts, no pairwise correlations
HEAD_IN_DIM = N_PATCHES * QFEATURES_PER_PATCH
EPS = 1e-8

BASE_DIR = CONFIG["BASE_DIR"]


# ============================================================
# Reproducibility
# ============================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Data loading and preprocessing
# ============================================================

# Fashion-MNIST images are originally 28x28. We add black padding of 2 pixels
# on each side to obtain 32x32 images. Then each image is split into four
# 16x16 patches.
transform = transforms.Compose([
    transforms.Pad(padding=2, fill=0),
    transforms.ToTensor(),
])


def get_datasets():
    os.makedirs("./data", exist_ok=True)
    train_dataset = torchvision.datasets.FashionMNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        "./data", train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def stratified_split_indices(dataset_list, n_train: int, n_val: int, seed: int):
    """Create non-overlapping, class-balanced train and validation subsets."""
    rng = random.Random(seed)
    buckets = {c: [] for c in range(N_CLASSES)}

    for i, (_, y) in enumerate(dataset_list):
        buckets[int(y)].append(i)

    def per_class_quota(total: int) -> List[int]:
        base = total // N_CLASSES
        rem = total % N_CLASSES
        return [base + (1 if c < rem else 0) for c in range(N_CLASSES)]

    q_train = per_class_quota(n_train)
    q_val = per_class_quota(n_val)

    train_idx, val_idx = [], []
    for c in range(N_CLASSES):
        pool = buckets[c][:]
        rng.shuffle(pool)

        train_idx.extend(pool[: q_train[c]])
        val_idx.extend(pool[q_train[c] : q_train[c] + q_val[c]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def stratified_indices(dataset_list, n_samples: int, seed: int):
    """Create one class-balanced subset, used here for test sampling."""
    rng = random.Random(seed)
    buckets = {c: [] for c in range(N_CLASSES)}

    for i, (_, y) in enumerate(dataset_list):
        buckets[int(y)].append(i)

    base = n_samples // N_CLASSES
    rem = n_samples % N_CLASSES

    idxs = []
    for c in range(N_CLASSES):
        take = base + (1 if c < rem else 0)
        take = min(take, len(buckets[c]))
        idxs.extend(rng.sample(buckets[c], take))

    rng.shuffle(idxs)
    return idxs


# ============================================================
# Feature extraction
# ============================================================

TRAIN_CACHE: Dict[int, Dict[str, torch.Tensor]] = {}
TEST_CACHE: Dict[int, Dict[str, torch.Tensor]] = {}


def image_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    return img_tensor.squeeze(0).cpu().numpy().astype(np.float32)


def split_patches_16x16(X: np.ndarray) -> List[np.ndarray]:
    """32x32 image -> four 16x16 patches: TL, TR, BL, BR."""
    return [
        X[0:16, 0:16],
        X[0:16, 16:32],
        X[16:32, 0:16],
        X[16:32, 16:32],
    ]


def patch_amplitudes_4x256(X: np.ndarray) -> np.ndarray:
    """32x32 image -> 4 independently normalized 16x16 amplitude vectors.

    Each 16x16 patch has 256 pixels, so it fits exactly into an 8-qubit
    amplitude embedding without resizing and without padding inside the quantum
    circuit.
    """
    amps = []
    for patch in split_patches_16x16(X):
        amp = patch.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(amp))

        if norm < EPS:
            amp = np.zeros(PATCH_PIXELS, dtype=np.float32)
            amp[0] = 1.0
        else:
            amp = amp / norm

        amps.append(amp)

    return np.stack(amps, axis=0).astype(np.float32)  # (4, 256)


def get_features(dataset_list, cache: Dict[int, Dict[str, torch.Tensor]], idx: int):
    if idx in cache:
        return cache[idx]

    x, y = dataset_list[idx]
    X = image_to_numpy(x)  # already padded to 32x32 by transform

    sample = {
        "amp4x256": torch.tensor(patch_amplitudes_4x256(X), dtype=torch.float32),
        "y": int(y),
    }

    cache[idx] = sample
    return sample


# ============================================================
# Quantum device
# ============================================================

def make_device(n_wires: int):
    try:
        dev = qml.device("lightning.qubit", wires=n_wires)
        return dev, "lightning.qubit", "adjoint"
    except Exception:
        dev = qml.device("default.qubit", wires=n_wires)
        return dev, "default.qubit", "backprop"


dev8, dev8_name, diff8 = make_device(N_QUBITS)
QNODE_KW = dict(interface="torch", diff_method=diff8)


# ============================================================
# Quantum circuit
# ============================================================

def conv8(theta_conv):
    """One 8-qubit convolutional block shared across the four patches."""
    for i in range(N_QUBITS):
        qml.RY(theta_conv[i], wires=i)

    # local pair interactions
    for a, b, k in [(0, 1, 8), (2, 3, 9), (4, 5, 10), (6, 7, 11)]:
        qml.CNOT(wires=[a, b])
        qml.RZ(theta_conv[k], wires=b)
        qml.CNOT(wires=[a, b])

    # shifted interactions
    for a, b, k in [(1, 2, 12), (3, 4, 13), (5, 6, 14), (7, 0, 15)]:
        qml.CNOT(wires=[a, b])
        qml.RZ(theta_conv[k], wires=b)
        qml.CNOT(wires=[a, b])


def hierarchical_pool_8_to_4(phi_pool):
    """One hierarchical pooling step: 8 qubits -> 4 retained qubits.

    The retained wires are [0, 2, 4, 6]. The odd wires are used as controls and
    then discarded by not measuring them.
    """
    pairs = [(1, 0), (3, 2), (5, 4), (7, 6)]
    for j, (control, target) in enumerate(pairs):
        qml.PauliX(wires=control)
        qml.CRX(phi_pool[2 * j], wires=[control, target])
        qml.PauliX(wires=control)
        qml.CRZ(phi_pool[2 * j + 1], wires=[control, target])


def readout_features():
    """Return only the 4 local Z features on retained wires."""
    return [qml.expval(qml.PauliZ(i)) for i in KEEP_WIRES]


@qml.qnode(dev8, **QNODE_KW)
def patch_amplitude_qnode(amp256, theta_conv, phi_pool):
    qml.AmplitudeEmbedding(
        amp256,
        wires=range(N_QUBITS),
        normalize=True,
    )
    conv8(theta_conv)
    hierarchical_pool_8_to_4(phi_pool)
    return readout_features()  # 4 features per patch


# ============================================================
# Model
# ============================================================

class PatchAmplitudeQCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # QCNN parameters shared across the four 16x16 patches.
        self.theta_conv = torch.nn.Parameter(0.01 * torch.randn(16, dtype=torch.float32))
        self.phi_pool = torch.nn.Parameter(0.01 * torch.randn(8, dtype=torch.float32))

        # Simplified classical head: 4 patches x 4 quantum features = 16 input features.
        # This keeps the readout intentionally small, so good performance is less
        # likely to be explained only by a powerful classical classifier.
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(HEAD_IN_DIM),
            torch.nn.Linear(HEAD_IN_DIM, N_CLASSES),
        )

    def quantum_features(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        patch_features = []
        amp4x256 = sample["amp4x256"]

        for p in range(N_PATCHES):
            out = patch_amplitude_qnode(
                amp4x256[p],
                self.theta_conv,
                self.phi_pool,
            )
            patch_features.append(torch.stack(out).to(dtype=torch.float32))

        return torch.cat(patch_features, dim=0)  # 4 x 4 = 16 features

    def forward(self, sample: Dict[str, torch.Tensor]):
        features = self.quantum_features(sample)
        logits = self.head(features)
        return logits, features


# ============================================================
# Training and evaluation
# ============================================================

ce_loss = torch.nn.CrossEntropyLoss()


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grad_norm(parameters) -> float:
    """L2 norm of gradients for a parameter iterable."""
    total = 0.0
    for p in parameters:
        if p is None or p.grad is None:
            continue
        total += float(p.grad.detach().norm().item() ** 2)
    return float(total ** 0.5)


def model_grad_norms(model: PatchAmplitudeQCNN) -> Dict[str, float]:
    """Separate gradient norms for the quantum kernel and the classical head."""
    qkernel_norm = grad_norm([model.theta_conv, model.phi_pool])
    head_norm = grad_norm(model.head.parameters())
    ratio = qkernel_norm / (head_norm + 1e-12)
    return {
        "grad_qkernel": qkernel_norm,
        "grad_head": head_norm,
        "grad_qkernel_over_head": ratio,
    }


def feature_separability_diagnostics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute simple class-separability diagnostics for quantum features.

    The most useful value is feature_between_within_ratio:
        high  -> class means are separated relative to within-class spread;
        low   -> quantum features overlap strongly across classes.
    """
    if features.size == 0:
        return {
            "feature_std_mean": 0.0,
            "feature_var_mean": 0.0,
            "feature_between_var_mean": 0.0,
            "feature_within_var_mean": 0.0,
            "feature_between_within_ratio": 0.0,
            "feature_class_center_distance_mean": 0.0,
        }

    F = features.astype(np.float64)
    y = labels.astype(np.int64)

    global_mean = F.mean(axis=0)
    global_var = F.var(axis=0)
    global_std = F.std(axis=0)

    class_means = []
    within_vars = []
    for c in range(N_CLASSES):
        Fc = F[y == c]
        if len(Fc) == 0:
            continue
        class_means.append(Fc.mean(axis=0))
        within_vars.append(Fc.var(axis=0).mean())

    if len(class_means) == 0:
        between_var_mean = 0.0
        within_var_mean = 0.0
        center_distance_mean = 0.0
    else:
        C = np.stack(class_means, axis=0)
        # Mean squared displacement of class centers from the global center.
        between_var_mean = float(((C - global_mean) ** 2).mean())
        within_var_mean = float(np.mean(within_vars)) if within_vars else 0.0

        if len(C) > 1:
            dists = []
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    dists.append(float(np.linalg.norm(C[i] - C[j])))
            center_distance_mean = float(np.mean(dists)) if dists else 0.0
        else:
            center_distance_mean = 0.0

    return {
        "feature_std_mean": float(global_std.mean()),
        "feature_var_mean": float(global_var.mean()),
        "feature_between_var_mean": float(between_var_mean),
        "feature_within_var_mean": float(within_var_mean),
        "feature_between_within_ratio": float(between_var_mean / (within_var_mean + 1e-12)),
        "feature_class_center_distance_mean": float(center_distance_mean),
    }


@torch.no_grad()
def evaluate(model, dataset_list, cache, indices, with_feature_diagnostics: bool = False):
    model.eval()
    losses = []
    correct = 0
    total = 0
    features_list = []
    labels_list = []

    for idx in indices:
        sample = get_features(dataset_list, cache, idx)
        logits, features = model(sample)
        y = torch.tensor([sample["y"]], dtype=torch.long)
        loss = ce_loss(logits.view(1, -1), y)

        pred = int(torch.argmax(logits).item())
        correct += int(pred == sample["y"])
        total += 1
        losses.append(float(loss.item()))

        if with_feature_diagnostics:
            features_list.append(features.detach().cpu().numpy())
            labels_list.append(sample["y"])

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": float(correct / total) if total else 0.0,
    }

    if with_feature_diagnostics:
        F = np.stack(features_list, axis=0) if features_list else np.empty((0, HEAD_IN_DIM))
        y = np.array(labels_list, dtype=np.int64)
        metrics.update(feature_separability_diagnostics(F, y))

    return metrics


def build_optimizer(model: PatchAmplitudeQCNN):
    param_groups = []

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": CONFIG["LR_HEAD"],
            "weight_decay": CONFIG["WEIGHT_DECAY_HEAD"],
        })

    qkernel_params = [p for p in [model.theta_conv, model.phi_pool] if p.requires_grad]
    if qkernel_params:
        param_groups.append({
            "params": qkernel_params,
            "lr": CONFIG["LR_QKERNEL"],
            "weight_decay": CONFIG["WEIGHT_DECAY_QKERNEL"],
        })

    if not param_groups:
        raise ValueError("No trainable parameters were passed to the optimizer.")

    return torch.optim.AdamW(param_groups)


def train_one_run(run_name: str, freeze_qcnn: bool, seed: int, train_data, test_data):
    run_dir = os.path.join(BASE_DIR, f"{run_name}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Resetting the seed here makes the trainable and frozen runs start from the
    # same initial quantum kernel and the same initial classifier weights.
    set_global_seed(seed)

    train_idx, val_idx = stratified_split_indices(
        train_data,
        CONFIG["TRAIN_SAMPLES"],
        CONFIG["VAL_SAMPLES"],
        seed=seed + 101,
    )
    test_idx = stratified_indices(
        test_data,
        CONFIG["TEST_SAMPLES"],
        seed=seed + 303,
    )

    model = PatchAmplitudeQCNN()
    theta_conv_init = model.theta_conv.detach().clone()
    phi_pool_init = model.phi_pool.detach().clone()

    if freeze_qcnn:
        model.theta_conv.requires_grad_(False)
        model.phi_pool.requires_grad_(False)

    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    print("\n" + "=" * 90)
    print(f"RUN={run_name} | seed={seed} | freeze_qcnn={freeze_qcnn}")
    print(f"Quantum device: {dev8_name} ({diff8})")
    print("Input: 28x28 -> black padding -> 32x32 -> four 16x16 patches")
    print("Encoding: amplitude encoding on each 16x16 patch")
    print("Convolution structure: 4 patch QNode calls per image, shared QCNN parameters")
    print(f"Classical head: LayerNorm({HEAD_IN_DIM}) + Linear({HEAD_IN_DIM} -> 10)")
    print(f"Trainable parameters: {count_trainable_params(model)}")
    print(f"Train/Val/Test samples: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    rows = []
    best_val_loss = float("inf")
    best_state = None
    patience = 0
    acc_steps = max(1, int(CONFIG["ACC_STEPS"]))

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        t0 = time.time()
        model.train()
        random.shuffle(train_idx)

        running_loss = 0.0
        grad_qkernel_values = []
        grad_head_values = []
        grad_ratio_values = []

        optimizer.zero_grad()
        step_in_acc = 0

        for idx in train_idx:
            sample = get_features(train_data, TRAIN_CACHE, idx)
            logits, _ = model(sample)
            y = torch.tensor([sample["y"]], dtype=torch.long)
            loss = ce_loss(logits.view(1, -1), y)

            (loss / acc_steps).backward()
            running_loss += float(loss.item())
            step_in_acc += 1

            if step_in_acc == acc_steps:
                norms = model_grad_norms(model)
                grad_qkernel_values.append(norms["grad_qkernel"])
                grad_head_values.append(norms["grad_head"])
                grad_ratio_values.append(norms["grad_qkernel_over_head"])

                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    CONFIG["CLIP_NORM"],
                )
                optimizer.step()
                optimizer.zero_grad()
                step_in_acc = 0

        if step_in_acc != 0:
            norms = model_grad_norms(model)
            grad_qkernel_values.append(norms["grad_qkernel"])
            grad_head_values.append(norms["grad_head"])
            grad_ratio_values.append(norms["grad_qkernel_over_head"])

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                CONFIG["CLIP_NORM"],
            )
            optimizer.step()
            optimizer.zero_grad()

        train_loss = running_loss / max(1, len(train_idx))

        # Validation includes feature separability diagnostics. Test is evaluated
        # normally during training and with diagnostics at the end.
        val_metrics = evaluate(
            model, train_data, TRAIN_CACHE, val_idx, with_feature_diagnostics=True
        )
        test_metrics = evaluate(model, test_data, TEST_CACHE, test_idx)
        scheduler.step(val_metrics["loss"])

        grad_qkernel_mean = float(np.mean(grad_qkernel_values)) if grad_qkernel_values else 0.0
        grad_head_mean = float(np.mean(grad_head_values)) if grad_head_values else 0.0
        grad_ratio_mean = float(np.mean(grad_ratio_values)) if grad_ratio_values else 0.0
        theta_conv_drift = float(torch.norm(model.theta_conv.detach() - theta_conv_init).item())
        phi_pool_drift = float(torch.norm(model.phi_pool.detach() - phi_pool_init).item())

        row = {
            "run_name": run_name,
            "freeze_qcnn": bool(freeze_qcnn),
            "seed": seed,
            "epoch": epoch,
            "nparams_trainable": count_trainable_params(model),
            "n_patch_convolutions_per_image": N_PATCHES,
            "grad_qkernel_mean": grad_qkernel_mean,
            "grad_head_mean": grad_head_mean,
            "grad_qkernel_over_head_mean": grad_ratio_mean,
            "theta_conv_drift": theta_conv_drift,
            "phi_pool_drift": phi_pool_drift,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "val_feature_std_mean": val_metrics["feature_std_mean"],
            "val_feature_var_mean": val_metrics["feature_var_mean"],
            "val_feature_between_var_mean": val_metrics["feature_between_var_mean"],
            "val_feature_within_var_mean": val_metrics["feature_within_var_mean"],
            "val_feature_between_within_ratio": val_metrics["feature_between_within_ratio"],
            "val_feature_class_center_distance_mean": val_metrics["feature_class_center_distance_mean"],
            "sec_epoch": time.time() - t0,
        }
        rows.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch == 1 or epoch % CONFIG["PRINT_EVERY"] == 0:
            print(
                f"epoch={epoch:02d} "
                f"train_loss={train_loss:.4f} "
                f"val_acc={val_metrics['acc']:.3f} val_loss={val_metrics['loss']:.4f} "
                f"test_acc={test_metrics['acc']:.3f} test_loss={test_metrics['loss']:.4f} "
                f"grad_q={grad_qkernel_mean:.2e} grad_head={grad_head_mean:.2e} "
                f"ratio={grad_ratio_mean:.2e} "
                f"theta_drift={theta_conv_drift:.2e} phi_drift={phi_pool_drift:.2e} "
                f"sep_ratio={val_metrics['feature_between_within_ratio']:.2e} "
                f"center_dist={val_metrics['feature_class_center_distance_mean']:.2e} "
                f"sec={row['sec_epoch']:.1f}"
            )

        if patience >= CONFIG["EARLY_STOP_PATIENCE"]:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate(
        model, train_data, TRAIN_CACHE, val_idx, with_feature_diagnostics=True
    )
    final_test = evaluate(
        model, test_data, TEST_CACHE, test_idx, with_feature_diagnostics=True
    )

    metrics_df = pd.DataFrame(rows)

    final = {
        "run_name": run_name,
        "freeze_qcnn": bool(freeze_qcnn),
        "seed": seed,
        "nparams_trainable": count_trainable_params(model),
        "n_patch_convolutions_per_image": N_PATCHES,
        "final_theta_conv_drift": float(torch.norm(model.theta_conv.detach() - theta_conv_init).item()),
        "final_phi_pool_drift": float(torch.norm(model.phi_pool.detach() - phi_pool_init).item()),
        "best_val_loss": best_val_loss,
        "final_val_acc": final_val["acc"],
        "final_val_loss": final_val["loss"],
        "final_test_acc": final_test["acc"],
        "final_test_loss": final_test["loss"],
        "final_val_feature_between_within_ratio": final_val["feature_between_within_ratio"],
        "final_val_feature_center_distance_mean": final_val["feature_class_center_distance_mean"],
        "final_test_feature_between_within_ratio": final_test["feature_between_within_ratio"],
        "final_test_feature_center_distance_mean": final_test["feature_class_center_distance_mean"],
        "final_val_feature_std_mean": final_val["feature_std_mean"],
        "final_test_feature_std_mean": final_test["feature_std_mean"],
    }

    with open(os.path.join(run_dir, "final_eval.json"), "w") as f:
        json.dump(final, f, indent=2)

    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

    print("Final:", json.dumps(final, indent=2))
    return metrics_df, final


def run_all(train_data, test_data):
    os.makedirs(BASE_DIR, exist_ok=True)

    all_metrics = []
    all_finals = []
    seed = int(CONFIG["SEED"])

    for run_cfg in CONFIG["RUNS"]:
        metrics_df, final = train_one_run(
            run_name=run_cfg["run_name"],
            freeze_qcnn=bool(run_cfg["freeze_qcnn"]),
            seed=seed,
            train_data=train_data,
            test_data=test_data,
        )
        all_metrics.append(metrics_df)
        all_finals.append(final)

    metrics_all = pd.concat(all_metrics, axis=0).reset_index(drop=True)
    finals_df = pd.DataFrame(all_finals)

    summary = finals_df[[
        "run_name", "freeze_qcnn", "final_val_acc", "final_val_loss",
        "final_test_acc", "final_test_loss", "nparams_trainable",
        "final_theta_conv_drift", "final_phi_pool_drift",
        "final_val_feature_between_within_ratio",
        "final_test_feature_between_within_ratio",
        "final_val_feature_center_distance_mean",
        "final_test_feature_center_distance_mean",
    ]].copy()

    with open(os.path.join(BASE_DIR, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("\n" + "=" * 90)
    print("FINAL TRAINABLE VS FROZEN SUMMARY")
    print(summary)

    return metrics_all, finals_df, summary


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    set_global_seed(CONFIG["SEED"])

    print("Downloading Fashion-MNIST...")
    train_dataset, test_dataset = get_datasets()

    print("Loading train split into memory...")
    train_data = [(x, int(y)) for x, y in train_dataset]

    print("Loading test split into memory...")
    test_data = [(x, int(y)) for x, y in test_dataset]

    print(
        f"Fashion-MNIST padded to 32x32 | train={len(train_data)} | "
        f"test={len(test_data)} | classes={N_CLASSES}"
    )

    run_all(train_data, test_data)
