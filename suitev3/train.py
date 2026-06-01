"""
train.py - Generic trainer for any v3 arm (quantum or classical baseline).

All arms return logits (B, 4) and are trained identically:
  - AdamW, single group, lr=V3.LR, weight_decay=V3.WEIGHT_DECAY  (1e-3 / 0.0, from suitev2)
  - F.cross_entropy with label_smoothing=BASELINE.LABEL_SMOOTHING during training,
    plain cross-entropy for val/test (matches suitev2's "smoothing on train only")
  - CosineAnnealingWarmRestarts(T_0=epochs//3, eta_min=1e-4)  (same as suitev2)
  - early stopping on val_loss, best-by-val-loss checkpoint used for the test eval

The DataLoader is suitev2's (identical 4-class split); it yields
(x_flat, quad_means_8, gA4, y) and the model uses only x_flat.

Per-run artifacts mirror suitev2: config.json, metrics.csv, best_state.pt, test.json.
"""

import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from suitev2.config import BASELINE
from suitev2.data import load_data
from suitev2.train import set_seed
from suitev3 import config as V3
from suitev3.baselines import build_model


# ============================================================
# Loss + metrics
# ============================================================

def compute_loss(logits, y, train: bool):
    ls = BASELINE.LABEL_SMOOTHING if train else 0.0
    return F.cross_entropy(logits, y, label_smoothing=ls)


def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def confusion_matrix(model, loader, device, n_classes=V3.N_CLASSES):
    model.eval()
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    with torch.no_grad():
        for x_flat, _qm8, _gA4, y in loader:
            x_flat = x_flat.to(device)
            preds = model(x_flat).argmax(dim=1).cpu()
            for t, p in zip(y, preds):
                cm[int(t), int(p)] += 1
    return cm


# ============================================================
# One epoch
# ============================================================

def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = total_acc = total_n = 0.0
    for x_flat, _qm8, _gA4, y in loader:
        x_flat = x_flat.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(x_flat)
        loss = compute_loss(logits, y, train=is_train)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()

        bs = x_flat.shape[0]
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), y) * bs
        total_n += bs

    return total_loss / total_n, total_acc / total_n


# ============================================================
# Optimizer (single AdamW group over trainable params)
# ============================================================

def build_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=V3.LR, weight_decay=V3.WEIGHT_DECAY)


# ============================================================
# Frozen arm: cached-feature fast path
# ------------------------------------------------------------
# The frozen circuit produces IDENTICAL features at every epoch, so we run the
# quantum forward ONCE over train/val/test and then train only the head on the
# cached features. This is exact (not an approximation) and turns ~20 quantum
# passes over the data into 1, cutting the frozen arm from ~30 min to ~2 min.
# ============================================================

def _cache_quantum_features(model, loader, device):
    """Run the frozen quantum circuit once over a loader -> (X feats, y) on CPU."""
    model.eval()
    Xs, ys = [], []
    with torch.no_grad():
        for x_flat, _qm8, _gA4, y in loader:
            Xs.append(model.features(x_flat.to(device)).cpu())
            ys.append(y)
    return torch.cat(Xs), torch.cat(ys)


def _head_epoch(head, loader, device, optimizer=None):
    """One epoch of head-only training/eval on cached (X, y) batches."""
    is_train = optimizer is not None
    head.train() if is_train else head.eval()
    tot_loss = tot_acc = n = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if is_train:
            optimizer.zero_grad()
        logits = head(xb)
        loss = F.cross_entropy(
            logits, yb,
            label_smoothing=BASELINE.LABEL_SMOOTHING if is_train else 0.0)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        bs = xb.shape[0]
        tot_loss += loss.item() * bs
        tot_acc += (logits.argmax(1) == yb).float().sum().item()
        n += bs
    return tot_loss / n, tot_acc / n


def _confusion_head(head, loader, device, n_classes=V3.N_CLASSES):
    head.eval()
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    with torch.no_grad():
        for xb, yb in loader:
            preds = head(xb.to(device)).argmax(1).cpu()
            for t, p in zip(yb, preds):
                cm[int(t), int(p)] += 1
    return cm


def train_frozen_cached(seed, run_dir, epochs=None, train_per_class=None, log_fn=print):
    """Train the 'frozen' arm via precomputed (cached) quantum features.

    Produces the same artifacts/schema as train_one_run so it is a drop-in.
    """
    if epochs is None:
        epochs = BASELINE.EPOCHS

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model("frozen").to(device)   # frozen circuit + trainable head
    n_params = model.n_trainable_params()

    # Data (optional smoke override of TRAIN_PER_CLASS)
    if train_per_class is None:
        train_loader, val_loader, test_loader = load_data(seed=seed)
    else:
        orig = BASELINE.TRAIN_PER_CLASS
        BASELINE.TRAIN_PER_CLASS = train_per_class
        try:
            train_loader, val_loader, test_loader = load_data(seed=seed)
        finally:
            BASELINE.TRAIN_PER_CLASS = orig

    # --- precompute the fixed quantum feature map ONCE ---
    t_cache = time.time()
    Xtr, ytr = _cache_quantum_features(model, train_loader, device)
    Xva, yva = _cache_quantum_features(model, val_loader, device)
    Xte, yte = _cache_quantum_features(model, test_loader, device)
    log_fn(f"  cached frozen features in {time.time()-t_cache:.1f}s "
           f"(train {tuple(Xtr.shape)} val {tuple(Xva.shape)} test {tuple(Xte.shape)})")

    bs = BASELINE.BATCH_SIZE
    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=bs, shuffle=False)
    te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=bs, shuffle=False)

    head = model.head
    optimizer = torch.optim.AdamW(head.parameters(), lr=V3.LR, weight_decay=V3.WEIGHT_DECAY)
    T0 = max(1, epochs // 3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T0, T_mult=1, eta_min=1e-4)

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "arm": "frozen",
            "seed": seed,
            "epochs": epochs,
            "lr": V3.LR,
            "weight_decay": V3.WEIGHT_DECAY,
            "batch_size": bs,
            "label_smoothing": BASELINE.LABEL_SMOOTHING,
            "train_per_class": train_per_class or BASELINE.TRAIN_PER_CLASS,
            "val_per_class": BASELINE.VAL_PER_CLASS,
            "test_per_class": BASELINE.TEST_PER_CLASS,
            "n_readout": V3.N_READOUT,
            "conv": V3.CONV_NAME,
            "n_trainable_params": n_params,
            "cached_features": True,
        }, f, indent=2)

    metrics_f = open(os.path.join(run_dir, "metrics.csv"), "w", newline="")
    metrics_w = csv.writer(metrics_f)
    metrics_w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_sec"])
    metrics_f.flush()

    best_val_loss = float("inf")
    best_epoch = -1
    best_head_state = None
    val_acc_best = -1.0
    patience = 0

    try:
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = _head_epoch(head, tr_loader, device, optimizer)
            val_loss, val_acc = _head_epoch(head, va_loader, device, optimizer=None)
            scheduler.step()
            dt = time.time() - t0

            metrics_w.writerow([epoch,
                                f"{train_loss:.6f}", f"{train_acc:.6f}",
                                f"{val_loss:.6f}", f"{val_acc:.6f}", f"{dt:.2f}"])
            metrics_f.flush()
            os.fsync(metrics_f.fileno())

            min_impr = abs(best_val_loss) * BASELINE.EARLY_STOP_MIN_REL_DELTA
            if not np.isfinite(best_val_loss) or val_loss < best_val_loss - min_impr:
                best_val_loss = val_loss
                best_epoch = epoch
                best_head_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                val_acc_best = val_acc
                patience = 0
            else:
                patience += 1

            log_fn(f"  epoch {epoch:02d} | train {train_loss:.4f}/{train_acc:.4f} "
                   f"| val {val_loss:.4f}/{val_acc:.4f} | {dt:.1f}s")

            if patience >= BASELINE.EARLY_STOP_PATIENCE:
                log_fn(f"  early stop at epoch {epoch}")
                break
    finally:
        metrics_f.close()

    if best_head_state is not None:
        head.load_state_dict(best_head_state)
    # Save the FULL model (frozen circuit + best head) for reproducibility.
    torch.save({"model_state": model.state_dict()}, os.path.join(run_dir, "best_state.pt"))

    test_loss, test_acc = _head_epoch(head, te_loader, device, optimizer=None)
    cm = _confusion_head(head, te_loader, device)

    result = {
        "arm": "frozen",
        "seed": seed,
        "n_params": n_params,
        "n_epochs_trained": best_epoch if best_epoch > 0 else epochs,
        "best_epoch": best_epoch if best_epoch > 0 else epochs,
        "best_val_loss": best_val_loss,
        "val_acc_best": val_acc_best if val_acc_best >= 0 else 0.0,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(run_dir, "test.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# One run (arm, seed)
# ============================================================

def train_one_run(arm, seed, run_dir, epochs=None, train_per_class=None, log_fn=print):
    # Frozen arm uses the cached-feature fast path (exact, ~15x fewer quantum evals).
    if arm == "frozen":
        return train_frozen_cached(seed, run_dir, epochs=epochs,
                                   train_per_class=train_per_class, log_fn=log_fn)

    if epochs is None:
        epochs = BASELINE.EPOCHS

    set_seed(seed)   # seeds data-index shuffle AND model init (paired across arms)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(arm).to(device)
    n_params = model.n_trainable_params()

    # Data (optional smoke override of TRAIN_PER_CLASS)
    if train_per_class is None:
        train_loader, val_loader, test_loader = load_data(seed=seed)
    else:
        orig = BASELINE.TRAIN_PER_CLASS
        BASELINE.TRAIN_PER_CLASS = train_per_class
        try:
            train_loader, val_loader, test_loader = load_data(seed=seed)
        finally:
            BASELINE.TRAIN_PER_CLASS = orig

    optimizer = build_optimizer(model)
    T0 = max(1, epochs // 3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T0, T_mult=1, eta_min=1e-4)

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "arm": arm,
            "seed": seed,
            "epochs": epochs,
            "lr": V3.LR,
            "weight_decay": V3.WEIGHT_DECAY,
            "batch_size": BASELINE.BATCH_SIZE,
            "label_smoothing": BASELINE.LABEL_SMOOTHING,
            "train_per_class": train_per_class or BASELINE.TRAIN_PER_CLASS,
            "val_per_class": BASELINE.VAL_PER_CLASS,
            "test_per_class": BASELINE.TEST_PER_CLASS,
            "n_readout": V3.N_READOUT,
            "conv": V3.CONV_NAME,
            "n_trainable_params": n_params,
        }, f, indent=2)

    metrics_f = open(os.path.join(run_dir, "metrics.csv"), "w", newline="")
    metrics_w = csv.writer(metrics_f)
    metrics_w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_sec"])
    metrics_f.flush()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    val_acc_best = -1.0
    patience = 0

    try:
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None)
            scheduler.step()
            dt = time.time() - t0

            metrics_w.writerow([epoch,
                                f"{train_loss:.6f}", f"{train_acc:.6f}",
                                f"{val_loss:.6f}", f"{val_acc:.6f}", f"{dt:.2f}"])
            metrics_f.flush()
            os.fsync(metrics_f.fileno())

            min_impr = abs(best_val_loss) * BASELINE.EARLY_STOP_MIN_REL_DELTA
            if not np.isfinite(best_val_loss) or val_loss < best_val_loss - min_impr:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                val_acc_best = val_acc
                patience = 0
            else:
                patience += 1

            log_fn(f"  epoch {epoch:02d} | train {train_loss:.4f}/{train_acc:.4f} "
                   f"| val {val_loss:.4f}/{val_acc:.4f} | {dt:.1f}s")

            if patience >= BASELINE.EARLY_STOP_PATIENCE:
                log_fn(f"  early stop at epoch {epoch}")
                break
    finally:
        metrics_f.close()

    if best_state is not None:
        torch.save({"model_state": best_state}, os.path.join(run_dir, "best_state.pt"))
        model.load_state_dict(best_state)

    test_loss, test_acc = run_epoch(model, test_loader, device, optimizer=None)
    cm = confusion_matrix(model, test_loader, device)

    result = {
        "arm": arm,
        "seed": seed,
        "n_params": n_params,
        "n_epochs_trained": best_epoch if best_epoch > 0 else epochs,
        "best_epoch": best_epoch if best_epoch > 0 else epochs,
        "best_val_loss": best_val_loss,
        "val_acc_best": val_acc_best if val_acc_best >= 0 else 0.0,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(run_dir, "test.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result
