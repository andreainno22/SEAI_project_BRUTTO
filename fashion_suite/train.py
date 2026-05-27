"""
train.py — Loss, metrics, per-run training, training extension.

Each run saves:
  - config.json   : run hyperparameters
  - metrics.csv   : per-epoch train/val metrics (append, flushed each epoch)
  - best_state.pt : model.state_dict() at the best val_loss epoch
  - last_state.pt : model.state_dict() + optimizer.state_dict() + bookkeeping,
                    for resuming with extend_train()
  - test.json     : last test eval (overwritten on extension)

The training extension API (extend_training) loads last_state.pt and continues
from the saved epoch, appending to metrics.csv. It overwrites the checkpoints
and test.json with the latest values; history of test_acc across training
depths lives in summary.csv as separate rows (chunk_id='base', different
n_epochs_trained values).
"""

import csv
import json
import os
import random
import time

import numpy as np
import torch

from fashion_suite.config import BASELINE
from fashion_suite.data import load_data, load_test_extension
from fashion_suite.model import FashionQCNN


# ============================================================
# Reproducibility helper
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Loss + metrics
# ============================================================

def probability_cross_entropy_loss(probs, labels, eps=1e-8):
    """Cross-entropy directly on measurement probabilities (pulito86 baseline)."""
    probs = torch.clamp(probs, min=eps, max=1.0)
    true_probs = probs[
        torch.arange(probs.shape[0], device=probs.device),
        labels,
    ]
    return -torch.log(true_probs).mean()


def accuracy(probs, labels):
    preds = torch.argmax(probs, dim=1)
    return (preds == labels).float().mean().item()


def grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def confusion_matrix(model, loader, device, n_classes=BASELINE.N_CLASSES):
    """Return an (n_classes x n_classes) int tensor of (true, predicted) counts."""
    model.eval()
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    with torch.no_grad():
        for x_flat, qm8, gA4, y in loader:
            x_flat = x_flat.to(device)
            qm8    = qm8.to(device)
            gA4    = gA4.to(device)
            y      = y.to(device)

            probs = model(x_flat, quad_means_8=qm8, gA4=gA4)
            preds = torch.argmax(probs, dim=1)
            for t, p in zip(y.cpu(), preds.cpu()):
                cm[t, p] += 1
    return cm


# ============================================================
# Single epoch
# ============================================================

def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_grad_norm = 0.0
    total_n = 0
    n_batches = 0

    for x_flat, qm8, gA4, y in loader:
        x_flat = x_flat.to(device)
        qm8    = qm8.to(device)
        gA4    = gA4.to(device)
        y      = y.to(device)

        if is_train:
            optimizer.zero_grad()

        probs = model(x_flat, quad_means_8=qm8, gA4=gA4)
        loss = probability_cross_entropy_loss(probs, y)

        if is_train:
            loss.backward()
            total_grad_norm += grad_norm(model)
            n_batches += 1
            optimizer.step()

        bs = x_flat.shape[0]
        total_loss += loss.item() * bs
        total_acc  += accuracy(probs.detach(), y) * bs
        total_n    += bs

    avg_loss = total_loss / total_n
    avg_acc = total_acc / total_n
    avg_grad_norm = total_grad_norm / max(n_batches, 1) if is_train else None
    return avg_loss, avg_acc, avg_grad_norm


# ============================================================
# Inner training loop (shared by fresh runs and extensions)
# ============================================================

def _train_loop(model, optimizer, train_loader, val_loader, device,
                metrics_writer, metrics_file,
                start_epoch, n_epochs_to_run,
                best_val_loss_so_far, log_fn):
    """Run training for `n_epochs_to_run` epochs, starting from `start_epoch`.

    Appends rows to metrics.csv (one per epoch). Returns:
      (best_state_dict, best_epoch, best_val_loss,
       train_acc_final, val_acc_final, val_acc_best,
       last_epoch_idx)
    """
    best_state = None
    best_epoch = -1
    best_val_loss = best_val_loss_so_far
    val_acc_best = -1.0

    train_acc_final = 0.0
    val_acc_final = 0.0
    last_epoch_idx = start_epoch - 1

    for offset in range(n_epochs_to_run):
        epoch = start_epoch + offset
        t0 = time.time()
        train_loss, train_acc, train_gnorm = run_epoch(
            model, train_loader, device, optimizer=optimizer,
        )
        val_loss, val_acc, _ = run_epoch(model, val_loader, device, optimizer=None)
        dt = time.time() - t0

        metrics_writer.writerow([
            epoch,
            f"{train_loss:.6f}", f"{train_acc:.6f}",
            f"{val_loss:.6f}",   f"{val_acc:.6f}",
            f"{train_gnorm:.6e}",
            f"{dt:.2f}",
        ])
        metrics_file.flush()
        os.fsync(metrics_file.fileno())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            val_acc_best = val_acc

        train_acc_final = train_acc
        val_acc_final   = val_acc
        last_epoch_idx  = epoch

        log_fn(
            f"  epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"gnorm={train_gnorm:.3e} | {dt:.1f}s"
        )

    return (best_state, best_epoch, best_val_loss,
            train_acc_final, val_acc_final, val_acc_best,
            last_epoch_idx)


# ============================================================
# Helpers: save / load checkpoints
# ============================================================

def save_best_state(run_dir, best_state_dict):
    if best_state_dict is None:
        return
    torch.save({"model_state": best_state_dict},
               os.path.join(run_dir, "best_state.pt"))


def save_last_state(run_dir, model, optimizer, last_epoch, best_val_loss):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "last_epoch": last_epoch,
        "best_val_loss": best_val_loss,
    }, os.path.join(run_dir, "last_state.pt"))


def load_last_state(run_dir, model, optimizer):
    """Returns (last_epoch, best_val_loss). Modifies model/optimizer in-place."""
    path = os.path.join(run_dir, "last_state.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"last_state.pt not found in {run_dir!r}")
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt["last_epoch"]), float(ckpt["best_val_loss"])


def load_best_state(run_dir, model):
    path = os.path.join(run_dir, "best_state.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"best_state.pt not found in {run_dir!r}")
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model_state"])


# ============================================================
# Fresh run (base)
# ============================================================

def train_one_run(ansatz_name, encoding_name, seed, run_dir,
                  epochs=None, train_per_class=None, log_fn=print):
    """Train one (ansatz, encoding, seed) configuration from scratch.

    Writes config.json, metrics.csv (per-epoch), best_state.pt, last_state.pt,
    test.json. Returns a dict consumed by run_suite.py to write a summary.csv
    row.
    """
    if epochs is None:
        epochs = BASELINE.EPOCHS

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionQCNN(ansatz_name, encoding_name).to(device)
    n_params = model.n_trainable_params()

    # Data loaders (with optional smoke-test override)
    if train_per_class is None:
        train_loader, val_loader, test_loader = load_data(seed=seed)
    else:
        from fashion_suite.config import BASELINE as _B
        orig = _B.TRAIN_PER_CLASS
        _B.TRAIN_PER_CLASS = train_per_class
        try:
            train_loader, val_loader, test_loader = load_data(seed=seed)
        finally:
            _B.TRAIN_PER_CLASS = orig

    optimizer = torch.optim.Adam(model.parameters(), lr=BASELINE.LR)

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "ansatz": ansatz_name,
            "encoding": encoding_name,
            "seed": seed,
            "epochs_initial": epochs,
            "lr": BASELINE.LR,
            "batch_size": BASELINE.BATCH_SIZE,
            "train_per_class": train_per_class or BASELINE.TRAIN_PER_CLASS,
            "val_per_class":   BASELINE.VAL_PER_CLASS,
            "test_per_class":  BASELINE.TEST_PER_CLASS,
            "n_qubits": model.n_qubits,
            "n_conv_params": model.n_conv,
            "n_trainable_params": n_params,
        }, f, indent=2)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    metrics_f = open(metrics_path, "w", newline="")
    metrics_w = csv.writer(metrics_f)
    metrics_w.writerow([
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
        "grad_norm", "time_sec",
    ])
    metrics_f.flush()

    try:
        (best_state, best_epoch, best_val_loss,
         train_acc_final, val_acc_final, val_acc_best,
         last_epoch) = _train_loop(
            model, optimizer, train_loader, val_loader, device,
            metrics_w, metrics_f,
            start_epoch=1, n_epochs_to_run=epochs,
            best_val_loss_so_far=float("inf"),
            log_fn=log_fn,
        )
    finally:
        metrics_f.close()

    # Persist checkpoints
    save_best_state(run_dir, best_state)
    save_last_state(run_dir, model, optimizer, last_epoch, best_val_loss)

    # Test eval uses the best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, _ = run_epoch(model, test_loader, device, optimizer=None)
    cm = confusion_matrix(model, test_loader, device)
    test_size = sum(sum(row) for row in cm.tolist())

    result = {
        "ansatz": ansatz_name,
        "encoding": encoding_name,
        "seed": seed,
        "n_params": n_params,
        "n_epochs_trained": last_epoch,
        "chunk_id": "base",
        "test_offset": 0,
        "test_size": test_size,
        "train_acc_final": train_acc_final,
        "val_acc_best":    val_acc_best if val_acc_best >= 0 else val_acc_final,
        "val_acc_final":   val_acc_final,
        "best_epoch":      best_epoch if best_epoch > 0 else last_epoch,
        "best_val_loss":   best_val_loss,
        "test_loss":       test_loss,
        "test_acc":        test_acc,
        "confusion_matrix": cm.tolist(),
    }

    with open(os.path.join(run_dir, "test.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# Extend training (resume)
# ============================================================

def extend_training(run_dir, extra_epochs, log_fn=print):
    """Resume an existing run for `extra_epochs` additional epochs.

    Reads run_dir/config.json to recover (ansatz, encoding, seed). Loads
    last_state.pt and resumes from there, appending to metrics.csv.
    Re-evaluates on the base test chunk after extension. Overwrites the
    checkpoints and test.json with the latest values.

    Returns a fresh result dict (to be appended as a new row in summary.csv).
    """
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {run_dir!r}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    ansatz_name = cfg["ansatz"]
    encoding_name = cfg["encoding"]
    seed = cfg["seed"]
    train_per_class = cfg.get("train_per_class", BASELINE.TRAIN_PER_CLASS)

    set_seed(seed)  # re-seed for reproducibility within the loaded indices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild model + optimizer, then load state
    model = FashionQCNN(ansatz_name, encoding_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BASELINE.LR)
    last_epoch_done, best_val_loss = load_last_state(run_dir, model, optimizer)

    # Override TRAIN_PER_CLASS if the original config used a smoke value
    from fashion_suite.config import BASELINE as _B
    orig_tpc = _B.TRAIN_PER_CLASS
    _B.TRAIN_PER_CLASS = train_per_class
    try:
        train_loader, val_loader, test_loader = load_data(seed=seed)
    finally:
        _B.TRAIN_PER_CLASS = orig_tpc

    # Append to metrics.csv (re-open in append mode)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    metrics_f = open(metrics_path, "a", newline="")
    metrics_w = csv.writer(metrics_f)

    log_fn(f"  RESUMING from epoch {last_epoch_done} for {extra_epochs} more epochs "
           f"(best_val_loss so far = {best_val_loss:.4f})")

    try:
        (best_state_new, best_epoch_new, best_val_loss_new,
         train_acc_final, val_acc_final, val_acc_best,
         last_epoch_new) = _train_loop(
            model, optimizer, train_loader, val_loader, device,
            metrics_w, metrics_f,
            start_epoch=last_epoch_done + 1,
            n_epochs_to_run=extra_epochs,
            best_val_loss_so_far=best_val_loss,
            log_fn=log_fn,
        )
    finally:
        metrics_f.close()

    # If a new best was found during extension, persist it; otherwise keep old
    if best_state_new is not None:
        save_best_state(run_dir, best_state_new)
    save_last_state(run_dir, model, optimizer, last_epoch_new, best_val_loss_new)

    # Test eval on base chunk using the (possibly updated) best state
    load_best_state(run_dir, model)
    test_loss, test_acc, _ = run_epoch(model, test_loader, device, optimizer=None)
    cm = confusion_matrix(model, test_loader, device)
    test_size = sum(sum(row) for row in cm.tolist())

    # Pull n_params from config (model hasn't changed shape)
    n_params = cfg.get("n_trainable_params", model.n_trainable_params())

    result = {
        "ansatz": ansatz_name,
        "encoding": encoding_name,
        "seed": seed,
        "n_params": n_params,
        "n_epochs_trained": last_epoch_new,
        "chunk_id": "base",
        "test_offset": 0,
        "test_size": test_size,
        "train_acc_final": train_acc_final,
        "val_acc_best":    val_acc_best if val_acc_best >= 0 else val_acc_final,
        "val_acc_final":   val_acc_final,
        "best_epoch":      best_epoch_new if best_epoch_new > 0 else cfg.get("best_epoch", last_epoch_done),
        "best_val_loss":   best_val_loss_new,
        "test_loss":       test_loss,
        "test_acc":        test_acc,
        "confusion_matrix": cm.tolist(),
    }

    # Overwrite test.json (latest snapshot for this run state)
    with open(os.path.join(run_dir, "test.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# Evaluate on a test extension chunk (no training)
# ============================================================

def evaluate_test_chunk(run_dir, extra_per_class, chunk_id, log_fn=print):
    """Evaluate the best-state model on a NEW test chunk (offset = base + previous extensions).

    The offset is automatically tracked: the chunk starts at `test_per_class`
    used in the base config (no support for arbitrary offsets — chunks must
    be contiguous, added sequentially).

    Caller (extend_run.py) is responsible for choosing chunk_id consistently
    across runs and for incrementing the offset between extensions.

    Returns a result dict for summary.csv. Does NOT modify checkpoints,
    config.json, or test.json (the base test snapshot stays intact).
    """
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {run_dir!r}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    ansatz_name = cfg["ansatz"]
    encoding_name = cfg["encoding"]
    seed = cfg["seed"]
    base_test_per_class = cfg.get("test_per_class", BASELINE.TEST_PER_CLASS)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FashionQCNN(ansatz_name, encoding_name).to(device)
    load_best_state(run_dir, model)

    # Build extension test loader (offset = base + previously-used extension)
    # extend_run.py passes the correct offset via env or arg; we expose it here
    # in the function args. For simplicity, evaluate_test_chunk receives the
    # offset directly so it doesn't have to re-scan summary.csv.
    raise NotImplementedError("Use evaluate_test_chunk_at_offset() instead")


def evaluate_test_chunk_at_offset(run_dir, offset_per_class, extra_per_class,
                                   chunk_id, log_fn=print):
    """Evaluate the best-state model on a NEW disjoint test chunk.

    Args:
      run_dir:          existing run directory.
      offset_per_class: where the new chunk starts (in per-class index space).
                        Caller computes this from summary.csv history.
      extra_per_class:  size of the new chunk per class.
      chunk_id:         label for this chunk (e.g., 'ext_1').
    Returns:
      result dict (to be appended to summary.csv).
    """
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {run_dir!r}")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    ansatz_name = cfg["ansatz"]
    encoding_name = cfg["encoding"]
    seed = cfg["seed"]
    n_params = cfg.get("n_trainable_params")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FashionQCNN(ansatz_name, encoding_name).to(device)
    load_best_state(run_dir, model)

    test_loader = load_test_extension(
        offset_per_class=offset_per_class,
        n_per_class=extra_per_class,
        seed=seed,
    )

    test_loss, test_acc, _ = run_epoch(model, test_loader, device, optimizer=None)
    cm = confusion_matrix(model, test_loader, device)
    test_size = sum(sum(row) for row in cm.tolist())

    # Recover bookkeeping fields from the most recent test.json (best snapshot)
    test_json = os.path.join(run_dir, "test.json")
    base_meta = {}
    if os.path.exists(test_json):
        with open(test_json, "r") as f:
            base_meta = json.load(f)

    return {
        "ansatz": ansatz_name,
        "encoding": encoding_name,
        "seed": seed,
        "n_params": n_params or 0,
        "n_epochs_trained": base_meta.get("n_epochs_trained", -1),
        "chunk_id": chunk_id,
        "test_offset": offset_per_class * BASELINE.N_CLASSES,
        "test_size": test_size,
        "train_acc_final": base_meta.get("train_acc_final", 0.0),
        "val_acc_best":    base_meta.get("val_acc_best",    0.0),
        "val_acc_final":   base_meta.get("val_acc_final",   0.0),
        "best_epoch":      base_meta.get("best_epoch",      0),
        "best_val_loss":   base_meta.get("best_val_loss",   0.0),
        "test_loss":       test_loss,
        "test_acc":        test_acc,
        "confusion_matrix": cm.tolist(),
    }
