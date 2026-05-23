"""
Fashion-MNIST QCNN with patch-wise amplitude encoding — v2.2

Pipeline:
    28x28 image
    -> zero-padded to 32x32 (transforms.Pad, no bilinear interpolation)
    -> split into four 16x16 patches
    -> amplitude encoding of each patch on 8 qubits
    -> patch L2 norm injected as RY rotation on wire 0 (recovers inter-patch contrast)
    -> conv8 -> pool_8_to_4 (CNOT-sandwich) -> conv4_retained
    -> 4 Z-expvals + 4 ZZ-correlations per patch = 8 features/patch
    -> concatenate 4 x 8 = 32 quantum features
    -> classical head: LayerNorm(32) + Linear(32 -> 10)
    -> 10 Fashion-MNIST logits

Experiments:
    trainable_qcnn : all 32 quantum params + head trained end-to-end
    frozen_qcnn    : quantum params frozen at init, only head trained (baseline)

Changes from v1 (fashion_patch_amplitude_qcnn_32_frozen_diagnostics.py):
    C1. Fix pooling gate  : CNOT-sandwich RY+CNOT+RZ+CNOT (CRX/CRZ not adjoint-differentiable)
    C2. Deeper circuit    : second conv layer on retained wires (8 params in theta_q[24:32])
    C3. Extended readout  : 4 Z + 4 ZZ correlations = 8 features/patch, HEAD_IN_DIM=32
    C4. Near-zero init    : 1e-3*randn for theta_q (zeros caused circuit symmetry deadlock)
    C5. LR rebalancing    : LR_QKERNEL 3e-4->1e-3, LR_HEAD 1e-3->5e-4
    C6. Cosine scheduler  : CosineAnnealingLR instead of ReduceLROnPlateau
    C7. Longer training   : EPOCHS=60, EARLY_STOP_PATIENCE=10
    C8. Transform fix     : Pad(2) instead of Resize; train-only RandomHorizontalFlip
    C9. Norm injection    : patch L2 norm injected into circuit via RY on wire 0

Post-run fixes (v2.1, after observing phi_pool_drift=0 and th1==th2 in first run):
    B1. CRX/CRZ -> CNOT-sandwich: adjoint differentiation gave zero gradient for
        CRX and CRZ gates; replaced with RY+CNOT+RZ+CNOT (all adjoint-differentiable).
    B2. Single theta_q tensor: passing three separate tensors to the QNode caused
        gradient index misalignment in PennyLane adjoint; unified into theta_q[32]
        with slices SL_CONV/SL_POOL/SL_CONV2 defined outside and applied inside QNode.
    B3. 1e-3*randn init: exact zero init creates a circuit symmetry where conv8 RY
        and pooling RY gates receive identical gradients indefinitely; small random
        noise breaks this symmetry while keeping angles near-identity.
    Diag. Per-segment gradient norms (grad_theta_conv, grad_phi_pool, grad_theta_conv2)
          added to metrics.csv print and CSV for monitoring learning per layer.

Tuning (v2.2, 2026-05-23):
    T1. BASE_DIR -> v2_2: separate output directory to preserve v2.1 results.
    T2. TRAIN_SAMPLES 10k -> 20k: v2.1 underfit (train_loss ≈ val_loss ≈ 0.86 at epoch 60).
        More data is the highest-leverage lever without changing architecture.
    T3. WEIGHT_DECAY_HEAD 1e-3 -> 1e-4: head (330 params) was over-regularized at 10k scale;
        with more data, less regularization is appropriate.
    T4. EPOCHS 60 -> 80, EARLY_STOP_PATIENCE 10 -> 15: 4 warm-restart cycles of 20 epochs;
        larger patience prevents early stopping immediately after a LR restart.
    T5. CosineAnnealingWarmRestarts(T_0=20, T_mult=1) replacing CosineAnnealingLR:
        plain cosine decayed LR to ~10% by epoch 40, causing hard plateau before convergence;
        warm restarts give 4 independent learning phases over 80 epochs.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if "OMP_NUM_THREADS" not in os.environ:
    cores = os.cpu_count() or 4
    os.environ["OMP_NUM_THREADS"] = str(cores)

import json
import time
import random
from typing import Dict, List, Tuple

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
    "BASE_DIR":              "./fashion_patch_amplitude_qcnn_32_v2_2",  # T1: new output dir
    "SEED":                  0,
    "RUNS": [
        {"run_name": "trainable_qcnn", "freeze_qcnn": False},
        {"run_name": "frozen_qcnn",    "freeze_qcnn": True},
    ],
    "TRAIN_SAMPLES":         20000,   # T2: was 10000 (20k = 2x more data)
    "VAL_SAMPLES":            2000,
    "TEST_SAMPLES":           2000,
    "EPOCHS":                   80,   # T4: was 60 (4 warm-restart cycles at T_0=20)
    "BATCH_SIZE":              128,
    "EARLY_STOP_PATIENCE":      15,   # T4: was 10 (survive LR restarts without early stop)
    "PRINT_EVERY":               1,
    "LR_HEAD":               5e-4,
    "LR_QKERNEL":            1e-3,
    "WEIGHT_DECAY_HEAD":     1e-4,    # T3: was 1e-3 (less aggressive head regularization)
    "WEIGHT_DECAY_QKERNEL":  1e-5,
    "CLIP_NORM":              1.0,
}

N_CLASSES           = 10
N_QUBITS            = 8
N_PATCHES           = 4
PATCH_SIZE          = 16
PATCH_PIXELS        = PATCH_SIZE * PATCH_SIZE           # 256 = 2^8
KEEP_WIRES          = [0, 2, 4, 6]
ZZ_PAIRS            = [(0, 2), (2, 4), (4, 6), (6, 0)]  # C3: ring ZZ among KEEP_WIRES
QFEATURES_PER_PATCH = len(KEEP_WIRES) + len(ZZ_PAIRS)   # C3: 4 Z + 4 ZZ = 8
HEAD_IN_DIM         = N_PATCHES * QFEATURES_PER_PATCH    # C3: 32
EPS                 = 1e-8

# Slice indices for theta_q (single quantum parameter tensor, shape 32).
# Passing three separate tensors to the QNode causes PennyLane adjoint to
# alias gradients across them; one tensor avoids the aliasing entirely.
SL_CONV  = slice(0,  16)   # theta_conv  (conv8,          16 params)
SL_POOL  = slice(16, 24)   # phi_pool    (pooling,         8 params)
SL_CONV2 = slice(24, 32)   # theta_conv2 (conv4_retained,  8 params)
N_THETA_Q = 32

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

# C8: Pad(2) adds 2 black pixels on each side: 28x28 -> 32x32 with no interpolation.
# v1 used Resize (bilinear) which distorted pixel intensities; Pad preserves them.
# Train transform includes RandomHorizontalFlip; eval/test does not.
transform_eval = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),   # C8: train-only augmentation
    transforms.Pad(2),
    transforms.ToTensor(),
])


def get_datasets():
    os.makedirs("./data", exist_ok=True)
    train_dataset = torchvision.datasets.FashionMNIST(
        "./data", train=True,  download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        "./data", train=False, download=True, transform=transform_eval
    )
    return train_dataset, test_dataset


def stratified_split_indices(dataset_list, n_train: int, n_val: int, seed: int):
    """Non-overlapping, class-balanced train and validation subsets."""
    rng     = random.Random(seed)
    buckets = {c: [] for c in range(N_CLASSES)}

    for i, (_, y) in enumerate(dataset_list):
        buckets[int(y)].append(i)

    def per_class_quota(total: int) -> List[int]:
        base = total // N_CLASSES
        rem  = total % N_CLASSES
        return [base + (1 if c < rem else 0) for c in range(N_CLASSES)]

    q_train = per_class_quota(n_train)
    q_val   = per_class_quota(n_val)

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
    """Class-balanced subset, used here for test sampling."""
    rng     = random.Random(seed)
    buckets = {c: [] for c in range(N_CLASSES)}

    for i, (_, y) in enumerate(dataset_list):
        buckets[int(y)].append(i)

    base = n_samples // N_CLASSES
    rem  = n_samples % N_CLASSES

    idxs = []
    for c in range(N_CLASSES):
        take = min(base + (1 if c < rem else 0), len(buckets[c]))
        idxs.extend(rng.sample(buckets[c], take))

    rng.shuffle(idxs)
    return idxs


# ============================================================
# Feature extraction
# ============================================================

TRAIN_CACHE: Dict[int, Dict[str, torch.Tensor]] = {}
TEST_CACHE:  Dict[int, Dict[str, torch.Tensor]] = {}


def image_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    return img_tensor.squeeze(0).cpu().numpy().astype(np.float32)


def split_patches_16x16(X: np.ndarray) -> List[np.ndarray]:
    """32x32 image -> four 16x16 patches: TL, TR, BL, BR."""
    return [
        X[0:16,  0:16],
        X[0:16,  16:32],
        X[16:32, 0:16],
        X[16:32, 16:32],
    ]


def patch_amplitudes_4x256(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """32x32 image -> 4 unit-norm amplitude vectors + relative patch-norm angles.

    C9: per-patch normalization loses inter-patch contrast (a dark patch and a
    bright patch both become unit vectors). Raw L2 norms are preserved as
    norm_angles in [0, pi], proportional to each patch share of total image L2
    energy, and injected into the quantum circuit via RY on wire 0.

    Returns:
        amps        : (4, 256) float32, one unit-norm row per patch
        norm_angles : (4,)    float32, values in [0, pi]
    """
    amps, raw_norms = [], []
    for patch in split_patches_16x16(X):
        amp  = patch.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(amp))
        raw_norms.append(norm)
        if norm < EPS:
            amp    = np.zeros(PATCH_PIXELS, dtype=np.float32)
            amp[0] = 1.0
        else:
            amp = amp / norm
        amps.append(amp)

    total       = float(sum(raw_norms)) + EPS
    norm_angles = np.array(
        [np.pi * n / total for n in raw_norms], dtype=np.float32
    )
    return np.stack(amps, axis=0), norm_angles  # (4, 256), (4,)


def get_features(dataset_list, cache: Dict[int, Dict[str, torch.Tensor]], idx: int):
    if idx in cache:
        return cache[idx]

    x, y            = dataset_list[idx]
    X               = image_to_numpy(x)
    amp4x256, norms = patch_amplitudes_4x256(X)

    sample = {
        "amp4x256":    torch.tensor(amp4x256, dtype=torch.float32),
        "norm_angles": torch.tensor(norms,    dtype=torch.float32),  # C9
        "y":           int(y),
    }
    cache[idx] = sample
    return sample


class QCNNDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list, cache):
        self.dataset_list = dataset_list
        self.cache        = cache

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        return get_features(self.dataset_list, self.cache, idx)


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
    """8-qubit conv: 8 RY + 4 even-pair CNOT-RZ + 4 shifted CNOT-RZ. 16 params."""
    for i in range(N_QUBITS):
        qml.RY(theta_conv[i], wires=i)
    for a, b, k in [(0, 1, 8), (2, 3, 9), (4, 5, 10), (6, 7, 11)]:
        qml.CNOT(wires=[a, b])
        qml.RZ(theta_conv[k], wires=b)
        qml.CNOT(wires=[a, b])
    for a, b, k in [(1, 2, 12), (3, 4, 13), (5, 6, 14), (7, 0, 15)]:
        qml.CNOT(wires=[a, b])
        qml.RZ(theta_conv[k], wires=b)
        qml.CNOT(wires=[a, b])


def hierarchical_pool_8_to_4(phi_pool):
    """Hierarchical pooling 8 -> 4 retained qubits [0, 2, 4, 6]. 8 params.

    C1 fix: replaced CRX+CRZ (zero gradient via lightning.qubit adjoint) with
    a CNOT-sandwich: RY(target) -> CNOT(ctrl,tgt) -> RZ(target) -> CNOT(ctrl,tgt).
    Information transfers from control to target via CNOT entanglement;
    RY and RZ are fully differentiable via adjoint.
    """
    pairs = [(1, 0), (3, 2), (5, 4), (7, 6)]
    for j, (control, target) in enumerate(pairs):
        qml.RY(phi_pool[2 * j],     wires=target)
        qml.CNOT(wires=[control, target])
        qml.RZ(phi_pool[2 * j + 1], wires=target)
        qml.CNOT(wires=[control, target])


def conv4_retained(theta_conv2):
    """Second conv on retained wires [0, 2, 4, 6] after pooling. 8 params.

    4 RY rotations + 4 CNOT-RZ entangling pairs forming a ring:
    (0->2), (2->4), (4->6), (6->0).

    C2: deepens the quantum circuit without increasing classical complexity.
    """
    for k, w in enumerate(KEEP_WIRES):
        qml.RY(theta_conv2[k], wires=w)
    for k in range(len(KEEP_WIRES) - 1):
        a, b = KEEP_WIRES[k], KEEP_WIRES[k + 1]
        qml.CNOT(wires=[a, b])
        qml.RZ(theta_conv2[4 + k], wires=b)
        qml.CNOT(wires=[a, b])
    # closing the ring: wire 6 -> wire 0
    qml.CNOT(wires=[KEEP_WIRES[-1], KEEP_WIRES[0]])
    qml.RZ(theta_conv2[7], wires=KEEP_WIRES[0])
    qml.CNOT(wires=[KEEP_WIRES[-1], KEEP_WIRES[0]])


def readout_features():
    """8 observables per patch: 4 local PauliZ + 4 ZZ correlations on KEEP_WIRES ring.

    C3: ZZ correlations expose two-qubit entanglement to the head, carrying
    information that single-qubit Z expectation values cannot capture.
    """
    z_vals  = [qml.expval(qml.PauliZ(i)) for i in KEEP_WIRES]
    zz_vals = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in ZZ_PAIRS]
    return z_vals + zz_vals


@qml.qnode(dev8, **QNODE_KW)
def patch_amplitude_qnode(amp256, norm_angle, theta_q):
    """Single trainable tensor theta_q avoids PennyLane adjoint gradient aliasing.

    Slicing inside the QNode: PennyLane differentiates one 32-element tensor and
    returns one gradient vector with correct independent values per slot.
      theta_q[SL_CONV]  = theta_conv  (conv8,         indices 0..15)
      theta_q[SL_POOL]  = phi_pool    (pooling,        indices 16..23)
      theta_q[SL_CONV2] = theta_conv2 (conv4_retained, indices 24..31)
    """
    qml.AmplitudeEmbedding(amp256, wires=range(N_QUBITS), normalize=True)
    qml.RY(norm_angle, wires=0)        # C9: inject relative patch brightness
    conv8(theta_q[SL_CONV])
    hierarchical_pool_8_to_4(theta_q[SL_POOL])
    conv4_retained(theta_q[SL_CONV2])  # C2
    return readout_features()


# ============================================================
# Model
# ============================================================

class PatchAmplitudeQCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # C4 rev: 1e-3 * randn instead of exact zeros.
        # Exact zeros cause a circuit symmetry at init: conv8 RY params and pooling RY
        # params operate on the same qubits with identical circuit context (all other
        # gates are identity), so they receive identical gradients and remain equal
        # indefinitely. A small random init (scale 1e-3, angles << 1 rad -> near-identity
        # gates) breaks this symmetry while keeping barren-plateau risk negligible.
        # Reproducible because set_global_seed(seed) is called before model creation.
        self.theta_q = torch.nn.Parameter(1e-3 * torch.randn(N_THETA_Q, dtype=torch.float32))

        # C3: HEAD_IN_DIM = 32 (4 patches x 8 features/patch)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(HEAD_IN_DIM),
            torch.nn.Linear(HEAD_IN_DIM, N_CLASSES),
        )

    def quantum_features(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        amp4x256    = sample["amp4x256"]
        norm_angles = sample["norm_angles"]

        if amp4x256.dim() == 2:          # single sample: (4,256) -> (1,4,256)
            amp4x256    = amp4x256.unsqueeze(0)
            norm_angles = norm_angles.unsqueeze(0)

        patch_features = []
        for p in range(N_PATCHES):
            out = patch_amplitude_qnode(
                amp4x256[:, p, :],   # (B, 256)
                norm_angles[:, p],   # (B,)   C9
                self.theta_q,        # (32,)  single quantum tensor
            )
            patch_features.append(torch.stack(out, dim=1).to(dtype=torch.float32))

        return torch.cat(patch_features, dim=1)  # (B, 32)

    def forward(self, sample: Dict[str, torch.Tensor]):
        features = self.quantum_features(sample)
        logits   = self.head(features)
        return logits, features


# ============================================================
# Training and evaluation
# ============================================================

ce_loss = torch.nn.CrossEntropyLoss()


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p is None or p.grad is None:
            continue
        total += float(p.grad.detach().norm().item() ** 2)
    return float(total ** 0.5)


def grad_norm_slice(param: torch.nn.Parameter, sl: slice) -> float:
    """L2 norm of a gradient slice (returns 0.0 if grad is None)."""
    if param.grad is None:
        return 0.0
    return float(param.grad[sl].norm().item())


def model_grad_norms(model: PatchAmplitudeQCNN) -> Dict[str, float]:
    """Gradient norms for the quantum kernel (theta_q and its segments) and head."""
    qkernel_norm = grad_norm([model.theta_q])
    head_norm    = grad_norm(model.head.parameters())
    ratio        = qkernel_norm / (head_norm + 1e-12)
    return {
        "grad_qkernel":           qkernel_norm,
        "grad_head":              head_norm,
        "grad_qkernel_over_head": ratio,
        "grad_theta_conv":        grad_norm_slice(model.theta_q, SL_CONV),
        "grad_phi_pool":          grad_norm_slice(model.theta_q, SL_POOL),
        "grad_theta_conv2":       grad_norm_slice(model.theta_q, SL_CONV2),
    }


def feature_separability_diagnostics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Between/within class variance ratio and mean inter-class center distance."""
    empty = {
        "feature_std_mean": 0.0, "feature_var_mean": 0.0,
        "feature_between_var_mean": 0.0, "feature_within_var_mean": 0.0,
        "feature_between_within_ratio": 0.0, "feature_class_center_distance_mean": 0.0,
    }
    if features.size == 0:
        return empty

    F           = features.astype(np.float64)
    y           = labels.astype(np.int64)
    global_mean = F.mean(axis=0)
    global_var  = F.var(axis=0)
    global_std  = F.std(axis=0)

    class_means, within_vars = [], []
    for c in range(N_CLASSES):
        Fc = F[y == c]
        if len(Fc) == 0:
            continue
        class_means.append(Fc.mean(axis=0))
        within_vars.append(Fc.var(axis=0).mean())

    if not class_means:
        return {**empty, "feature_std_mean": float(global_std.mean()),
                "feature_var_mean": float(global_var.mean())}

    C                = np.stack(class_means, axis=0)
    between_var_mean = float(((C - global_mean) ** 2).mean())
    within_var_mean  = float(np.mean(within_vars))

    dists = [
        float(np.linalg.norm(C[i] - C[j]))
        for i in range(len(C))
        for j in range(i + 1, len(C))
    ]
    center_distance_mean = float(np.mean(dists)) if dists else 0.0

    return {
        "feature_std_mean":                   float(global_std.mean()),
        "feature_var_mean":                   float(global_var.mean()),
        "feature_between_var_mean":           between_var_mean,
        "feature_within_var_mean":            within_var_mean,
        "feature_between_within_ratio":       float(between_var_mean / (within_var_mean + 1e-12)),
        "feature_class_center_distance_mean": center_distance_mean,
    }


@torch.no_grad()
def evaluate(model, dataset_list, cache, indices, with_feature_diagnostics: bool = False):
    model.eval()
    losses, correct, total     = [], 0, 0
    features_list, labels_list = [], []

    subset = torch.utils.data.Subset(QCNNDataset(dataset_list, cache), indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=CONFIG.get("BATCH_SIZE", 32), shuffle=False
    )

    for batch in loader:
        logits, features = model(batch)
        y    = batch["y"]
        loss = ce_loss(logits, y)

        preds    = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item())
        total   += len(y)
        losses.append(float(loss.item()) * len(y))

        if with_feature_diagnostics:
            features_list.append(features.detach().cpu().numpy())
            labels_list.append(y.cpu().numpy())

    metrics = {
        "loss": float(np.sum(losses) / total) if total else 0.0,
        "acc":  float(correct / total)         if total else 0.0,
    }

    if with_feature_diagnostics:
        F = np.concatenate(features_list, axis=0) if features_list else np.empty((0, HEAD_IN_DIM))
        y = np.concatenate(labels_list,   axis=0) if labels_list   else np.empty((0,), dtype=np.int64)
        metrics.update(feature_separability_diagnostics(F, y))

    return metrics


def build_optimizer(model: PatchAmplitudeQCNN):
    param_groups = []

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    if head_params:
        param_groups.append({
            "params":       head_params,
            "lr":           CONFIG["LR_HEAD"],
            "weight_decay": CONFIG["WEIGHT_DECAY_HEAD"],
        })

    if model.theta_q.requires_grad:
        param_groups.append({
            "params":       [model.theta_q],
            "lr":           CONFIG["LR_QKERNEL"],
            "weight_decay": CONFIG["WEIGHT_DECAY_QKERNEL"],
        })

    if not param_groups:
        raise ValueError("No trainable parameters found.")

    return torch.optim.AdamW(param_groups)


def train_one_run(run_name: str, freeze_qcnn: bool, seed: int, train_data, test_data):
    run_dir = os.path.join(BASE_DIR, f"{run_name}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    set_global_seed(seed)

    train_idx, val_idx = stratified_split_indices(
        train_data, CONFIG["TRAIN_SAMPLES"], CONFIG["VAL_SAMPLES"], seed=seed + 101
    )
    test_idx = stratified_indices(test_data, CONFIG["TEST_SAMPLES"], seed=seed + 303)

    model         = PatchAmplitudeQCNN()
    theta_q_init  = model.theta_q.detach().clone()    # zeros(32)

    if freeze_qcnn:
        model.theta_q.requires_grad_(False)

    optimizer = build_optimizer(model)

    # T5: CosineAnnealingWarmRestarts — LR resets every T_0=20 epochs.
    # v2.1 plain cosine (T_max=60) decayed LR to ~10% by epoch 40, causing a hard
    # plateau before the model converged. With T_0=20 over 80 epochs we get 4 independent
    # learning phases; each restart lets the model escape local minima and resume learning.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-6
    )

    print("\n" + "=" * 90)
    print(f"RUN={run_name} | seed={seed} | freeze_qcnn={freeze_qcnn}")
    print(f"Quantum device: {dev8_name} ({diff8})")
    print(f"Input  : 28x28 -> zero-padded 32x32 (Pad, no interpolation) -> four 16x16 patches")
    print(f"Encode : AmplitudeEmbedding(8 qubits) + patch-norm RY on wire 0 (C9)")
    print(f"Circuit: conv8[16p] -> pool_8_to_4 CNOT-sandwich[8p] (B1 fix) -> conv4_retained[8p] (C2)")
    print(f"Readout: {len(KEEP_WIRES)} Z + {len(ZZ_PAIRS)} ZZ = {QFEATURES_PER_PATCH} feat/patch "
          f"x {N_PATCHES} patches = {HEAD_IN_DIM} total (C3)")
    print(f"Head   : LayerNorm({HEAD_IN_DIM}) + Linear({HEAD_IN_DIM}->{N_CLASSES})")
    print(f"Trainable params: {count_trainable_params(model)}")
    print(f"Train/Val/Test  : {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    rows          = []
    best_val_loss = float("inf")
    best_state    = None
    patience      = 0

    train_subset = torch.utils.data.Subset(QCNNDataset(train_data, TRAIN_CACHE), train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=CONFIG.get("BATCH_SIZE", 32), shuffle=True
    )

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        t0 = time.time()
        model.train()

        running_loss         = 0.0
        grad_qkernel_values  = []
        grad_head_values     = []
        grad_ratio_values    = []
        grad_conv_values     = []
        grad_pool_values     = []
        grad_conv2_values    = []

        for batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model(batch)
            y         = batch["y"]
            loss      = ce_loss(logits, y)

            loss.backward()
            running_loss += float(loss.item()) * len(y)

            norms = model_grad_norms(model)
            grad_qkernel_values.append(norms["grad_qkernel"])
            grad_head_values.append(norms["grad_head"])
            grad_ratio_values.append(norms["grad_qkernel_over_head"])
            grad_conv_values.append(norms["grad_theta_conv"])
            grad_pool_values.append(norms["grad_phi_pool"])
            grad_conv2_values.append(norms["grad_theta_conv2"])

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                CONFIG["CLIP_NORM"],
            )
            optimizer.step()

        train_loss = running_loss / max(1, len(train_idx))

        val_metrics  = evaluate(model, train_data, TRAIN_CACHE, val_idx,  with_feature_diagnostics=True)
        test_metrics = evaluate(model, test_data,  TEST_CACHE,  test_idx)

        scheduler.step()  # C6: cosine step — no val-loss argument

        grad_qkernel_mean = float(np.mean(grad_qkernel_values)) if grad_qkernel_values else 0.0
        grad_head_mean    = float(np.mean(grad_head_values))    if grad_head_values    else 0.0
        grad_ratio_mean   = float(np.mean(grad_ratio_values))   if grad_ratio_values   else 0.0
        grad_conv_mean    = float(np.mean(grad_conv_values))    if grad_conv_values    else 0.0
        grad_pool_mean    = float(np.mean(grad_pool_values))    if grad_pool_values    else 0.0
        grad_conv2_mean   = float(np.mean(grad_conv2_values))   if grad_conv2_values   else 0.0
        tq = model.theta_q.detach()
        theta_conv_drift  = float(torch.norm(tq[SL_CONV]  - theta_q_init[SL_CONV]).item())
        phi_pool_drift    = float(torch.norm(tq[SL_POOL]  - theta_q_init[SL_POOL]).item())
        theta_conv2_drift = float(torch.norm(tq[SL_CONV2] - theta_q_init[SL_CONV2]).item())

        row = {
            "run_name":                            run_name,
            "freeze_qcnn":                         bool(freeze_qcnn),
            "seed":                                seed,
            "epoch":                               epoch,
            "nparams_trainable":                   count_trainable_params(model),
            "n_patch_convolutions_per_image":      N_PATCHES,
            "grad_qkernel_mean":                   grad_qkernel_mean,
            "grad_head_mean":                      grad_head_mean,
            "grad_qkernel_over_head_mean":         grad_ratio_mean,
            "grad_theta_conv_mean":                grad_conv_mean,
            "grad_phi_pool_mean":                  grad_pool_mean,
            "grad_theta_conv2_mean":               grad_conv2_mean,
            "theta_conv_drift":                    theta_conv_drift,
            "phi_pool_drift":                      phi_pool_drift,
            "theta_conv2_drift":                   theta_conv2_drift,
            "train_loss":                          train_loss,
            "val_loss":                            val_metrics["loss"],
            "val_acc":                             val_metrics["acc"],
            "test_loss":                           test_metrics["loss"],
            "test_acc":                            test_metrics["acc"],
            "val_feature_std_mean":                val_metrics["feature_std_mean"],
            "val_feature_var_mean":                val_metrics["feature_var_mean"],
            "val_feature_between_var_mean":        val_metrics["feature_between_var_mean"],
            "val_feature_within_var_mean":         val_metrics["feature_within_var_mean"],
            "val_feature_between_within_ratio":    val_metrics["feature_between_within_ratio"],
            "val_feature_class_center_distance_mean": val_metrics["feature_class_center_distance_mean"],
            "sec_epoch":                           time.time() - t0,
        }
        rows.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience      = 0
        else:
            patience += 1

        if epoch == 1 or epoch % CONFIG["PRINT_EVERY"] == 0:
            print(
                f"epoch={epoch:02d} "
                f"train_loss={train_loss:.4f} "
                f"val_acc={val_metrics['acc']:.3f} val_loss={val_metrics['loss']:.4f} "
                f"test_acc={test_metrics['acc']:.3f} test_loss={test_metrics['loss']:.4f} "
                f"gconv={grad_conv_mean:.2e} gpool={grad_pool_mean:.2e} gconv2={grad_conv2_mean:.2e} "
                f"grad_q={grad_qkernel_mean:.2e} grad_head={grad_head_mean:.2e} "
                f"ratio={grad_ratio_mean:.2e} "
                f"th1={theta_conv_drift:.2e} phi={phi_pool_drift:.2e} th2={theta_conv2_drift:.2e} "
                f"sep_ratio={val_metrics['feature_between_within_ratio']:.2e} "
                f"center_dist={val_metrics['feature_class_center_distance_mean']:.2e} "
                f"sec={row['sec_epoch']:.1f}"
            )

        if patience >= CONFIG["EARLY_STOP_PATIENCE"]:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val  = evaluate(model, train_data, TRAIN_CACHE, val_idx,  with_feature_diagnostics=True)
    final_test = evaluate(model, test_data,  TEST_CACHE,  test_idx, with_feature_diagnostics=True)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    final = {
        "run_name":                                    run_name,
        "freeze_qcnn":                                 bool(freeze_qcnn),
        "seed":                                        seed,
        "nparams_trainable":                           count_trainable_params(model),
        "n_patch_convolutions_per_image":              N_PATCHES,
        "final_theta_conv_drift":  float(torch.norm(model.theta_q[SL_CONV].detach()  - theta_q_init[SL_CONV]).item()),
        "final_phi_pool_drift":    float(torch.norm(model.theta_q[SL_POOL].detach()  - theta_q_init[SL_POOL]).item()),
        "final_theta_conv2_drift": float(torch.norm(model.theta_q[SL_CONV2].detach() - theta_q_init[SL_CONV2]).item()),
        "best_val_loss":                               best_val_loss,
        "final_val_acc":                               final_val["acc"],
        "final_val_loss":                              final_val["loss"],
        "final_test_acc":                              final_test["acc"],
        "final_test_loss":                             final_test["loss"],
        "final_val_feature_between_within_ratio":      final_val["feature_between_within_ratio"],
        "final_val_feature_center_distance_mean":      final_val["feature_class_center_distance_mean"],
        "final_test_feature_between_within_ratio":     final_test["feature_between_within_ratio"],
        "final_test_feature_center_distance_mean":     final_test["feature_class_center_distance_mean"],
        "final_val_feature_std_mean":                  final_val["feature_std_mean"],
        "final_test_feature_std_mean":                 final_test["feature_std_mean"],
    }

    with open(os.path.join(run_dir, "final_eval.json"), "w") as f:
        json.dump(final, f, indent=2)

    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

    print("Final:", json.dumps(final, indent=2))
    return metrics_df, final


def run_all(train_data, test_data):
    os.makedirs(BASE_DIR, exist_ok=True)

    all_metrics, all_finals = [], []
    seed = int(CONFIG["SEED"])

    for run_cfg in CONFIG["RUNS"]:
        metrics_df, final = train_one_run(
            run_name    = run_cfg["run_name"],
            freeze_qcnn = bool(run_cfg["freeze_qcnn"]),
            seed        = seed,
            train_data  = train_data,
            test_data   = test_data,
        )
        all_metrics.append(metrics_df)
        all_finals.append(final)

    metrics_all = pd.concat(all_metrics, axis=0).reset_index(drop=True)
    finals_df   = pd.DataFrame(all_finals)

    summary = finals_df[[
        "run_name", "freeze_qcnn", "final_val_acc", "final_val_loss",
        "final_test_acc", "final_test_loss", "nparams_trainable",
        "final_theta_conv_drift", "final_phi_pool_drift", "final_theta_conv2_drift",
        "final_val_feature_between_within_ratio",
        "final_test_feature_between_within_ratio",
        "final_val_feature_center_distance_mean",
        "final_test_feature_center_distance_mean",
    ]].copy()

    metrics_all.to_csv(os.path.join(BASE_DIR, "ALL_metrics.csv"),    index=False)
    finals_df.to_csv(  os.path.join(BASE_DIR, "ALL_final_eval.csv"), index=False)

    with open(os.path.join(BASE_DIR, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    print("\n" + "=" * 90)
    print("FINAL TRAINABLE VS FROZEN SUMMARY")
    print(summary.to_string())

    return metrics_all, finals_df, summary


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    set_global_seed(CONFIG["SEED"])

    print("Downloading Fashion-MNIST...")
    train_dataset, test_dataset = get_datasets()

    print("Loading train split (transform_train applied once per sample)...")
    train_data = [(x, int(y)) for x, y in train_dataset]

    print("Loading test split (transform_eval applied once per sample)...")
    test_data = [(x, int(y)) for x, y in test_dataset]

    print(
        f"Fashion-MNIST | train={len(train_data)} | test={len(test_data)} | "
        f"classes={N_CLASSES} | aug=RandomHorizontalFlip(p=0.5, train only)"
    )

    run_all(train_data, test_data)
