# %% [markdown]
# # Classificazione Quantistica delle Immagini (QVC) — 8 Qubit
#
# Versione estesa a 8 qubit su Fashion-MNIST (16x16).
# Struttura identica a Test.py; le differenze sono documentate in modifiche_8_qubit.md.

# %%
import os, json, time, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import pennylane as qml

torch.set_num_threads(1)

# ----------------------------
# CONFIG
# ----------------------------
CONFIG = {
    "BASE_DIR": "./fashion_8qubit_16x16",
    "VARIANTS": [
        "E1",
        "E2",
        "E3",
        "E4",
    ],
    "SEEDS": [0, 1, 2],
    "TRAIN_SAMPLES": 1000,
    "VAL_SAMPLES": 500,
    "TEST_SAMPLES": 500,
    "EPOCHS": 15,
    "LR": 1e-2,
    "WEIGHT_DECAY": 0.0,
    "CLIP_NORM": 0.5,
    "ACC_STEPS": 5,
    "PRINT_EVERY": 3,
    "DIAG_TRAIN_SUBSET": 500,
    "E3_FALLBACK_WARN_IF_GT": 0,
    "FAIR_DIM_MATCH": True,
    "DO_DRAW": False,
    "SAVE_FIGS": True,
    "DO_CONFUSION_SEED0": True,
    # ---- Global injection options (E1 only)
    "E1NP_USE_REUPLOAD": True,
    "E1NP_OMEGA_TRAINABLE": False,
    "E1NP_OMEGA_FIXED": math.pi / 2,
    # ---- Separate learning rates
    "LR_HEAD": 1e-2,
    "LR_QKERNEL": 1e-3,
    "LR_EMBED": 5e-4,
    "E1_INIT_A": 0.2,
}

N_CLASSES = 10  # Fashion-MNIST: tutte le 10 classi, nessun filtraggio
N_QUBITS = 8

LAMBDA_FUSION = np.pi / 4
EPS = 1e-8
BETA_GLOBAL = np.array([1.0, 10.0, 10.0, 1.0], dtype=np.float32)

BASE_DIR = CONFIG["BASE_DIR"]
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")


# ----------------------------
# Seeds
# ----------------------------
def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------
# Data (Fashion-MNIST -> 16x16)
# ----------------------------
transform = transforms.Compose(
    [
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ]
)


def get_datasets():
    os.makedirs("./data", exist_ok=True)
    train_dataset = torchvision.datasets.FashionMNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        "./data", train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


# ----------------------------
# Feature extraction (cached)
# ----------------------------
def image_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    return img_tensor.squeeze(0).cpu().numpy().astype(np.float32)


def extract_patches_2x2(X: np.ndarray) -> np.ndarray:
    # 16x16 image -> 8x8 grid = 64 patches of 2x2
    patches = []
    for r in range(8):
        for c in range(8):
            patches.append(
                np.array(
                    [
                        X[2 * r, 2 * c],
                        X[2 * r, 2 * c + 1],
                        X[2 * r + 1, 2 * c],
                        X[2 * r + 1, 2 * c + 1],
                    ],
                    dtype=np.float32,
                )
            )
    return np.stack(patches, axis=0)  # (64, 4)


def global_set_A_features(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    g1 = float(X.mean())
    g2 = float(((X - g1) ** 2).mean())
    dx = X[:, 1:] - X[:, :-1]   # (16, 15)
    dy = X[1:, :] - X[:-1, :]   # (15, 16)
    dx_o = dx[0:15, 0:15]        # full overlap region for 16x16
    dy_o = dy[0:15, 0:15]
    g3 = float((dx_o**2 + dy_o**2).mean())
    H = float((dx**2).sum())
    V = float((dy**2).sum())
    g4 = float((V - H) / (V + H + eps))
    return np.array([g1, g2, g3, g4], dtype=np.float32)


def quadrants_8x8_flat(X: np.ndarray) -> np.ndarray:
    # 16x16 image -> 4 quadrants of 8x8 = 64 pixels each
    return np.stack(
        [
            X[0:8, 0:8].reshape(-1),
            X[0:8, 8:16].reshape(-1),
            X[8:16, 0:8].reshape(-1),
            X[8:16, 8:16].reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)  # (4, 64)


# Patch grid: 8x8, patch (r,c) -> index r*8+c
# Each quadrant covers a 4x4 sub-grid of patches (16 patches).
# 8 meta-patches per quadrant = pairs of horizontally adjacent patches.
QUAD_META_PATCH_IDXS = [
    # Q0: image [0:8, 0:8]  -> patch rows 0-3, cols 0-3
    [[0, 1], [2, 3], [8, 9], [10, 11], [16, 17], [18, 19], [24, 25], [26, 27]],
    # Q1: image [0:8, 8:16] -> patch rows 0-3, cols 4-7
    [[4, 5], [6, 7], [12, 13], [14, 15], [20, 21], [22, 23], [28, 29], [30, 31]],
    # Q2: image [8:16, 0:8] -> patch rows 4-7, cols 0-3
    [[32, 33], [34, 35], [40, 41], [42, 43], [48, 49], [50, 51], [56, 57], [58, 59]],
    # Q3: image [8:16, 8:16]-> patch rows 4-7, cols 4-7
    [[36, 37], [38, 39], [44, 45], [46, 47], [52, 53], [54, 55], [60, 61], [62, 63]],
]

TRAIN_CACHE = {}
TEST_CACHE = {}


def get_features(dataset_list, cache, idx: int):
    if idx in cache:
        return cache[idx]
    x, y = dataset_list[idx]
    X = image_to_numpy(x)

    patches = extract_patches_2x2(X)   # (64, 4)
    means64 = patches.mean(axis=1)     # (64,)
    quads64 = quadrants_8x8_flat(X)    # (4, 64)
    gA4 = global_set_A_features(X)     # (4,)

    # 8 meta-patch means per quadrant: mean of 2 adjacent patch means
    quad_means = np.stack(
        [
            np.array(
                [(means64[p[0]] + means64[p[1]]) * 0.5 for p in QUAD_META_PATCH_IDXS[q]],
                dtype=np.float32,
            )
            for q in range(4)
        ],
        axis=0,
    )  # (4, 8)

    sample = {
        "patches": torch.tensor(patches, dtype=torch.float32),      # (64, 4)
        "means64": torch.tensor(means64, dtype=torch.float32),      # (64,)
        "quad_means": torch.tensor(quad_means, dtype=torch.float32),# (4, 8)
        "quads64": torch.tensor(quads64, dtype=torch.float32),      # (4, 64)
        "gA4": torch.tensor(gA4, dtype=torch.float32),              # (4,)
        "y": int(y),
    }
    cache[idx] = sample
    return sample


# ----------------------------
# Stratified splits
# ----------------------------
def stratified_split_indices(dataset_list, n_train: int, n_val: int, seed: int):
    rng = random.Random(seed)
    buckets = {c: [] for c in range(N_CLASSES)}
    for i, (_, y) in enumerate(dataset_list):
        buckets[int(y)].append(i)

    def per_class_quota(total):
        per = total // N_CLASSES
        rem = total % N_CLASSES
        return [per + (1 if c < rem else 0) for c in range(N_CLASSES)]

    q_train = per_class_quota(n_train)
    q_val = per_class_quota(n_val)

    train_idx, val_idx = [], []
    for c in range(N_CLASSES):
        pool = buckets[c][:]
        rng.shuffle(pool)

        take_tr = min(q_train[c], len(pool))
        tr = pool[:take_tr]
        pool = pool[take_tr:]

        take_va = min(q_val[c], len(pool))
        va = pool[:take_va]

        train_idx.extend(tr)
        val_idx.extend(va)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def stratified_indices(dataset_list, n_samples: int, seed: int):
    rng = random.Random(seed)
    buckets = {c: [] for c in range(N_CLASSES)}
    for i, (_, y) in enumerate(dataset_list):
        buckets[int(y)].append(i)
    per = n_samples // N_CLASSES
    rem = n_samples % N_CLASSES
    idxs = []
    for c in range(N_CLASSES):
        take = per + (1 if c < rem else 0)
        take = min(take, len(buckets[c]))
        idxs.extend(rng.sample(buckets[c], take))
    rng.shuffle(idxs)
    return idxs


# ----------------------------
# Devices
# ----------------------------
def make_device(n_wires: int):
    try:
        d = qml.device("lightning.qubit", wires=n_wires)
        return d, "lightning.qubit", "adjoint"
    except Exception:
        d = qml.device("default.qubit", wires=n_wires)
        return d, "default.qubit", "backprop"


dev8, dev8_name, diff8 = make_device(8)
dev9, dev9_name, diff9 = make_device(9)  # E1: 8 local + 1 global wire

QNODE8_KW = dict(interface="torch", diff_method=diff8)
QNODE9_KW = dict(interface="torch", diff_method=diff9)


# ----------------------------
# Shared quantum kernel: conv8 + pool8
# ----------------------------
def conv8(theta, wires):
    q = wires
    for i in range(8):
        qml.RY(theta[i], wires=q[i])
    qml.CNOT(wires=[q[0], q[1]]); qml.RZ(theta[8],  wires=q[1])
    qml.CNOT(wires=[q[2], q[3]]); qml.RZ(theta[9],  wires=q[3])
    qml.CNOT(wires=[q[4], q[5]]); qml.RZ(theta[10], wires=q[5])
    qml.CNOT(wires=[q[6], q[7]]); qml.RZ(theta[11], wires=q[7])
    qml.CNOT(wires=[q[1], q[2]]); qml.RZ(theta[12], wires=q[2])
    qml.CNOT(wires=[q[3], q[4]]); qml.RZ(theta[13], wires=q[4])
    qml.CNOT(wires=[q[5], q[6]]); qml.RZ(theta[14], wires=q[6])
    qml.CNOT(wires=[q[7], q[0]]); qml.RZ(theta[15], wires=q[0])


def pool8(phi, wires):
    q = wires
    qml.CRZ(phi[0], wires=[q[1], q[0]])
    qml.CRZ(phi[1], wires=[q[3], q[2]])
    qml.CRZ(phi[2], wires=[q[5], q[4]])
    qml.CRZ(phi[3], wires=[q[7], q[6]])
    qml.CRX(phi[4], wires=[q[2], q[1]])
    qml.CRX(phi[5], wires=[q[4], q[3]])
    qml.CRX(phi[6], wires=[q[6], q[5]])
    qml.CRX(phi[7], wires=[q[0], q[7]])


# ----------------------------
# Embeddings
# ----------------------------
def embed_E2_local_8(quad_means_8):
    for i in range(8):
        qml.RY(math.pi * quad_means_8[i], wires=i)


def embed_E4_local_8(quad_means_8, a8, c8):
    for i in range(8):
        qml.RY(a8[i] * (math.pi * quad_means_8[i]) + c8[i], wires=i)


def _gamma_vec_from_gA4_torch(gA4_vec: torch.Tensor) -> torch.Tensor:
    beta = torch.tensor(BETA_GLOBAL, dtype=gA4_vec.dtype, device=gA4_vec.device)
    return math.pi * torch.tanh(beta * gA4_vec)


def inject_global_on_wire_8(
    gA4_vec, omega=None, use_reupload=True, omega_fixed_tensor=None
):
    gammas = _gamma_vec_from_gA4_torch(gA4_vec)
    g0, g1, g2, g3 = gammas[0], gammas[1], gammas[2], gammas[3]

    qml.RY(g0, wires=8)
    qml.RZ(g1, wires=8)
    qml.RX(g2, wires=8)
    qml.RZ(g3, wires=8)

    if use_reupload:
        if omega is None:
            if omega_fixed_tensor is None:
                omega_fixed_tensor = torch.tensor(
                    float(CONFIG["E1NP_OMEGA_FIXED"]), dtype=torch.float32
                )
            qml.RY(omega_fixed_tensor, wires=8)
        else:
            qml.RY(omega, wires=8)

        qml.RZ(g0, wires=8)
        qml.RX(g1, wires=8)
        qml.RY(g2, wires=8)
        qml.RZ(g3, wires=8)


def fuse_global_to_locals_8(lam=LAMBDA_FUSION):
    for i in range(8):
        qml.CNOT(wires=[8, i])
        qml.RZ(lam, wires=i)
        qml.CNOT(wires=[8, i])


# ----------------------------
# QNodes
# ----------------------------
@qml.qnode(dev8, **QNODE8_KW)
def qnode_quadrant_E2_8(quad_means_8, theta_conv, phi_pool):
    embed_E2_local_8(quad_means_8)
    conv8(theta_conv, wires=list(range(8)))
    pool8(phi_pool, wires=list(range(8)))
    return [qml.expval(qml.PauliZ(i)) for i in range(8)]


@qml.qnode(dev8, **QNODE8_KW)
def qnode_quadrant_E3_8(quad_amp_64, theta_conv, phi_pool):
    # pad_with=0.0 estende 64 → 256 (= 2^8); normalize=True normalizza a norma 1.
    # Fallback su near-zero: AmplitudeEmbedding con normalize=True fallirebbe su vettore nullo.
    amp = torch.clamp(quad_amp_64, 0.0, 1.0)
    nrm = torch.linalg.norm(amp)
    if nrm.item() < 1e-12:
        amp = torch.zeros(64, dtype=amp.dtype)
        amp[0] = 1.0
    qml.AmplitudeEmbedding(amp, wires=range(8), pad_with=0.0, normalize=True)
    conv8(theta_conv, wires=list(range(8)))
    pool8(phi_pool, wires=list(range(8)))
    return [qml.expval(qml.PauliZ(i)) for i in range(8)]


@qml.qnode(dev8, **QNODE8_KW)
def qnode_quadrant_E4_8(quad_means_8, a8, c8, theta_conv, phi_pool):
    embed_E4_local_8(quad_means_8, a8, c8)
    conv8(theta_conv, wires=list(range(8)))
    pool8(phi_pool, wires=list(range(8)))
    return [qml.expval(qml.PauliZ(i)) for i in range(8)]


@qml.qnode(dev9, **QNODE9_KW)
def qnode_quadrant_E1_8(
    quad_means_8,
    gA4_vec,
    a8,
    c8,
    theta_conv,
    phi_pool,
    include_global_readout: bool,
    omega=None,
    use_reupload=True,
):
    embed_E4_local_8(quad_means_8, a8, c8)
    inject_global_on_wire_8(gA4_vec, omega=omega, use_reupload=use_reupload)
    fuse_global_to_locals_8(lam=LAMBDA_FUSION)
    conv8(theta_conv, wires=list(range(8)))
    pool8(phi_pool, wires=list(range(8)))

    outs = [qml.expval(qml.PauliZ(i)) for i in range(8)]
    if include_global_readout:
        outs.append(qml.expval(qml.PauliZ(8)))
    return outs


# ----------------------------
# Model
# ----------------------------
class QuanvEmbedModel(torch.nn.Module):
    def __init__(self, variant: str, fair_dim_match: bool = True):
        super().__init__()
        assert variant in ("E1", "E2", "E3", "E4")
        self.variant = variant
        self.fair_dim_match = bool(fair_dim_match)

        # 32 params = 4 quadrants x 8 meta-patches each
        self.e1_a32 = torch.nn.Parameter(
            torch.full((32,), float(CONFIG["E1_INIT_A"]), dtype=torch.float32)
        )
        self.e1_c32 = torch.nn.Parameter(torch.zeros(32, dtype=torch.float32))

        if CONFIG["E1NP_OMEGA_TRAINABLE"]:
            self.e1np_omega = torch.nn.Parameter(
                torch.tensor(float(CONFIG["E1NP_OMEGA_FIXED"]), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "e1np_omega",
                torch.tensor(float(CONFIG["E1NP_OMEGA_FIXED"]), dtype=torch.float32),
            )

        self.register_buffer(
            "e1np_omega_fixed_tensor",
            torch.tensor(float(CONFIG["E1NP_OMEGA_FIXED"]), dtype=torch.float32),
        )

        self.e4_a32 = torch.nn.Parameter(torch.ones(32, dtype=torch.float32))
        self.e4_c32 = torch.nn.Parameter(torch.zeros(32, dtype=torch.float32))

        self.theta_conv = torch.nn.Parameter(0.01 * torch.randn(16, dtype=torch.float32))
        self.phi_pool = torch.nn.Parameter(torch.zeros(8, dtype=torch.float32))

        # E1 with fair_dim_match=False: each of 4 qnodes returns 9 values (8+1 global)
        if self.variant == "E1" and (not self.fair_dim_match):
            out_dim = 36
        else:
            out_dim = 32

        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(out_dim),
            torch.nn.Linear(out_dim, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.05),
            torch.nn.Linear(64, N_CLASSES),
        )

    def _quadrant_params(self, a32, c32):
        # (32,) -> (4, 8): one row per quadrant
        return a32.reshape(4, 8), c32.reshape(4, 8)

    def features_from_sample(self, sample: dict):
        feats = []
        qtime = 0.0
        e3_fallback_quadrants = 0

        if self.variant == "E1":
            quad_means = sample["quad_means"]  # (4, 8)
            gA4 = sample["gA4"]
            include_global = not self.fair_dim_match
            a8s, c8s = self._quadrant_params(self.e1_a32, self.e1_c32)

            for q in range(4):
                t0 = time.time()
                out = qnode_quadrant_E1_8(
                    quad_means[q],
                    gA4,
                    a8s[q],
                    c8s[q],
                    self.theta_conv,
                    self.phi_pool,
                    include_global_readout=include_global,
                    omega=self.e1np_omega,
                    use_reupload=CONFIG["E1NP_USE_REUPLOAD"],
                )
                qtime += time.time() - t0
                feats.append(torch.stack(out))

            feat_vec = torch.cat(feats, dim=0)

        elif self.variant == "E2":
            quad_means = sample["quad_means"]  # (4, 8)
            for q in range(4):
                t0 = time.time()
                out = qnode_quadrant_E2_8(quad_means[q], self.theta_conv, self.phi_pool)
                qtime += time.time() - t0
                feats.append(torch.stack(out))
            feat_vec = torch.cat(feats, dim=0)

        elif self.variant == "E3":
            quads64 = sample["quads64"]  # (4, 64)
            for q in range(4):
                amp = torch.clamp(quads64[q], 0.0, 1.0)
                if torch.linalg.norm(amp).item() < 1e-12:
                    e3_fallback_quadrants += 1
                t0 = time.time()
                out = qnode_quadrant_E3_8(quads64[q], self.theta_conv, self.phi_pool)
                qtime += time.time() - t0
                feats.append(torch.stack(out))
            feat_vec = torch.cat(feats, dim=0)

        else:  # E4
            quad_means = sample["quad_means"]  # (4, 8)
            a8s, c8s = self._quadrant_params(self.e4_a32, self.e4_c32)
            for q in range(4):
                t0 = time.time()
                out = qnode_quadrant_E4_8(
                    quad_means[q], a8s[q], c8s[q], self.theta_conv, self.phi_pool
                )
                qtime += time.time() - t0
                feats.append(torch.stack(out))
            feat_vec = torch.cat(feats, dim=0)

        if torch.isnan(feat_vec).any() or torch.isinf(feat_vec).any():
            raise FloatingPointError(
                f"NaN/Inf in quantum features for variant={self.variant}"
            )
        return (
            feat_vec.to(dtype=torch.float32),
            float(qtime),
            int(e3_fallback_quadrants),
        )

    def forward(self, sample: dict):
        feat_vec, qtime, e3_fb = self.features_from_sample(sample)
        logits = self.head(feat_vec)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise FloatingPointError(f"NaN/Inf in logits for variant={self.variant}")
        return logits, feat_vec, qtime, e3_fb


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------
# Diagnostics
# ----------------------------
ce_loss = torch.nn.CrossEntropyLoss()


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def grad_norm_of_params(params):
    s = 0.0
    for p in params:
        if p is None or (p.grad is None):
            continue
        g = p.grad.detach()
        s += float((g.norm() ** 2).item())
    return float(math.sqrt(s))


def group_grad_norms(model: QuanvEmbedModel):
    return {
        "grad_e1_embed": grad_norm_of_params(
            [model.e1_a32, model.e1_c32, getattr(model, "e1np_omega", None)]
        )
        if model.variant == "E1"
        else 0.0,
        "grad_e4_embed": grad_norm_of_params([model.e4_a32, model.e4_c32])
        if model.variant == "E4"
        else 0.0,
        "grad_qkernel": grad_norm_of_params([model.theta_conv, model.phi_pool]),
        "grad_head": grad_norm_of_params(list(model.head.parameters())),
    }


@torch.no_grad()
def eval_subset(model, dataset_list, cache, indices):
    model.eval()
    ys, preds, losses, gaps = [], [], [], []
    feat_list = []
    qtime = 0.0
    e3_fallback_total = 0

    for idx in indices:
        sample = get_features(dataset_list, cache, idx)
        logits, feat, qsec, e3_fb = model(sample)

        qtime += qsec
        e3_fallback_total += e3_fb

        y = sample["y"]
        loss = ce_loss(logits.view(1, -1), torch.tensor([y], dtype=torch.long)).item()
        losses.append(loss)

        p = int(torch.argmax(logits).item())
        ys.append(y)
        preds.append(p)

        top2 = torch.topk(logits, k=2).values
        gaps.append(float((top2[0] - top2[1]).item()))
        feat_list.append(feat.detach().cpu().numpy())

    ys = np.array(ys, dtype=int)
    preds = np.array(preds, dtype=int)
    acc = float((ys == preds).mean()) if len(ys) else 0.0

    F = np.stack(feat_list, axis=0) if len(feat_list) else None
    feat_mean = F.mean(axis=0) if F is not None else None
    feat_std = F.std(axis=0) if F is not None else None

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": acc,
        "gap": float(np.mean(gaps)) if gaps else 0.0,
        "y_true": ys,
        "y_pred": preds,
        "qtime_sec": float(qtime),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "e3_fallback_quadrants_total": int(e3_fallback_total),
    }


# ----------------------------
# Circuit draw
# ----------------------------
def draw_circuits(save: bool = True):
    quad_means_8 = torch.ones(8, dtype=torch.float32) * 0.5
    quad_amp_64 = torch.ones(64, dtype=torch.float32) * (1.0 / 64.0)
    a8 = torch.ones(8, dtype=torch.float32)
    c8 = torch.zeros(8, dtype=torch.float32)
    gA4_vec = torch.tensor([0.5, 0.1, 0.1, 0.0], dtype=torch.float32)
    theta_conv = torch.zeros(16, dtype=torch.float32)
    phi_pool = torch.zeros(8, dtype=torch.float32)

    plt.rcParams["figure.facecolor"] = "white"

    if "E1" in CONFIG["VARIANTS"]:
        plt.figure(figsize=(24, 4), facecolor="white")
        qml.draw_mpl(qnode_quadrant_E1_8)(
            quad_means_8,
            gA4_vec,
            a8,
            c8,
            theta_conv,
            phi_pool,
            (not CONFIG["FAIR_DIM_MATCH"]),
            omega=torch.tensor(float(CONFIG["E1NP_OMEGA_FIXED"])),
            use_reupload=CONFIG["E1NP_USE_REUPLOAD"],
        )
        plt.title("Circuit (per quadrant) — E1 (8 qubit)")
        if save and CONFIG["SAVE_FIGS"]:
            plt.savefig(
                os.path.join(ANALYSIS_DIR, "circuit_E1_8qubit.png"),
                dpi=200,
                bbox_inches="tight",
            )
        plt.close()

    if "E2" in CONFIG["VARIANTS"]:
        plt.figure(figsize=(24, 4), facecolor="white")
        qml.draw_mpl(qnode_quadrant_E2_8)(quad_means_8, theta_conv, phi_pool)
        plt.title("Circuit (per quadrant) — E2 (8 qubit)")
        if save and CONFIG["SAVE_FIGS"]:
            plt.savefig(
                os.path.join(ANALYSIS_DIR, "circuit_E2_8qubit.png"),
                dpi=200,
                bbox_inches="tight",
            )
        plt.close()

    if "E3" in CONFIG["VARIANTS"]:
        plt.figure(figsize=(24, 4), facecolor="white")
        qml.draw_mpl(qnode_quadrant_E3_8)(quad_amp_64, theta_conv, phi_pool)
        plt.title("Circuit (per quadrant) — E3 (8 qubit)")
        if save and CONFIG["SAVE_FIGS"]:
            plt.savefig(
                os.path.join(ANALYSIS_DIR, "circuit_E3_8qubit.png"),
                dpi=200,
                bbox_inches="tight",
            )
        plt.close()

    if "E4" in CONFIG["VARIANTS"]:
        plt.figure(figsize=(24, 4), facecolor="white")
        qml.draw_mpl(qnode_quadrant_E4_8)(quad_means_8, a8, c8, theta_conv, phi_pool)
        plt.title("Circuit (per quadrant) — E4 (8 qubit)")
        if save and CONFIG["SAVE_FIGS"]:
            plt.savefig(
                os.path.join(ANALYSIS_DIR, "circuit_E4_8qubit.png"),
                dpi=200,
                bbox_inches="tight",
            )
        plt.close()


# ----------------------------
# Training loop
# ----------------------------
def run_one(
    variant: str,
    seed: int,
    train_idx,
    val_idx,
    test_idx,
    out_dir: str,
    train_data,
    test_data,
):
    os.makedirs(out_dir, exist_ok=True)
    set_global_seed(seed)

    model = QuanvEmbedModel(variant, fair_dim_match=CONFIG["FAIR_DIM_MATCH"])
    nparams = count_trainable_params(model)

    head_params = list(model.head.parameters())
    qkernel_params = [model.theta_conv, model.phi_pool]

    embed_params = []
    if variant == "E1":
        embed_params += [model.e1_a32, model.e1_c32, getattr(model, "e1np_omega", None)]
        embed_params = [p for p in embed_params if p is not None]
    elif variant == "E4":
        embed_params += [model.e4_a32, model.e4_c32]

    opt = torch.optim.Adam(
        [
            {
                "params": head_params,
                "lr": CONFIG["LR_HEAD"],
                "weight_decay": CONFIG["WEIGHT_DECAY"],
            },
            {
                "params": qkernel_params,
                "lr": CONFIG["LR_QKERNEL"],
                "weight_decay": CONFIG["WEIGHT_DECAY"],
            },
            {
                "params": embed_params,
                "lr": CONFIG["LR_EMBED"],
                "weight_decay": CONFIG["WEIGHT_DECAY"],
            }
            if len(embed_params)
            else {
                "params": [],
                "lr": CONFIG["LR_EMBED"],
                "weight_decay": CONFIG["WEIGHT_DECAY"],
            },
        ]
    )

    rows = []
    acc_steps = max(1, int(CONFIG["ACC_STEPS"]))

    for ep in range(1, CONFIG["EPOCHS"] + 1):
        t_ep = time.time()
        model.train()
        random.shuffle(train_idx)

        qtime_train = 0.0
        total_loss = 0.0
        e3_fb_train = 0

        opt.zero_grad()
        step_in_acc = 0

        for idx in train_idx:
            sample = get_features(train_data, TRAIN_CACHE, idx)
            logits, _, qsec, e3_fb = model(sample)

            qtime_train += qsec
            e3_fb_train += e3_fb

            y = torch.tensor([sample["y"]], dtype=torch.long)
            loss = ce_loss(logits.view(1, -1), y)

            (loss / acc_steps).backward()
            total_loss += float(loss.item())
            step_in_acc += 1

            if step_in_acc == acc_steps:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=CONFIG["CLIP_NORM"]
                )
                opt.step()
                opt.zero_grad()
                step_in_acc = 0

        if step_in_acc != 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=CONFIG["CLIP_NORM"]
            )
            opt.step()
            opt.zero_grad()

        # diagnostic backward for grad norms
        model.train()
        opt.zero_grad()
        diag_sample = get_features(train_data, TRAIN_CACHE, train_idx[0])
        diag_logits, _, _, _ = model(diag_sample)
        diag_y = torch.tensor([diag_sample["y"]], dtype=torch.long)
        diag_loss = ce_loss(diag_logits.view(1, -1), diag_y)
        diag_loss.backward()
        gnorms = group_grad_norms(model)
        opt.zero_grad()

        diag_tr_idx = train_idx[: min(CONFIG["DIAG_TRAIN_SUBSET"], len(train_idx))]
        tr = eval_subset(model, train_data, TRAIN_CACHE, diag_tr_idx)
        va = eval_subset(model, train_data, TRAIN_CACHE, val_idx)
        te = eval_subset(model, test_data, TEST_CACHE, test_idx)

        cm_test = confusion_matrix(te["y_true"], te["y_pred"], N_CLASSES)
        feat_std_mean_test = (
            float(np.mean(te["feat_std"])) if te["feat_std"] is not None else None
        )

        row = {
            "seed": seed,
            "variant": variant,
            "epoch": ep,
            "nparams_trainable": int(nparams),
            "train_loss_stepmean": total_loss / max(1, len(train_idx)),
            "train_loss_diag": tr["loss"],
            "train_acc_diag": tr["acc"],
            "train_gap_diag": tr["gap"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_gap": va["gap"],
            "test_loss": te["loss"],
            "test_acc": te["acc"],
            "test_gap": te["gap"],
            "sec_epoch": time.time() - t_ep,
            "sec_quantum_train": float(qtime_train),
            "sec_quantum_val": float(va["qtime_sec"]),
            "sec_quantum_test": float(te["qtime_sec"]),
            "e3_fallback_quadrants_train_total": int(e3_fb_train),
            "e3_fallback_quadrants_val_total": int(va["e3_fallback_quadrants_total"]),
            "e3_fallback_quadrants_test_total": int(te["e3_fallback_quadrants_total"]),
            "pred_hist_test": np.bincount(te["y_pred"], minlength=N_CLASSES).tolist(),
            "feat_std_mean_test": feat_std_mean_test,
            **gnorms,
        }
        rows.append(row)

        if (ep == 1) or (ep % CONFIG["PRINT_EVERY"] == 0) or (ep == CONFIG["EPOCHS"]):
            warn_e3 = ""
            if (
                variant == "E3"
                and row["e3_fallback_quadrants_test_total"]
                > CONFIG["E3_FALLBACK_WARN_IF_GT"]
            ):
                warn_e3 = f" | WARNING: E3 fallback_quads(test)={row['e3_fallback_quadrants_test_total']}"
            print(
                f"  ep={ep:02d} "
                f"tr_acc={row['train_acc_diag']:.3f} va_acc={row['val_acc']:.3f} te_acc={row['test_acc']:.3f} "
                f"va_loss={row['val_loss']:.3f} "
                f"q_tr={row['sec_quantum_train']:.1f}s q_va={row['sec_quantum_val']:.1f}s q_te={row['sec_quantum_test']:.1f}s "
                f"epoch={row['sec_epoch']:.1f}s | "
                f"grad(qkern)={row['grad_qkernel']:.2e} grad(head)={row['grad_head']:.2e}"
                f"{warn_e3}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    final = rows[-1].copy()
    final["confusion_matrix_test"] = cm_test.tolist()
    with open(os.path.join(out_dir, "final_eval.json"), "w") as f:
        json.dump(final, f, indent=2)

    return df, final, cm_test


def run_all(train_data, test_data):
    all_metrics = []
    all_finals = []
    seed0_confusions = {}

    for seed in CONFIG["SEEDS"]:
        train_idx, val_idx = stratified_split_indices(
            train_data, CONFIG["TRAIN_SAMPLES"], CONFIG["VAL_SAMPLES"], seed=seed + 101
        )
        test_idx = stratified_indices(
            test_data, CONFIG["TEST_SAMPLES"], seed=seed + 303
        )

        for variant in CONFIG["VARIANTS"]:
            run_id = f"{variant}_seed{seed}"
            out_dir = os.path.join(BASE_DIR, run_id)

            print("\n" + "=" * 90)
            print(
                f"RUN {run_id} | epochs={CONFIG['EPOCHS']} | train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}"
            )

            df, final, cm = run_one(
                variant,
                seed,
                train_idx,
                val_idx,
                test_idx,
                out_dir,
                train_data,
                test_data,
            )
            all_metrics.append(df)

            all_finals.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "nparams_trainable": final["nparams_trainable"],
                    "val_acc": final["val_acc"],
                    "val_loss": final["val_loss"],
                    "val_gap": final["val_gap"],
                    "test_acc": final["test_acc"],
                    "test_loss": final["test_loss"],
                    "test_gap": final["test_gap"],
                    "sec_epoch": final["sec_epoch"],
                    "sec_quantum_train": final["sec_quantum_train"],
                    "sec_quantum_val": final["sec_quantum_val"],
                    "sec_quantum_test": final["sec_quantum_test"],
                    "e3_fb_test": final["e3_fallback_quadrants_test_total"],
                }
            )

            if CONFIG["DO_CONFUSION_SEED0"] and seed == 0:
                seed0_confusions[variant] = cm

    metrics_df = pd.concat(all_metrics, axis=0).reset_index(drop=True)
    finals_df = pd.DataFrame(all_finals)

    metrics_df.to_csv(os.path.join(BASE_DIR, "ALL_metrics.csv"), index=False)
    finals_df.to_csv(os.path.join(BASE_DIR, "ALL_final_eval.csv"), index=False)

    summary_cols = [
        "val_acc", "val_loss", "val_gap",
        "test_acc", "test_loss", "test_gap",
        "sec_epoch", "sec_quantum_train", "sec_quantum_val", "sec_quantum_test",
        "nparams_trainable", "e3_fb_test",
    ]
    summary = (
        finals_df.groupby("variant")[summary_cols].agg(["mean", "std"]).reset_index()
    )
    summary.to_csv(
        os.path.join(ANALYSIS_DIR, "FINAL_summary_mean_std.csv"), index=False
    )

    if CONFIG["DO_CONFUSION_SEED0"] and (0 in CONFIG["SEEDS"]):
        all_seed0 = {
            v: seed0_confusions[v].tolist()
            for v in CONFIG["VARIANTS"]
            if v in seed0_confusions
        }
        with open(
            os.path.join(ANALYSIS_DIR, "confusions_seed0_all_variants.json"), "w"
        ) as f:
            json.dump(all_seed0, f, indent=2)

        for v in CONFIG["VARIANTS"]:
            if v in seed0_confusions:
                cm = seed0_confusions[v]
                plt.figure(figsize=(5, 5), facecolor="white")
                plt.imshow(cm)
                plt.title(f"Confusion matrix (seed0) — {v}")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.colorbar()
                if CONFIG["SAVE_FIGS"]:
                    plt.savefig(
                        os.path.join(ANALYSIS_DIR, f"confusion_seed0_{v}.png"),
                        dpi=200,
                        bbox_inches="tight",
                    )
                plt.close()

    print("\nSaved outputs in:", BASE_DIR)
    return metrics_df, finals_df


# ----------------------------
# Plots
# ----------------------------
def plot_mean_std_curves(
    metrics_df: pd.DataFrame, y_col: str, title: str, fname: str = None
):
    plt.figure(figsize=(7, 4), facecolor="white")
    for v in CONFIG["VARIANTS"]:
        sub = metrics_df[metrics_df["variant"] == v]
        if len(sub) == 0:
            continue
        grp = sub.groupby("epoch")[y_col]
        mean = grp.mean()
        std = grp.std()
        plt.plot(mean.index, mean.values, label=v)
        plt.fill_between(
            mean.index, (mean - std).values, (mean + std).values, alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    if fname and CONFIG["SAVE_FIGS"]:
        plt.savefig(os.path.join(ANALYSIS_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close()


def boxplot_final_test_acc(finals_df: pd.DataFrame, fname: str = None):
    plt.figure(figsize=(7, 4), facecolor="white")
    variants = CONFIG["VARIANTS"]
    data = [finals_df[finals_df["variant"] == v]["test_acc"].values for v in variants]
    plt.boxplot(data, labels=variants)
    plt.ylabel("Test accuracy (subset)")
    plt.title("Final test accuracy by variant (multi-seed)")
    if fname and CONFIG["SAVE_FIGS"]:
        plt.savefig(os.path.join(ANALYSIS_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    print(f"Quantum devices: dev8={dev8_name}({diff8}), dev9={dev9_name}({diff9})")

    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print("Downloading Fashion-MNIST...")
    train_dataset, test_dataset = get_datasets()

    # Build lists once: (tensor, int_label)
    print("Loading train split into memory...")
    train_data = [(x, int(y)) for x, y in train_dataset]
    print("Loading test split into memory...")
    test_data = [(x, int(y)) for x, y in test_dataset]

    print(
        f"Fashion-MNIST 16x16 | train={len(train_data)} | test={len(test_data)} | classes={N_CLASSES}"
    )

    if CONFIG["DO_DRAW"]:
        draw_circuits(save=True)

    metrics_df, finals_df = run_all(train_data, test_data)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY (mean±std over seeds)")
    summary_view = finals_df.groupby(["variant"])[
        ["val_acc", "val_loss", "val_gap", "test_acc", "test_loss", "test_gap"]
    ].agg(["mean", "std"])
    print(summary_view)

    plot_mean_std_curves(
        metrics_df, "train_acc_diag", "Train accuracy (diag) mean±std",
        fname="curve_train_acc_diag.png",
    )
    plot_mean_std_curves(
        metrics_df, "train_loss_diag", "Train loss (diag) mean±std",
        fname="curve_train_loss_diag.png",
    )
    plot_mean_std_curves(
        metrics_df, "train_gap_diag", "Train confidence gap (diag) mean±std",
        fname="curve_train_gap_diag.png",
    )
    plot_mean_std_curves(
        metrics_df, "val_acc", "Val accuracy mean±std", fname="curve_val_acc.png"
    )
    plot_mean_std_curves(
        metrics_df, "val_loss", "Val loss mean±std", fname="curve_val_loss.png"
    )
    plot_mean_std_curves(
        metrics_df, "val_gap", "Val confidence gap mean±std", fname="curve_val_gap.png"
    )
    plot_mean_std_curves(
        metrics_df, "test_acc", "Test accuracy mean±std (subset)",
        fname="curve_test_acc.png",
    )
    plot_mean_std_curves(
        metrics_df, "test_loss", "Test loss mean±std (subset)",
        fname="curve_test_loss.png",
    )
    plot_mean_std_curves(
        metrics_df, "test_gap", "Test confidence gap mean±std (subset)",
        fname="curve_test_gap.png",
    )

    boxplot_final_test_acc(finals_df, fname="boxplot_test_acc.png")

    print("\nSaved outputs in:", BASE_DIR)
    print("Analysis figures + tables in:", ANALYSIS_DIR)
