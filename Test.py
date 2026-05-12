# %% [markdown]
# # Classificazione Quantistica delle Immagini (QVC)
#
# Questo notebook esplora l'uso di Circuiti Quantistici Variazionali per la classificazione del dataset MNIST. Il codice è strutturato per essere chiaro, modulare e orientato alla ricerca accademica.

# %% [markdown]
# ## 1. Importazioni e Configurazione Globale (Esperimento a 5 Classi)
#
# In questa prima cella importiamo le librerie necessarie, tra cui `pennylane` per la simulazione quantistica e `torch` per la gestione del Deep Learning. Impostiamo inoltre i parametri del primo esperimento, che lavorerà su un sottoinsieme di 5 classi.

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
# CONFIG (v2-uniform)
# ----------------------------
CONFIG = {
    "BASE_DIR": "./mean_1k35_clip0.5_5class",
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
    # ---- NEW: separate learning rates
    "LR_HEAD": 1e-2,
    "LR_QKERNEL": 1e-3,
    "LR_EMBED": 5e-4,
    "E1_INIT_A": 0.2,
}

TARGET_DIGITS = [0, 1, 2, 3, 8]
LABEL_MAP = {d: i for i, d in enumerate(TARGET_DIGITS)}
N_CLASSES = len(TARGET_DIGITS)

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
# Data (MNIST -> filter -> 8x8)
# ----------------------------
transform = transforms.Compose(
    [
        transforms.Resize((8, 8)),
        transforms.ToTensor(),
    ]
)


def get_datasets():
    # Creazione folder per i dati se non esiste
    os.makedirs("./data", exist_ok=True)
    train_dataset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def filter_and_remap(dataset):
    out = []
    for x, y in dataset:
        y = int(y)
        if y in LABEL_MAP:
            out.append((x, LABEL_MAP[y]))
    return out


# %% [markdown]
# ## 2. Estrazione delle Feature e Data Processing
#
# Prima di elaborare le immagini quantisticamente, queste vengono ridimensionate (8x8) ed elaborate per estrarre patch e valori medi. Per ottimizzare l'esecuzione, viene implementato un meccanismo di caching (`TRAIN_CACHE`, `TEST_CACHE`) e vengono definite funzioni rigorose per la divisione stratificata dei dataset.


# %%
# ----------------------------
# Feature extraction (cached)
# ----------------------------
def image_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    return img_tensor.squeeze(0).cpu().numpy().astype(np.float32)


def extract_patches_2x2(X: np.ndarray) -> np.ndarray:
    patches = []
    for r in range(4):
        for c in range(4):
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
    return np.stack(patches, axis=0)  # (16,4)


def global_set_A_features(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    g1 = float(X.mean())
    g2 = float(((X - g1) ** 2).mean())
    dx = X[:, 1:] - X[:, :-1]
    dy = X[1:, :] - X[:-1, :]
    dx_o = dx[0:7, 0:7]
    dy_o = dy[0:7, 0:7]
    g3 = float((dx_o**2 + dy_o**2).mean())
    H = float((dx**2).sum())
    V = float((dy**2).sum())
    g4 = float((V - H) / (V + H + eps))
    return np.array([g1, g2, g3, g4], dtype=np.float32)


def quadrants_4x4_flat(X: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            X[0:4, 0:4].reshape(-1),
            X[0:4, 4:8].reshape(-1),
            X[4:8, 0:4].reshape(-1),
            X[4:8, 4:8].reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)  # (4,16)


QUAD_PATCH_IDXS = [
    [0, 1, 4, 5],
    [2, 3, 6, 7],
    [8, 9, 12, 13],
    [10, 11, 14, 15],
]

TRAIN_CACHE = {}
TEST_CACHE = {}


def get_features(dataset_list, cache, idx: int):
    if idx in cache:
        return cache[idx]
    x, y = dataset_list[idx]
    X = image_to_numpy(x)

    patches = extract_patches_2x2(X)  # (16,4)  values in [0,1]
    means16 = patches.mean(axis=1)  # (16,)
    quads16 = quadrants_4x4_flat(X)  # (4,16)
    gA4 = global_set_A_features(X)  # (4,)

    quad_means = np.stack([means16[q] for q in QUAD_PATCH_IDXS], axis=0)  # (4,4)

    sample = {
        "patches": torch.tensor(patches, dtype=torch.float32),  # (16,4)
        "means16": torch.tensor(means16, dtype=torch.float32),  # (16,)
        "quad_means": torch.tensor(quad_means, dtype=torch.float32),  # (4,4)
        "quads16": torch.tensor(quads16, dtype=torch.float32),  # (4,16)
        "gA4": torch.tensor(gA4, dtype=torch.float32),  # (4,)
        "y": int(y),
    }
    cache[idx] = sample
    return sample


# ----------------------------
# Stratified splits (disjoint train/val) (v2-uniform)
# ----------------------------
def stratified_split_indices(dataset_list, n_train: int, n_val: int, seed: int):
    rng = random.Random(seed)
    buckets = {c: [] for c in range(N_CLASSES)}
    for i, (_, y) in enumerate(dataset_list):
        buckets[y].append(i)

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
        buckets[y].append(i)
    per = n_samples // N_CLASSES
    rem = n_samples % N_CLASSES
    idxs = []
    for c in range(N_CLASSES):
        take = per + (1 if c < rem else 0)
        take = min(take, len(buckets[c]))
        idxs.extend(rng.sample(buckets[c], take))
    rng.shuffle(idxs)
    return idxs


# %% [markdown]
# ## 3. Dispositivi Quantistici e Definizione dei QNode
#
# Definiamo qui i nodi quantistici (`QNode`) utilizzando `pennylane`. La logica include l'embedding dei dati classici in stati quantistici e i successivi ansatz variazionali che fungono da feature extractor non lineari.


# %%
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


dev4, dev4_name, diff4 = make_device(4)
dev5, dev5_name, diff5 = make_device(5)

QNODE4_KW = dict(interface="torch", diff_method=diff4)
QNODE5_KW = dict(interface="torch", diff_method=diff5)


# ----------------------------
# Shared quantum kernel (conv + pool)
# ----------------------------
def conv4(theta, wires):
    q0, q1, q2, q3 = wires
    qml.RY(theta[0], wires=q0)
    qml.RY(theta[1], wires=q1)
    qml.RY(theta[2], wires=q2)
    qml.RY(theta[3], wires=q3)
    qml.CNOT(wires=[q0, q1])
    qml.RZ(theta[4], wires=q1)
    qml.CNOT(wires=[q2, q3])
    qml.RZ(theta[5], wires=q3)
    qml.CNOT(wires=[q1, q2])
    qml.RZ(theta[6], wires=q2)
    qml.CNOT(wires=[q3, q0])
    qml.RZ(theta[7], wires=q0)


def pool4(phi, wires):
    q0, q1, q2, q3 = wires
    qml.CRZ(phi[0], wires=[q1, q0])
    qml.CRZ(phi[1], wires=[q3, q2])
    qml.CRX(phi[2], wires=[q2, q1])
    qml.CRX(phi[3], wires=[q0, q3])


# ----------------------------
# Embeddings
# ----------------------------
def embed_E2_local(quad_means_4):
    for i in range(4):
        qml.RY(math.pi * quad_means_4[i], wires=i)


def embed_E4_local(quad_means_4, a4, c4):
    for i in range(4):
        qml.RY(a4[i] * (math.pi * quad_means_4[i]) + c4[i], wires=i)


def embed_E4_pixels_local(pixels4, a4, c4):
    for i in range(4):
        qml.RY(a4[i] * (math.pi * pixels4[i]) + c4[i], wires=i)


def _gamma_vec_from_gA4_torch(gA4_vec: torch.Tensor) -> torch.Tensor:
    beta = torch.tensor(BETA_GLOBAL, dtype=gA4_vec.dtype, device=gA4_vec.device)  # (4,)
    return math.pi * torch.tanh(beta * gA4_vec)  # (4,)


def inject_global_on_wire(
    gA4_vec, omega=None, use_reupload=True, omega_fixed_tensor=None
):
    gammas = _gamma_vec_from_gA4_torch(gA4_vec)  # (4,)
    g0, g1, g2, g3 = gammas[0], gammas[1], gammas[2], gammas[3]

    qml.RY(g0, wires=4)
    qml.RZ(g1, wires=4)
    qml.RX(g2, wires=4)
    qml.RZ(g3, wires=4)

    if use_reupload:
        if omega is None:
            if omega_fixed_tensor is None:
                omega_fixed_tensor = torch.tensor(
                    float(CONFIG["E1NP_OMEGA_FIXED"]), dtype=torch.float32
                )
            qml.RY(omega_fixed_tensor, wires=4)
        else:
            qml.RY(omega, wires=4)

        qml.RZ(g0, wires=4)
        qml.RX(g1, wires=4)
        qml.RY(g2, wires=4)
        qml.RZ(g3, wires=4)


def fuse_global_to_locals(lam=LAMBDA_FUSION):
    for i in range(4):
        qml.CNOT(wires=[4, i])
        qml.RZ(lam, wires=i)
        qml.CNOT(wires=[4, i])


# ----------------------------
# QNodes
# ----------------------------
@qml.qnode(dev4, **QNODE4_KW)
def qnode_quadrant_E2(quad_means_4, theta_conv, phi_pool):
    embed_E2_local(quad_means_4)
    conv4(theta_conv, wires=[0, 1, 2, 3])
    pool4(phi_pool, wires=[0, 1, 2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


@qml.qnode(dev4, **QNODE4_KW)
def qnode_quadrant_E3(quad_amp_16, theta_conv, phi_pool):
    amp = torch.clamp(quad_amp_16, 0.0, 1.0)
    nrm = torch.linalg.norm(amp)
    if nrm.item() < 1e-12:
        amp = torch.zeros_like(amp)
        amp[0] = 1.0
    else:
        amp = amp / nrm
    qml.AmplitudeEmbedding(amp, wires=[0, 1, 2, 3], normalize=False)
    conv4(theta_conv, wires=[0, 1, 2, 3])
    pool4(phi_pool, wires=[0, 1, 2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


@qml.qnode(dev4, **QNODE4_KW)
def qnode_quadrant_E4(quad_means_4, a4, c4, theta_conv, phi_pool):
    embed_E4_local(quad_means_4, a4, c4)
    conv4(theta_conv, wires=[0, 1, 2, 3])
    pool4(phi_pool, wires=[0, 1, 2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


@qml.qnode(dev5, **QNODE5_KW)
def qnode_quadrant_E1_optionB(
    quad_means_4,
    gA4_vec,
    a4,
    c4,
    theta_conv,
    phi_pool,
    include_global_readout: bool,
    omega=None,
    use_reupload=True,
):
    embed_E4_local(quad_means_4, a4, c4)
    inject_global_on_wire(gA4_vec, omega=omega, use_reupload=use_reupload)
    fuse_global_to_locals(lam=LAMBDA_FUSION)
    conv4(theta_conv, wires=[0, 1, 2, 3])
    pool4(phi_pool, wires=[0, 1, 2, 3])

    outs = [qml.expval(qml.PauliZ(i)) for i in range(4)]
    if include_global_readout:
        outs.append(qml.expval(qml.PauliZ(4)))
    return outs


# %% [markdown]
# ## 4. Rete Neurale Ibrida (PyTorch)
#
# La classe `QuanvEmbedModel` incapsula l'intero processo: passa le feature quantistiche ai circuiti e raccoglie gli output per immetterli in uno strato classico terminale (Classical Head) responsabile della classificazione finale.


# %%
# ----------------------------
# Model (UNIFIED; head stays 32 for 5-class)
# ----------------------------
class QuanvEmbedModel(torch.nn.Module):
    def __init__(self, variant: str, fair_dim_match: bool = True):
        super().__init__()
        assert variant in ("E1", "E2", "E3", "E4")
        self.variant = variant
        self.fair_dim_match = bool(fair_dim_match)

        self.e1_a16 = torch.nn.Parameter(
            torch.full((16,), float(CONFIG["E1_INIT_A"]), dtype=torch.float32)
        )
        self.e1_c16 = torch.nn.Parameter(torch.zeros(16, dtype=torch.float32))

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

        self.e4_a16 = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
        self.e4_c16 = torch.nn.Parameter(torch.zeros(16, dtype=torch.float32))

        self.theta_conv = torch.nn.Parameter(0.01 * torch.randn(8, dtype=torch.float32))
        self.phi_pool = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))

        if self.variant == "E1" and (not self.fair_dim_match):
            out_dim = 20
        else:
            out_dim = 16

        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(out_dim),
            torch.nn.Linear(out_dim, 32),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.05),
            torch.nn.Linear(32, N_CLASSES),
        )

    def _quadrant_params_from_a16c16(self, a16, c16):
        a4s = torch.stack([a16[idxs] for idxs in QUAD_PATCH_IDXS], dim=0)  # (4,4)
        c4s = torch.stack([c16[idxs] for idxs in QUAD_PATCH_IDXS], dim=0)  # (4,4)
        return a4s, c4s

    def features_from_sample(self, sample: dict):
        feats = []
        qtime = 0.0
        e3_fallback_quadrants = 0

        if self.variant == "E1":
            quad_means = sample["quad_means"]
            gA4 = sample["gA4"]
            include_global = not self.fair_dim_match

            e1_a4s, e1_c4s = self._quadrant_params_from_a16c16(self.e1_a16, self.e1_c16)

            for q in range(4):
                t0 = time.time()
                out = qnode_quadrant_E1_optionB(
                    quad_means[q],
                    gA4,
                    e1_a4s[q],
                    e1_c4s[q],
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
            quad_means = sample["quad_means"]
            for q in range(4):
                t0 = time.time()
                out = qnode_quadrant_E2(quad_means[q], self.theta_conv, self.phi_pool)
                qtime += time.time() - t0
                feats.append(torch.stack(out))
            feat_vec = torch.cat(feats, dim=0)

        elif self.variant == "E3":
            quads16 = sample["quads16"]
            for q in range(4):
                amp = torch.clamp(quads16[q], 0.0, 1.0)
                if torch.linalg.norm(amp).item() < 1e-12:
                    e3_fallback_quadrants += 1
                t0 = time.time()
                out = qnode_quadrant_E3(quads16[q], self.theta_conv, self.phi_pool)
                qtime += time.time() - t0
                feats.append(torch.stack(out))
            feat_vec = torch.cat(feats, dim=0)

        else:  # E4
            quad_means = sample["quad_means"]
            a4s, c4s = self._quadrant_params_from_a16c16(self.e4_a16, self.e4_c16)
            for q in range(4):
                t0 = time.time()
                out = qnode_quadrant_E4(
                    quad_means[q], a4s[q], c4s[q], self.theta_conv, self.phi_pool
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


# %% [markdown]
# ## 5. Routine di Training e Validazione
#
# In questa cella vengono scritte le funzioni che orchestrano i loop di addestramento e validazione, il calcolo delle loss, la generazione delle stampe di diagnostica e la serializzazione dei risultati in file CSV.

# %%
# ----------------------------
# Diagnostics utils
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
            [model.e1_a16, model.e1_c16, getattr(model, "e1np_omega", None)]
        )
        if model.variant == "E1"
        else 0.0,
        "grad_e4_embed": grad_norm_of_params([model.e4_a16, model.e4_c16])
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
    quad_means_4 = torch.ones(4, dtype=torch.float32) * 0.5
    quad_amp_16 = torch.ones(16, dtype=torch.float32) * (1.0 / 16.0)
    a4 = torch.ones(4, dtype=torch.float32)
    c4 = torch.zeros(4, dtype=torch.float32)

    gA4_vec = torch.tensor([0.5, 0.1, 0.1, 0.0], dtype=torch.float32)

    theta_conv = torch.zeros(8, dtype=torch.float32)
    phi_pool = torch.zeros(4, dtype=torch.float32)

    plt.rcParams["figure.facecolor"] = "white"

    if "E1" in CONFIG["VARIANTS"]:
        plt.figure(figsize=(18, 3), facecolor="white")
        qml.draw_mpl(qnode_quadrant_E1_optionB)(
            quad_means_4,
            gA4_vec,
            a4,
            c4,
            theta_conv,
            phi_pool,
            (not CONFIG["FAIR_DIM_MATCH"]),
            omega=torch.tensor(float(CONFIG["E1NP_OMEGA_FIXED"])),
            use_reupload=CONFIG["E1NP_USE_REUPLOAD"],
        )
        plt.title("Circuit (per quadrant) — E1")
        if save and CONFIG["SAVE_FIGS"]:
            plt.savefig(
                os.path.join(ANALYSIS_DIR, "circuit_E1_optionB.png"),
                dpi=200,
                bbox_inches="tight",
            )
        plt.close()

    if "E2" in CONFIG["VARIANTS"]:
        plt.figure(figsize=(18, 3), facecolor="white")
        qml.draw_mpl(qnode_quadrant_E2)(quad_means_4, theta_conv, phi_pool)
        plt.title("Circuit (per quadrant) — E2")
        if save and CONFIG["SAVE_FIGS"]:
            plt.savefig(
                os.path.join(ANALYSIS_DIR, "circuit_E2.png"),
                dpi=200,
                bbox_inches="tight",
            )
        plt.close()

    if "E3" in CONFIG["VARIANTS"]:
        plt.figure(figsize=(18, 3), facecolor="white")
        qml.draw_mpl(qnode_quadrant_E3)(quad_amp_16, theta_conv, phi_pool)
        plt.title("Circuit (per quadrant) — E3")
        if save and CONFIG["SAVE_FIGS"]:
            plt.savefig(
                os.path.join(ANALYSIS_DIR, "circuit_E3.png"),
                dpi=200,
                bbox_inches="tight",
            )
        plt.close()

    if "E4" in CONFIG["VARIANTS"]:
        plt.figure(figsize=(18, 3), facecolor="white")
        qml.draw_mpl(qnode_quadrant_E4)(quad_means_4, a4, c4, theta_conv, phi_pool)
        plt.title("Circuit (per quadrant) — E4")
        if save and CONFIG["SAVE_FIGS"]:
            plt.savefig(
                os.path.join(ANALYSIS_DIR, "circuit_E4.png"),
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
        embed_params += [model.e1_a16, model.e1_c16, getattr(model, "e1np_omega", None)]
        embed_params = [p for p in embed_params if p is not None]
    elif variant == "E4":
        embed_params += [model.e4_a16, model.e4_c16]

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

        # grad norms via one diagnostic backward
        model.train()
        opt.zero_grad()
        diag_sample = get_features(train_data, TRAIN_CACHE, train_idx[0])
        diag_logits, _, _, _ = model(diag_sample)
        diag_y = torch.tensor([diag_sample["y"]], dtype=torch.long)
        diag_loss = ce_loss(diag_logits.view(1, -1), diag_y)
        diag_loss.backward()
        gnorms = group_grad_norms(model)
        opt.zero_grad()

        # evals
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
        "val_acc",
        "val_loss",
        "val_gap",
        "test_acc",
        "test_loss",
        "test_gap",
        "sec_epoch",
        "sec_quantum_train",
        "sec_quantum_val",
        "sec_quantum_test",
        "nparams_trainable",
        "e3_fb_test",
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
                plt.figure(figsize=(4, 4), facecolor="white")
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
# Plots (mean±std)
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


# %% [markdown]
# ## 6. Esecuzione del Training (5 Classi)
#
# Avviamo l'esecuzione. I risultati verranno salvati nella directory definita in precedenza.

# %%
# ----------------------------
# MAIN Execution
# ----------------------------
if __name__ == "__main__":
    print(f"Quantum devices: dev4={dev4_name}({diff4}), dev5={dev5_name}({diff5})")

    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print("Downloading and filtering datasets...")
    train_dataset, test_dataset = get_datasets()
    train_data = filter_and_remap(train_dataset)
    test_data = filter_and_remap(test_dataset)

    print("Classes:", TARGET_DIGITS, "->", [LABEL_MAP[d] for d in TARGET_DIGITS])
    print(
        "Filtered Train size:", len(train_data), "| Filtered Test size:", len(test_data)
    )

    if CONFIG["DO_DRAW"]:
        draw_circuits(save=True)

    metrics_df, finals_df = run_all(train_data, test_data)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY (mean±std over seeds)")
    summary_view = finals_df.groupby(["variant"])[
        ["val_acc", "val_loss", "val_gap", "test_acc", "test_loss", "test_gap"]
    ].agg(["mean", "std"])
    print(summary_view)  # Bugfix: era display(summary_view)

    plot_mean_std_curves(
        metrics_df,
        "train_acc_diag",
        "Train accuracy (diag) mean±std",
        fname="curve_train_acc_diag.png",
    )
    plot_mean_std_curves(
        metrics_df,
        "train_loss_diag",
        "Train loss (diag) mean±std",
        fname="curve_train_loss_diag.png",
    )
    plot_mean_std_curves(
        metrics_df,
        "train_gap_diag",
        "Train confidence gap (diag) mean±std",
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
        metrics_df,
        "test_acc",
        "Test accuracy mean±std (subset)",
        fname="curve_test_acc.png",
    )
    plot_mean_std_curves(
        metrics_df,
        "test_loss",
        "Test loss mean±std (subset)",
        fname="curve_test_loss.png",
    )
    plot_mean_std_curves(
        metrics_df,
        "test_gap",
        "Test confidence gap mean±std (subset)",
        fname="curve_test_gap.png",
    )

    boxplot_final_test_acc(finals_df, fname="boxplot_test_acc.png")

    print("\nSaved outputs in:", BASE_DIR)
    print("Analysis figures + tables in:", ANALYSIS_DIR)


# %% [markdown]
# # Esperimento Esteso: 10 Classi
#
# Per validare la robustezza del modello, ripetiamo l'intero setup sull'intero dataset MNIST (10 classi). Al fine di garantire la coerenza dell'ambiente ed evitare collisioni di variabili globali (es. `N_CLASSES`, `TARGET_DIGITS`, `LABEL_MAP`), rigeneriamo le funzioni e le configurazioni adattate.

# %% [markdown]
# ## 1. Configurazione Aggiornata per 10 Classi
#
# Adeguamento delle costanti e della directory di output.

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
# CONFIG (v2-uniform)
# ----------------------------
CONFIG = {
    "BASE_DIR": "./mean_1k35_clip0.5_10class",
    "VARIANTS": [
        "E1",
        "E2",
        "E3",
        "E4",
    ],
    "SEEDS": [0, 1, 2],
    "TRAIN_SAMPLES": 1000,
    "VAL_SAMPLES": 300,
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
    # ---- NEW: separate learning rates
    "LR_HEAD": 1e-2,
    "LR_QKERNEL": 1e-3,
    "LR_EMBED": 5e-4,
    "E1_INIT_A": 0.2,
}

TARGET_DIGITS = list(range(10))
LABEL_MAP = {d: i for i, d in enumerate(TARGET_DIGITS)}
N_CLASSES = 10

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
# Data (MNIST -> filter -> 8x8)
# ----------------------------
transform = transforms.Compose(
    [
        transforms.Resize((8, 8)),
        transforms.ToTensor(),
    ]
)


def get_datasets():
    # Creazione folder per i dati se non esiste
    os.makedirs("./data", exist_ok=True)
    train_dataset = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def filter_and_remap(dataset):
    out = []
    for x, y in dataset:
        y = int(y)
        if y in LABEL_MAP:
            out.append((x, LABEL_MAP[y]))
    return out


# %% [markdown]
# ## 2. Data Processing (10 Classi)


# %%
# ----------------------------
# Feature extraction (cached)
# ----------------------------
def image_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    return img_tensor.squeeze(0).cpu().numpy().astype(np.float32)


def extract_patches_2x2(X: np.ndarray) -> np.ndarray:
    patches = []
    for r in range(4):
        for c in range(4):
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
    return np.stack(patches, axis=0)  # (16,4)


def global_set_A_features(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    g1 = float(X.mean())
    g2 = float(((X - g1) ** 2).mean())
    dx = X[:, 1:] - X[:, :-1]
    dy = X[1:, :] - X[:-1, :]
    dx_o = dx[0:7, 0:7]
    dy_o = dy[0:7, 0:7]
    g3 = float((dx_o**2 + dy_o**2).mean())
    H = float((dx**2).sum())
    V = float((dy**2).sum())
    g4 = float((V - H) / (V + H + eps))
    return np.array([g1, g2, g3, g4], dtype=np.float32)


def quadrants_4x4_flat(X: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            X[0:4, 0:4].reshape(-1),
            X[0:4, 4:8].reshape(-1),
            X[4:8, 0:4].reshape(-1),
            X[4:8, 4:8].reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)  # (4,16)


QUAD_PATCH_IDXS = [
    [0, 1, 4, 5],
    [2, 3, 6, 7],
    [8, 9, 12, 13],
    [10, 11, 14, 15],
]

TRAIN_CACHE = {}
TEST_CACHE = {}


def get_features(dataset_list, cache, idx: int):
    if idx in cache:
        return cache[idx]
    x, y = dataset_list[idx]
    X = image_to_numpy(x)

    patches = extract_patches_2x2(X)  # (16,4)  values in [0,1]
    means16 = patches.mean(axis=1)  # (16,)
    quads16 = quadrants_4x4_flat(X)  # (4,16)
    gA4 = global_set_A_features(X)  # (4,)

    quad_means = np.stack([means16[q] for q in QUAD_PATCH_IDXS], axis=0)  # (4,4)

    sample = {
        "patches": torch.tensor(patches, dtype=torch.float32),  # (16,4)
        "means16": torch.tensor(means16, dtype=torch.float32),  # (16,)
        "quad_means": torch.tensor(quad_means, dtype=torch.float32),  # (4,4)
        "quads16": torch.tensor(quads16, dtype=torch.float32),  # (4,16)
        "gA4": torch.tensor(gA4, dtype=torch.float32),  # (4,)
        "y": int(y),
    }
    cache[idx] = sample
    return sample


# ----------------------------
# Stratified splits (disjoint train/val) (v2-uniform)
# ----------------------------
def stratified_split_indices(dataset_list, n_train: int, n_val: int, seed: int):
    rng = random.Random(seed)
    buckets = {c: [] for c in range(N_CLASSES)}
    for i, (_, y) in enumerate(dataset_list):
        buckets[y].append(i)

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
        buckets[y].append(i)
    per = n_samples // N_CLASSES
    rem = n_samples % N_CLASSES
    idxs = []
    for c in range(N_CLASSES):
        take = per + (1 if c < rem else 0)
        take = min(take, len(buckets[c]))
        idxs.extend(rng.sample(buckets[c], take))
    rng.shuffle(idxs)
    return idxs


# %% [markdown]
# ## 3. Circuiti Quantistici (10 Classi)


# %%
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


dev4, dev4_name, diff4 = make_device(4)
dev5, dev5_name, diff5 = make_device(5)

QNODE4_KW = dict(interface="torch", diff_method=diff4)
QNODE5_KW = dict(interface="torch", diff_method=diff5)


# ----------------------------
# Shared quantum kernel (conv + pool)
# ----------------------------
def conv4(theta, wires):
    q0, q1, q2, q3 = wires
    qml.RY(theta[0], wires=q0)
    qml.RY(theta[1], wires=q1)
    qml.RY(theta[2], wires=q2)
    qml.RY(theta[3], wires=q3)
    qml.CNOT(wires=[q0, q1])
    qml.RZ(theta[4], wires=q1)
    qml.CNOT(wires=[q2, q3])
    qml.RZ(theta[5], wires=q3)
    qml.CNOT(wires=[q1, q2])
    qml.RZ(theta[6], wires=q2)
    qml.CNOT(wires=[q3, q0])
    qml.RZ(theta[7], wires=q0)


def pool4(phi, wires):
    q0, q1, q2, q3 = wires
    qml.CRZ(phi[0], wires=[q1, q0])
    qml.CRZ(phi[1], wires=[q3, q2])
    qml.CRX(phi[2], wires=[q2, q1])
    qml.CRX(phi[3], wires=[q0, q3])


# ----------------------------
# Embeddings
# ----------------------------
def embed_E2_local(quad_means_4):
    for i in range(4):
        qml.RY(math.pi * quad_means_4[i], wires=i)


def embed_E4_local(quad_means_4, a4, c4):
    for i in range(4):
        qml.RY(a4[i] * (math.pi * quad_means_4[i]) + c4[i], wires=i)


def embed_E4_pixels_local(pixels4, a4, c4):
    for i in range(4):
        qml.RY(a4[i] * (math.pi * pixels4[i]) + c4[i], wires=i)


def _gamma_vec_from_gA4_torch(gA4_vec: torch.Tensor) -> torch.Tensor:
    beta = torch.tensor(BETA_GLOBAL, dtype=gA4_vec.dtype, device=gA4_vec.device)  # (4,)
    return math.pi * torch.tanh(beta * gA4_vec)  # (4,)


def inject_global_on_wire(
    gA4_vec, omega=None, use_reupload=True, omega_fixed_tensor=None
):
    gammas = _gamma_vec_from_gA4_torch(gA4_vec)  # (4,)
    g0, g1, g2, g3 = gammas[0], gammas[1], gammas[2], gammas[3]

    qml.RY(g0, wires=4)
    qml.RZ(g1, wires=4)
    qml.RX(g2, wires=4)
    qml.RZ(g3, wires=4)

    if use_reupload:
        if omega is None:
            if omega_fixed_tensor is None:
                omega_fixed_tensor = torch.tensor(
                    float(CONFIG["E1NP_OMEGA_FIXED"]), dtype=torch.float32
                )
            qml.RY(omega_fixed_tensor, wires=4)
        else:
            qml.RY(omega, wires=4)

        qml.RZ(g0, wires=4)
        qml.RX(g1, wires=4)
        qml.RY(g2, wires=4)
        qml.RZ(g3, wires=4)


def fuse_global_to_locals(lam=LAMBDA_FUSION):
    for i in range(4):
        qml.CNOT(wires=[4, i])
        qml.RZ(lam, wires=i)
        qml.CNOT(wires=[4, i])


# ----------------------------
# QNodes
# ----------------------------
@qml.qnode(dev4, **QNODE4_KW)
def qnode_quadrant_E2(quad_means_4, theta_conv, phi_pool):
    embed_E2_local(quad_means_4)
    conv4(theta_conv, wires=[0, 1, 2, 3])
    pool4(phi_pool, wires=[0, 1, 2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


@qml.qnode(dev4, **QNODE4_KW)
def qnode_quadrant_E3(quad_amp_16, theta_conv, phi_pool):
    amp = torch.clamp(quad_amp_16, 0.0, 1.0)
    nrm = torch.linalg.norm(amp)
    if nrm.item() < 1e-12:
        amp = torch.zeros_like(amp)
        amp[0] = 1.0
    else:
        amp = amp / nrm
    qml.AmplitudeEmbedding(amp, wires=[0, 1, 2, 3], normalize=False)
    conv4(theta_conv, wires=[0, 1, 2, 3])
    pool4(phi_pool, wires=[0, 1, 2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


@qml.qnode(dev4, **QNODE4_KW)
def qnode_quadrant_E4(quad_means_4, a4, c4, theta_conv, phi_pool):
    embed_E4_local(quad_means_4, a4, c4)
    conv4(theta_conv, wires=[0, 1, 2, 3])
    pool4(phi_pool, wires=[0, 1, 2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


@qml.qnode(dev5, **QNODE5_KW)
def qnode_quadrant_E1_optionB(
    quad_means_4,
    gA4_vec,
    a4,
    c4,
    theta_conv,
    phi_pool,
    include_global_readout: bool,
    omega=None,
    use_reupload=True,
):
    embed_E4_local(quad_means_4, a4, c4)
    inject_global_on_wire(gA4_vec, omega=omega, use_reupload=use_reupload)
    fuse_global_to_locals(lam=LAMBDA_FUSION)
    conv4(theta_conv, wires=[0, 1, 2, 3])
    pool4(phi_pool, wires=[0, 1, 2, 3])

    outs = [qml.expval(qml.PauliZ(i)) for i in range(4)]
    if include_global_readout:
        outs.append(qml.expval(qml.PauliZ(4)))
    return outs


# %% [markdown]
# ## 4. Modello Ibrido Scalato (10 Classi)
#
# La Classical Head viene ingrandita per gestire 10 classi di output.


# %%
# ----------------------------
# Model (UNIFIED; head stays 32 for 5-class)
# ----------------------------
class QuanvEmbedModel(torch.nn.Module):
    def __init__(self, variant: str, fair_dim_match: bool = True):
        super().__init__()
        assert variant in ("E1", "E2", "E3", "E4")
        self.variant = variant
        self.fair_dim_match = bool(fair_dim_match)

        self.e1_a16 = torch.nn.Parameter(
            torch.full((16,), float(CONFIG["E1_INIT_A"]), dtype=torch.float32)
        )
        self.e1_c16 = torch.nn.Parameter(torch.zeros(16, dtype=torch.float32))

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

        self.e4_a16 = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
        self.e4_c16 = torch.nn.Parameter(torch.zeros(16, dtype=torch.float32))

        self.theta_conv = torch.nn.Parameter(0.01 * torch.randn(8, dtype=torch.float32))
        self.phi_pool = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))

        if self.variant == "E1" and (not self.fair_dim_match):
            out_dim = 20
        else:
            out_dim = 16

        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(out_dim),
            torch.nn.Linear(out_dim, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.05),
            torch.nn.Linear(64, N_CLASSES),
        )

    def _quadrant_params_from_a16c16(self, a16, c16):
        a4s = torch.stack([a16[idxs] for idxs in QUAD_PATCH_IDXS], dim=0)  # (4,4)
        c4s = torch.stack([c16[idxs] for idxs in QUAD_PATCH_IDXS], dim=0)  # (4,4)
        return a4s, c4s

    def features_from_sample(self, sample: dict):
        feats = []
        qtime = 0.0
        e3_fallback_quadrants = 0

        if self.variant == "E1":
            quad_means = sample["quad_means"]
            gA4 = sample["gA4"]
            include_global = not self.fair_dim_match

            e1_a4s, e1_c4s = self._quadrant_params_from_a16c16(self.e1_a16, self.e1_c16)

            for q in range(4):
                t0 = time.time()
                out = qnode_quadrant_E1_optionB(
                    quad_means[q],
                    gA4,
                    e1_a4s[q],
                    e1_c4s[q],
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
            quad_means = sample["quad_means"]
            for q in range(4):
                t0 = time.time()
                out = qnode_quadrant_E2(quad_means[q], self.theta_conv, self.phi_pool)
                qtime += time.time() - t0
                feats.append(torch.stack(out))
            feat_vec = torch.cat(feats, dim=0)

        elif self.variant == "E3":
            quads16 = sample["quads16"]
            for q in range(4):
                amp = torch.clamp(quads16[q], 0.0, 1.0)
                if torch.linalg.norm(amp).item() < 1e-12:
                    e3_fallback_quadrants += 1
                t0 = time.time()
                out = qnode_quadrant_E3(quads16[q], self.theta_conv, self.phi_pool)
                qtime += time.time() - t0
                feats.append(torch.stack(out))
            feat_vec = torch.cat(feats, dim=0)

        else:  # E4
            quad_means = sample["quad_means"]
            a4s, c4s = self._quadrant_params_from_a16c16(self.e4_a16, self.e4_c16)
            for q in range(4):
                t0 = time.time()
                out = qnode_quadrant_E4(
                    quad_means[q], a4s[q], c4s[q], self.theta_conv, self.phi_pool
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


# %% [markdown]
# ## 5. Routine di Addestramento (10 Classi)

# %%
# ----------------------------
# Diagnostics utils
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
            [model.e1_a16, model.e1_c16, getattr(model, "e1np_omega", None)]
        )
        if model.variant == "E1"
        else 0.0,
        "grad_e4_embed": grad_norm_of_params([model.e4_a16, model.e4_c16])
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
    quad_means_4 = torch.ones(4, dtype=torch.float32) * 0.5
    quad_amp_16 = torch.ones(16, dtype=torch.float32) * (1.0 / 16.0)
    a4 = torch.ones(4, dtype=torch.float32)
    c4 = torch.zeros(4, dtype=torch.float32)

    gA4_vec = torch.tensor([0.5, 0.1, 0.1, 0.0], dtype=torch.float32)

    theta_conv = torch.zeros(8, dtype=torch.float32)
    phi_pool = torch.zeros(4, dtype=torch.float32)

    plt.rcParams["figure.facecolor"] = "white"

    if "E1" in CONFIG["VARIANTS"]:
        plt.figure(figsize=(18, 3), facecolor="white")
        qml.draw_mpl(qnode_quadrant_E1_optionB)(
            quad_means_4,
            gA4_vec,
            a4,
            c4,
            theta_conv,
            phi_pool,
            (not CONFIG["FAIR_DIM_MATCH"]),
            omega=torch.tensor(float(CONFIG["E1NP_OMEGA_FIXED"])),
            use_reupload=CONFIG["E1NP_USE_REUPLOAD"],
        )
        plt.title("Circuit (per quadrant) — E1")
        if save and CONFIG["SAVE_FIGS"]:
            plt.savefig(
                os.path.join(ANALYSIS_DIR, "circuit_E1_optionB.png"),
                dpi=200,
                bbox_inches="tight",
            )
        plt.close()

    if "E2" in CONFIG["VARIANTS"]:
        plt.figure(figsize=(18, 3), facecolor="white")
        qml.draw_mpl(qnode_quadrant_E2)(quad_means_4, theta_conv, phi_pool)
        plt.title("Circuit (per quadrant) — E2")
        if save and CONFIG["SAVE_FIGS"]:
            plt.savefig(
                os.path.join(ANALYSIS_DIR, "circuit_E2.png"),
                dpi=200,
                bbox_inches="tight",
            )
        plt.close()

    if "E3" in CONFIG["VARIANTS"]:
        plt.figure(figsize=(18, 3), facecolor="white")
        qml.draw_mpl(qnode_quadrant_E3)(quad_amp_16, theta_conv, phi_pool)
        plt.title("Circuit (per quadrant) — E3")
        if save and CONFIG["SAVE_FIGS"]:
            plt.savefig(
                os.path.join(ANALYSIS_DIR, "circuit_E3.png"),
                dpi=200,
                bbox_inches="tight",
            )
        plt.close()

    if "E4" in CONFIG["VARIANTS"]:
        plt.figure(figsize=(18, 3), facecolor="white")
        qml.draw_mpl(qnode_quadrant_E4)(quad_means_4, a4, c4, theta_conv, phi_pool)
        plt.title("Circuit (per quadrant) — E4")
        if save and CONFIG["SAVE_FIGS"]:
            plt.savefig(
                os.path.join(ANALYSIS_DIR, "circuit_E4.png"),
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
        embed_params += [model.e1_a16, model.e1_c16, getattr(model, "e1np_omega", None)]
        embed_params = [p for p in embed_params if p is not None]
    elif variant == "E4":
        embed_params += [model.e4_a16, model.e4_c16]

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

        # grad norms via one diagnostic backward
        model.train()
        opt.zero_grad()
        diag_sample = get_features(train_data, TRAIN_CACHE, train_idx[0])
        diag_logits, _, _, _ = model(diag_sample)
        diag_y = torch.tensor([diag_sample["y"]], dtype=torch.long)
        diag_loss = ce_loss(diag_logits.view(1, -1), diag_y)
        diag_loss.backward()
        gnorms = group_grad_norms(model)
        opt.zero_grad()

        # evals
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
        "val_acc",
        "val_loss",
        "val_gap",
        "test_acc",
        "test_loss",
        "test_gap",
        "sec_epoch",
        "sec_quantum_train",
        "sec_quantum_val",
        "sec_quantum_test",
        "nparams_trainable",
        "e3_fb_test",
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
                plt.figure(figsize=(4, 4), facecolor="white")
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
# Plots (mean±std)
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


# %% [markdown]
# ## 6. Esecuzione del Training (10 Classi)

# %%
# ----------------------------
# MAIN Execution
# ----------------------------
if __name__ == "__main__":
    print(f"Quantum devices: dev4={dev4_name}({diff4}), dev5={dev5_name}({diff5})")

    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    print("Downloading and filtering datasets...")
    train_dataset, test_dataset = get_datasets()
    train_data = filter_and_remap(train_dataset)
    test_data = filter_and_remap(test_dataset)

    print("Classes:", TARGET_DIGITS, "->", [LABEL_MAP[d] for d in TARGET_DIGITS])
    print(
        "Filtered Train size:", len(train_data), "| Filtered Test size:", len(test_data)
    )

    if CONFIG["DO_DRAW"]:
        draw_circuits(save=True)

    metrics_df, finals_df = run_all(train_data, test_data)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY (mean±std over seeds)")
    summary_view = finals_df.groupby(["variant"])[
        ["val_acc", "val_loss", "val_gap", "test_acc", "test_loss", "test_gap"]
    ].agg(["mean", "std"])
    print(summary_view)  # Bugfix: era display(summary_view)

    plot_mean_std_curves(
        metrics_df,
        "train_acc_diag",
        "Train accuracy (diag) mean±std",
        fname="curve_train_acc_diag.png",
    )
    plot_mean_std_curves(
        metrics_df,
        "train_loss_diag",
        "Train loss (diag) mean±std",
        fname="curve_train_loss_diag.png",
    )
    plot_mean_std_curves(
        metrics_df,
        "train_gap_diag",
        "Train confidence gap (diag) mean±std",
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
        metrics_df,
        "test_acc",
        "Test accuracy mean±std (subset)",
        fname="curve_test_acc.png",
    )
    plot_mean_std_curves(
        metrics_df,
        "test_loss",
        "Test loss mean±std (subset)",
        fname="curve_test_loss.png",
    )
    plot_mean_std_curves(
        metrics_df,
        "test_gap",
        "Test confidence gap mean±std (subset)",
        fname="curve_test_gap.png",
    )

    boxplot_final_test_acc(finals_df, fname="boxplot_test_acc.png")

    print("\nSaved outputs in:", BASE_DIR)
    print("Analysis figures + tables in:", ANALYSIS_DIR)


# %% [markdown]
# # Analisi Visiva e Valutazione delle Prestazioni
#
# Attraverso la libreria `matplotlib`, generiamo i grafici che mostrano l'andamento della loss, dell'accuracy e delle confidenze medie (gap) al variare delle epoche. L'utilizzo di boxplot permette un confronto statistico tra le run eseguite con seed differenti.

# %%
# ============================================================
# PLOTS FROM CSV (ALL_metrics_5/10.csv + ALL_final_eval_5/10.csv)
# - Uses matplotlib only
# - Produces: mean±std curves vs epoch, and boxplots (final)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load CSVs (adjust paths if needed)
# ----------------------------
METRICS_5_PATH = "./mean_1k35_clip0.5_5class/ALL_metrics.csv"
METRICS_10_PATH = "./mean_1k35_clip0.5_10class/ALL_metrics.csv"
FINALS_5_PATH = "./mean_1k35_clip0.5_5class/ALL_final_eval.csv"
FINALS_10_PATH = "./mean_1k35_clip0.5_10class/ALL_final_eval.csv"

m5 = pd.read_csv(METRICS_5_PATH)
m10 = pd.read_csv(METRICS_10_PATH)
f5 = pd.read_csv(FINALS_5_PATH)
f10 = pd.read_csv(FINALS_10_PATH)

# Optional output dir for saving figures
OUTDIR = "./plots_results"
os.makedirs(OUTDIR, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def _mean_std_by_epoch(df: pd.DataFrame, ycol: str, variant: str):
    sub = df[df["variant"] == variant].copy()
    grp = sub.groupby("epoch")[ycol]
    mean = grp.mean()
    std = grp.std()
    return mean, std


def plot_mean_std_curves(df: pd.DataFrame, ycol: str, title: str, savepath: str = None):
    plt.figure(figsize=(7, 4))
    variants = sorted(df["variant"].unique())
    for v in variants:
        mean, std = _mean_std_by_epoch(df, ycol, v)
        x = mean.index.values
        plt.plot(x, mean.values, label=v)
        # fill only where std is defined
        std_vals = std.values
        std_vals = np.nan_to_num(std_vals, nan=0.0)
        plt.fill_between(
            x, (mean.values - std_vals), (mean.values + std_vals), alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def boxplot_final_metric(
    finals_df: pd.DataFrame, ycol: str, title: str, savepath: str = None
):
    plt.figure(figsize=(6.5, 4))
    variants = sorted(finals_df["variant"].unique())
    data = [finals_df[finals_df["variant"] == v][ycol].values for v in variants]
    plt.boxplot(data, labels=variants)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


def summarize_grad_peaks(df: pd.DataFrame, col: str):
    # Useful sanity check to see if there are peaks but still finite
    s = df[col]
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "nan": int(s.isna().sum()),
        "inf": int(np.isinf(s).sum()),
    }


# ============================================================
# 6.2-RELEVANT PLOTS (Training Stability & Optimization)
# ============================================================

# ---- Gradient norms vs epoch (mean±std)
for setting_name, df in [("5class", m5), ("10class", m10)]:
    plot_mean_std_curves(
        df,
        "grad_qkernel",
        f"Gradient norm (quantum kernel) mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"grad_qkernel_{setting_name}.png"),
    )
    plot_mean_std_curves(
        df,
        "grad_head",
        f"Gradient norm (classical head) mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"grad_head_{setting_name}.png"),
    )
    # embedding gradients (E1 and E4 are meaningful, others are zero)
    plot_mean_std_curves(
        df,
        "grad_e1_embed",
        f"Gradient norm (E1 embedding params) mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"grad_e1_embed_{setting_name}.png"),
    )
    plot_mean_std_curves(
        df,
        "grad_e4_embed",
        f"Gradient norm (E4 embedding params) mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"grad_e4_embed_{setting_name}.png"),
    )

# ---- Training diagnostics vs epoch (mean±std)
for setting_name, df in [("5class", m5), ("10class", m10)]:
    plot_mean_std_curves(
        df,
        "train_acc_diag",
        f"Train accuracy (diagnostic subset) mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"train_acc_diag_{setting_name}.png"),
    )
    plot_mean_std_curves(
        df,
        "train_loss_diag",
        f"Train loss (diagnostic subset) mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"train_loss_diag_{setting_name}.png"),
    )
    plot_mean_std_curves(
        df,
        "train_gap_diag",
        f"Train confidence gap (diagnostic subset) mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"train_gap_diag_{setting_name}.png"),
    )


# ---- E3 fallback counts (tables and optional curves)
def e3_fallback_table(df: pd.DataFrame, setting_name: str):
    e3 = df[df["variant"] == "E3"].copy()
    cols = [
        "e3_fallback_quadrants_train_total",
        "e3_fallback_quadrants_val_total",
        "e3_fallback_quadrants_test_total",
    ]
    agg = e3.groupby(["epoch"])[cols].agg(["mean", "std", "min", "max"])
    print(f"\nE3 fallback summary by epoch — {setting_name}")
    display(agg.head(10))
    return agg


fb5 = e3_fallback_table(m5, "5class")
fb10 = e3_fallback_table(m10, "10class")


# Optional: plot fallback vs epoch (mean±std) for E3
def plot_e3_fallback(df: pd.DataFrame, col: str, title: str, savepath: str = None):
    e3 = df[df["variant"] == "E3"].copy()
    grp = e3.groupby("epoch")[col]
    mean = grp.mean()
    std = grp.std().fillna(0.0)
    x = mean.index.values
    plt.figure(figsize=(7, 4))
    plt.plot(x, mean.values, label="E3")
    plt.fill_between(
        x, (mean.values - std.values), (mean.values + std.values), alpha=0.2
    )
    plt.xlabel("Epoch")
    plt.ylabel(col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()


plot_e3_fallback(
    m5,
    "e3_fallback_quadrants_test_total",
    "E3 fallback activations (test) mean±std — 5class",
    savepath=os.path.join(OUTDIR, "e3_fallback_test_5class.png"),
)

plot_e3_fallback(
    m10,
    "e3_fallback_quadrants_test_total",
    "E3 fallback activations (test) mean±std — 10class",
    savepath=os.path.join(OUTDIR, "e3_fallback_test_10class.png"),
)

# ============================================================
# 6.1-RELEVANT PLOTS (Overall Performance Comparison)
# ============================================================

for setting_name, df in [("5class", m5), ("10class", m10)]:
    plot_mean_std_curves(
        df,
        "val_acc",
        f"Validation accuracy mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"val_acc_{setting_name}.png"),
    )
    plot_mean_std_curves(
        df,
        "test_acc",
        f"Test accuracy mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"test_acc_{setting_name}.png"),
    )
    plot_mean_std_curves(
        df,
        "val_loss",
        f"Validation loss mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"val_loss_{setting_name}.png"),
    )
    plot_mean_std_curves(
        df,
        "test_loss",
        f"Test loss mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"test_loss_{setting_name}.png"),
    )
    plot_mean_std_curves(
        df,
        "val_gap",
        f"Validation confidence gap mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"val_gap_{setting_name}.png"),
    )
    plot_mean_std_curves(
        df,
        "test_gap",
        f"Test confidence gap mean±std — {setting_name}",
        savepath=os.path.join(OUTDIR, f"test_gap_{setting_name}.png"),
    )

# Final boxplots (multi-seed, last epoch per run already in finals)
for setting_name, finals in [("5class", f5), ("10class", f10)]:
    boxplot_final_metric(
        finals,
        "test_acc",
        f"Final test accuracy (multi-seed) — {setting_name}",
        savepath=os.path.join(OUTDIR, f"box_test_acc_{setting_name}.png"),
    )
    boxplot_final_metric(
        finals,
        "test_gap",
        f"Final test confidence gap (multi-seed) — {setting_name}",
        savepath=os.path.join(OUTDIR, f"box_test_gap_{setting_name}.png"),
    )

# ============================================================
# Quick sanity checks (optional prints)
# ============================================================
print("\nSanity check grad peaks (5class):", summarize_grad_peaks(m5, "grad_qkernel"))
print("Sanity check grad peaks (10class):", summarize_grad_peaks(m10, "grad_qkernel"))
print("\nSaved plots in:", OUTDIR)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================
PLOTS_DIR = "plots_results"
os.makedirs(PLOTS_DIR, exist_ok=True)

m5 = pd.read_csv(METRICS_5_PATH)
f5 = pd.read_csv(FINALS_5_PATH)
m10 = pd.read_csv(METRICS_10_PATH)
f10 = pd.read_csv(FINALS_10_PATH)

VARIANTS = ["E1", "E2", "E3", "E4"]


# ============================================================
# Helpers
# ============================================================
def add_quantum_totals(df):
    df = df.copy()
    df["sec_quantum_total"] = (
        df["sec_quantum_train"] + df["sec_quantum_val"] + df["sec_quantum_test"]
    )
    df["quantum_frac_epoch"] = df["sec_quantum_total"] / df["sec_epoch"]
    return df


m5 = add_quantum_totals(m5)
m10 = add_quantum_totals(m10)
f5 = add_quantum_totals(f5)
f10 = add_quantum_totals(f10)

plt.rcParams["figure.facecolor"] = "white"


# ============================================================
# 1) Quantum time vs epoch (train / val / test / total)
# ============================================================
def plot_quantum_time_curves(metrics_df, tag):
    for col in [
        "sec_quantum_train",
        "sec_quantum_val",
        "sec_quantum_test",
        "sec_quantum_total",
    ]:
        plt.figure(figsize=(7, 4))
        for v in VARIANTS:
            sub = metrics_df[metrics_df["variant"] == v]
            grp = sub.groupby("epoch")[col]
            mean = grp.mean()
            std = grp.std()
            plt.plot(mean.index, mean.values, label=v)
            plt.fill_between(
                mean.index, (mean - std).values, (mean + std).values, alpha=0.2
            )
        plt.xlabel("Epoch")
        plt.ylabel(col)
        plt.title(f"{tag} | {col}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_{col}_vs_epoch.png"), dpi=200)
        plt.close()


plot_quantum_time_curves(m5, "5class")
plot_quantum_time_curves(m10, "10class")


# ============================================================
# 2) Fraction of epoch time spent in quantum execution
# ============================================================
def plot_quantum_fraction(metrics_df, tag):
    plt.figure(figsize=(7, 4))
    for v in VARIANTS:
        sub = metrics_df[metrics_df["variant"] == v]
        grp = sub.groupby("epoch")["quantum_frac_epoch"]
        mean = grp.mean()
        std = grp.std()
        plt.plot(mean.index, mean.values, label=v)
        plt.fill_between(
            mean.index, (mean - std).values, (mean + std).values, alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("Quantum fraction of epoch time")
    plt.title(f"{tag} | Quantum fraction vs epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{tag}_quantum_fraction_vs_epoch.png"), dpi=200
    )
    plt.close()


plot_quantum_fraction(m5, "5class")
plot_quantum_fraction(m10, "10class")


# ============================================================
# 3) Boxplot: per-epoch total quantum time
# ============================================================
def boxplot_quantum_total(metrics_df, tag):
    plt.figure(figsize=(7, 4))
    data = [
        metrics_df[metrics_df["variant"] == v]["sec_quantum_total"].values
        for v in VARIANTS
    ]
    plt.boxplot(data, labels=VARIANTS)
    plt.ylabel("Total quantum time per epoch (s)")
    plt.title(f"{tag} | Quantum time distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_boxplot_quantum_time.png"), dpi=200)
    plt.close()


boxplot_quantum_total(m5, "5class")
boxplot_quantum_total(m10, "10class")


# ============================================================
# 4) Accuracy vs quantum time (trade-off)
# ============================================================
def scatter_accuracy_vs_time(final_df, tag):
    plt.figure(figsize=(6, 4))
    for v in VARIANTS:
        sub = final_df[final_df["variant"] == v]
        plt.scatter(sub["sec_quantum_total"], sub["test_acc"], label=v)
    plt.xlabel("Total quantum time per epoch (s)")
    plt.ylabel("Final test accuracy")
    plt.title(f"{tag} | Accuracy vs quantum time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_accuracy_vs_quantum_time.png"), dpi=200)
    plt.close()


scatter_accuracy_vs_time(f5, "5class")
scatter_accuracy_vs_time(f10, "10class")


# ============================================================
# 5) Epoch wallclock vs quantum time (sanity check)
# ============================================================
def scatter_epoch_vs_quantum(metrics_df, tag):
    plt.figure(figsize=(6, 4))
    for v in VARIANTS:
        sub = metrics_df[metrics_df["variant"] == v]
        plt.scatter(sub["sec_quantum_total"], sub["sec_epoch"], label=v, alpha=0.6)
    plt.xlabel("Total quantum time per epoch (s)")
    plt.ylabel("Epoch wallclock time (s)")
    plt.title(f"{tag} | Epoch time vs quantum time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_epoch_vs_quantum_time.png"), dpi=200)
    plt.close()


scatter_epoch_vs_quantum(m5, "5class")
scatter_epoch_vs_quantum(m10, "10class")

print(f"All plots saved in: {PLOTS_DIR}")


# %% [markdown]
# # Esportazione degli Output
#
# Eseguiamo la compressione delle directory contenenti i risultati tabellari (CSV) e i grafici generati, permettendo così il download in locale dei dati elaborati.

# %%
# ============================================================
# ZIP EXPORT (Locale) per directory complete
# ============================================================

import os
import zipfile


DIRS_TO_DOWNLOAD = [
    "./mean_1k35_clip0.5_5class",
    "./mean_1k35_clip0.5_10class",
    "./plots_results",
]


def zip_dir(src_dir: str, zip_path: str):
    """Crea uno zip includendo TUTTI i file e sottocartelle di src_dir."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, filenames in os.walk(src_dir):
            for fn in filenames:
                abs_path = os.path.join(root, fn)
                # Mantiene nello zip un percorso relativo "pulito" rispetto alla cartella parent di src_dir
                rel_path = os.path.relpath(abs_path, start=os.path.dirname(src_dir))
                zf.write(abs_path, arcname=rel_path)


for d in DIRS_TO_DOWNLOAD:
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Directory non trovata: {d}")

    base_name = os.path.basename(d.rstrip("/"))
    zip_path = os.path.abspath(f"{base_name}.zip")

    print(f"[1/2] Zipping: {d}  ->  {zip_path}")
    zip_dir(d, zip_path)

    print(f"[2/2] Download: {zip_path}")
    print(f"Archivio pronto: {zip_path}")

print("Fatto.")
