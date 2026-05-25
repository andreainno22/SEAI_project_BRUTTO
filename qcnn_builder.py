"""
qcnn_builder.py — Dynamic QCNN builder for the ablation test suite.

Supports:
  - Encoding   : "amplitude" (8 qubit, AmplitudeEmbedding + norm injection)
                 "E1"        (9 qubit, trainable affine angle + global ancilla)
  - Ansatz     : "v2_deep"     — conv8 -> pool(8->4) -> conv4_retained  [32 params]
                 "qfix_shallow"— conv8 -> pool(8->4) -> readout          [24 params]
                 "flat_no_pool"— conv8 -> readout (all 8 wires)          [16 params]
  - Readout    : "Z"    — expval(PauliZ) on kept wires
                 "Z_ZZ" — expval(PauliZ) + expval(PauliZ@PauliZ) ring

Design constraints:
  - QNodes receive only plain tensors (no dicts): PennyLane adjoint requires
    a static tape whose inputs are differentiable tensors.
  - All classical pre-processing (gamma computation for E1, norm angle for
    amplitude) is performed OUTSIDE the QNode, before the call site.
  - Batch dimension is handled by PennyLane automatically when tensors of
    shape (B, N) are passed as inputs.
"""

import math
import torch
import torch.nn as nn
import pennylane as qml
from typing import List, Tuple

EPS = 1e-8
BETA_GLOBAL = [1.0, 10.0, 10.0, 1.0]   # scaling for tanh compression of gA4


# ============================================================
# Quantum Layer Functions
# ============================================================

def conv8(theta_conv):
    """8-qubit convolutional layer.

    Structure (16 params):
      - 8 single-qubit RY rotations
      - 4 even-pair CNOT-RZ-CNOT entanglers  (wires 0-1, 2-3, 4-5, 6-7)
      - 4 shifted CNOT-RZ-CNOT entanglers    (wires 1-2, 3-4, 5-6, 7-0)
    """
    for i in range(8):
        qml.RY(theta_conv[i], wires=i)
    for a, b, k in [(0, 1, 8), (2, 3, 9), (4, 5, 10), (6, 7, 11)]:
        qml.CNOT(wires=[a, b])
        qml.RZ(theta_conv[k], wires=b)
        qml.CNOT(wires=[a, b])
    for a, b, k in [(1, 2, 12), (3, 4, 13), (5, 6, 14), (7, 0, 15)]:
        qml.CNOT(wires=[a, b])
        qml.RZ(theta_conv[k], wires=b)
        qml.CNOT(wires=[a, b])


def pool_8_to_4(phi_pool):
    """Pooling layer: 8 -> 4 retained qubits [0, 2, 4, 6].

    Uses CNOT-sandwich (RY -> CNOT -> RZ -> CNOT) instead of CRX/CRZ.
    CRX and CRZ produce zero gradients with lightning.qubit adjoint;
    the CNOT-sandwich uses only gates that are fully adjoint-differentiable.

    8 params (2 per pair: RY angle + RZ angle).
    """
    pairs = [(1, 0), (3, 2), (5, 4), (7, 6)]  # (control, target)
    for j, (control, target) in enumerate(pairs):
        qml.RY(phi_pool[2 * j],     wires=target)
        qml.CNOT(wires=[control, target])
        qml.RZ(phi_pool[2 * j + 1], wires=target)
        qml.CNOT(wires=[control, target])


def conv4_retained(theta_conv2):
    """Second convolutional layer on the 4 retained wires [0, 2, 4, 6].

    Structure (8 params):
      - 4 single-qubit RY rotations
      - 4 CNOT-RZ-CNOT entanglers forming a ring: 0->2, 2->4, 4->6, 6->0
    """
    keep_wires = [0, 2, 4, 6]
    for k, w in enumerate(keep_wires):
        qml.RY(theta_conv2[k], wires=w)
    for k in range(len(keep_wires) - 1):
        a, b = keep_wires[k], keep_wires[k + 1]
        qml.CNOT(wires=[a, b])
        qml.RZ(theta_conv2[4 + k], wires=b)
        qml.CNOT(wires=[a, b])
    # Close ring: wire 6 -> wire 0
    qml.CNOT(wires=[keep_wires[-1], keep_wires[0]])
    qml.RZ(theta_conv2[7], wires=keep_wires[0])
    qml.CNOT(wires=[keep_wires[-1], keep_wires[0]])


# ============================================================
# E1 Pre-computation (runs OUTSIDE the QNode)
# ============================================================

def compute_gamma(gA4_vec: torch.Tensor) -> torch.Tensor:
    """Compress global features gA4 into 4 rotation angles in [-pi, pi].

    Called OUTSIDE the QNode so that torch operations don't appear on the
    PennyLane tape (PennyLane adjoint requires a static, pure-quantum tape).

    Args:
        gA4_vec: (B, 4) or (4,) tensor of global image statistics.
    Returns:
        gammas: same shape as input, values in (-pi, pi).
    """
    beta = gA4_vec.new_tensor(BETA_GLOBAL)   # same dtype/device as input
    return math.pi * torch.tanh(beta * gA4_vec)


# ============================================================
# QNode Builders — one per encoding type
# (Separate QNodes are needed because PennyLane adjoint requires
#  a static tape; mixing runtime-conditional wires/operations on
#  the same QNode is not allowed.)
# ============================================================

def _make_device(n_wires: int):
    try:
        dev = qml.device("lightning.qubit", wires=n_wires)
        return dev, "adjoint"
    except Exception:
        dev = qml.device("default.qubit", wires=n_wires)
        return dev, "backprop"


def _build_amplitude_qnode(config: dict, keep_wires: List[int],
                            zz_pairs: List[Tuple[int, int]]):
    """QNode for amplitude encoding. Inputs: amp256, norm_angle, theta_q."""
    dev, diff_method = _make_device(8)
    ansatz_type = config["ANSATZ_TYPE"]
    readout_type = config["READOUT"]
    inject_norm  = config["INJECT_NORM"]

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def qnode(amp256, norm_angle, theta_q):
        # 1. Encoding
        qml.AmplitudeEmbedding(amp256, wires=range(8), normalize=True)
        if inject_norm:
            qml.RY(norm_angle, wires=0)

        # 2. Ansatz
        conv8(theta_q[0:16])
        if ansatz_type == "v2_deep":
            pool_8_to_4(theta_q[16:24])
            conv4_retained(theta_q[24:32])
        elif ansatz_type == "qfix_shallow":
            pool_8_to_4(theta_q[16:24])
        # "flat_no_pool": no further gates

        # 3. Readout
        z_obs  = [qml.expval(qml.PauliZ(i)) for i in keep_wires]
        if readout_type == "Z_ZZ":
            zz_obs = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in zz_pairs]
            return z_obs + zz_obs
        return z_obs

    return qnode


def _build_E1_qnode(config: dict, keep_wires: List[int],
                     zz_pairs: List[Tuple[int, int]]):
    """QNode for E1 encoding (9 wires: 8 local + 1 global ancilla on wire 8).

    Inputs: quad_means (8,), gammas (4,), a_embed (8,), c_embed (8,), theta_q.
    gamma is pre-computed outside the QNode (see compute_gamma).
    """
    dev, diff_method = _make_device(9)
    ansatz_type  = config["ANSATZ_TYPE"]
    readout_type = config["READOUT"]
    lam          = config["LAMBDA_FUSION"]
    omega_fixed  = config["E1_OMEGA_FIXED"]

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def qnode(quad_means, gammas, a_embed, c_embed, theta_q):
        # 1a. Local affine angle embedding on wires 0-7
        for i in range(8):
            qml.RY(a_embed[i] * (math.pi * quad_means[i]) + c_embed[i], wires=i)

        # 1b. Global ancilla encoding (wire 8) — gammas pre-computed outside
        qml.RY(gammas[0], wires=8)
        qml.RZ(gammas[1], wires=8)
        qml.RX(gammas[2], wires=8)
        qml.RZ(gammas[3], wires=8)
        qml.RY(omega_fixed, wires=8)  # re-upload separator (fixed omega = E1_OMEGA_FIXED)
        qml.RZ(gammas[0], wires=8)
        qml.RX(gammas[1], wires=8)
        qml.RY(gammas[2], wires=8)
        qml.RZ(gammas[3], wires=8)

        # 1c. Fusion: entangle global ancilla with each local wire
        for i in range(8):
            qml.CNOT(wires=[8, i])
            qml.RZ(lam, wires=i)
            qml.CNOT(wires=[8, i])

        # 2. Ansatz (same as amplitude, on wires 0-7)
        conv8(theta_q[0:16])
        if ansatz_type == "v2_deep":
            pool_8_to_4(theta_q[16:24])
            conv4_retained(theta_q[24:32])
        elif ansatz_type == "qfix_shallow":
            pool_8_to_4(theta_q[16:24])
        # "flat_no_pool": no further gates

        # 3. Readout (local wires only, ancilla wire 8 is traced out)
        z_obs  = [qml.expval(qml.PauliZ(i)) for i in keep_wires]
        if readout_type == "Z_ZZ":
            zz_obs = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in zz_pairs]
            return z_obs + zz_obs
        return z_obs

    return qnode


def build_qnode(config: dict):
    """Factory: returns (qnode, keep_wires, zz_pairs) for the given config."""
    ansatz_type  = config["ANSATZ_TYPE"]
    readout_type = config["READOUT"]
    encoding     = config["ENCODING"]

    if ansatz_type in ("v2_deep", "qfix_shallow"):
        keep_wires = [0, 2, 4, 6]
    elif ansatz_type == "flat_no_pool":
        keep_wires = list(range(8))
    else:
        raise ValueError(f"Unknown ansatz_type: {ansatz_type!r}")

    n = len(keep_wires)
    zz_pairs = [(keep_wires[i], keep_wires[(i + 1) % n]) for i in range(n)]

    if encoding == "amplitude":
        qnode = _build_amplitude_qnode(config, keep_wires, zz_pairs)
    elif encoding == "E1":
        qnode = _build_E1_qnode(config, keep_wires, zz_pairs)
    else:
        raise ValueError(f"Unknown encoding: {encoding!r}")

    return qnode, keep_wires, zz_pairs


# ============================================================
# Parameter count helpers
# ============================================================

def _n_theta_q(ansatz_type: str) -> int:
    return {"v2_deep": 32, "qfix_shallow": 24, "flat_no_pool": 16}[ansatz_type]


# ============================================================
# PyTorch Module
# ============================================================

class DynamicQCNN(nn.Module):
    """QCNN whose architecture is fully determined by `config`.

    `config` keys that control structure:
      ENCODING        : "amplitude" | "E1"
      ANSATZ_TYPE     : "v2_deep" | "qfix_shallow" | "flat_no_pool"
      READOUT         : "Z" | "Z_ZZ"
      QCNN_TRAINABLE  : bool
      INJECT_NORM     : bool (amplitude only)
      INIT_NOISE_SCALE: float (>0 for near-zero random init, 0 for zeros)
      E1_INIT_A       : float (initial scale of a_embed for E1)
      LAMBDA_FUSION   : float (fixed omega / fusion angle for E1)
    """

    N_PATCHES = 4

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        encoding = config["ENCODING"]

        self.qnode, self._keep_wires, self._zz_pairs = build_qnode(config)

        # Quantum parameter tensor
        n_theta = _n_theta_q(config["ANSATZ_TYPE"])
        scale   = config["INIT_NOISE_SCALE"]
        init    = scale * torch.randn(n_theta) if scale > 0 else torch.zeros(n_theta)
        self.theta_q = nn.Parameter(init.float())

        # E1-only embedding parameters
        if encoding == "E1":
            init_a = config["E1_INIT_A"]
            self.a_embed = nn.Parameter(torch.full((8,), init_a, dtype=torch.float32))
            self.c_embed = nn.Parameter(torch.zeros(8, dtype=torch.float32))
        else:
            self.a_embed = None
            self.c_embed = None

        # Freeze quantum params if requested
        if not config["QCNN_TRAINABLE"]:
            self.theta_q.requires_grad_(False)
            if self.a_embed is not None:
                self.a_embed.requires_grad_(False)
                self.c_embed.requires_grad_(False)

        # Classical head (size depends on readout + ansatz)
        n_per_patch = len(self._keep_wires)
        if config["READOUT"] == "Z_ZZ":
            n_per_patch += len(self._zz_pairs)
        self.head_in_dim = self.N_PATCHES * n_per_patch
        self.head = nn.Sequential(
            nn.LayerNorm(self.head_in_dim),
            nn.Linear(self.head_in_dim, 10),
        )

    # ----------------------------------------------------------
    def _run_amplitude_patch(self, sample: dict, p: int) -> torch.Tensor:
        amp256     = sample["amp4x256"][:, p, :]    # (B, 256)
        norm_angle = sample["norm_angles"][:, p]    # (B,)
        out = self.qnode(amp256, norm_angle, self.theta_q)
        return torch.stack(out, dim=1).float()       # (B, n_features)

    def _run_E1_patch(self, sample: dict, p: int) -> torch.Tensor:
        quad_means = sample["quad_means"][:, p, :]  # (B, 8)
        gA4        = sample["gA4"]                  # (B, 4)
        gammas     = compute_gamma(gA4)              # (B, 4) — outside QNode
        out = self.qnode(quad_means, gammas, self.a_embed, self.c_embed, self.theta_q)
        return torch.stack(out, dim=1).float()       # (B, n_features)

    # ----------------------------------------------------------
    def quantum_features(self, sample: dict) -> torch.Tensor:
        encoding = self.config["ENCODING"]
        patch_tensors = []
        for p in range(self.N_PATCHES):
            if encoding == "amplitude":
                t = self._run_amplitude_patch(sample, p)
            else:
                t = self._run_E1_patch(sample, p)
            patch_tensors.append(t)
        return torch.cat(patch_tensors, dim=1)       # (B, head_in_dim)

    def forward(self, sample: dict):
        features = self.quantum_features(sample)
        logits   = self.head(features)
        return logits, features
