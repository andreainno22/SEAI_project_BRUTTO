"""
encodings.py — Data encoding strategies for the suite.

  - encoding_e3: amplitude embedding of 256-pixel image on 8 qubits
                 (identical to pulito86 baseline).
  - encoding_e1: trainable affine angle embedding on 8 qubits + a global
                 ancilla (wire 8) re-uploading 4 global statistics, fused
                 to the local wires via CNOT-RZ-CNOT.
                 Replicates the E1 of versione_1_test/qcnn_builder.py
                 (wires 0..7 data, wire 8 ancilla; 9 qubits total).
  - encoding_dense: placeholder (NotImplementedError).

All encoding functions assume the QNode has wires properly allocated
(8 for E3, 9 for E1). Trainable parameters (a_embed, c_embed for E1)
are owned by the model and passed in.
"""

import math
import torch
import pennylane as qml

from fashion_suite.config import BASELINE


# ============================================================
# Pre-computation outside the QNode (torch ops can't go on the tape
# for adjoint diff method; default.qubit/backprop is more forgiving
# but we keep the same convention for portability).
# ============================================================

def compute_gamma(gA4_vec: torch.Tensor) -> torch.Tensor:
    """Compress global features gA4 into 4 angles in (-pi, pi).

    gA4_vec: (B, 4) or (4,). Returns same shape.
    """
    beta = gA4_vec.new_tensor(list(BASELINE.BETA_GLOBAL))
    return math.pi * torch.tanh(beta * gA4_vec)


# ============================================================
# E3 — amplitude embedding (256 amplitudes on 8 qubits)
# ============================================================

def encoding_e3(x_flat, n_qubits=8):
    """Apply amplitude embedding of a 256-d (or 2^n_qubits) input.

    PennyLane's AmplitudeEmbedding handles normalization internally.
    Matches pulito86's encoding step.
    """
    qml.AmplitudeEmbedding(features=x_flat, wires=range(n_qubits), normalize=True)


# ============================================================
# E1 — affine angle embedding + global ancilla fusion (9 qubits)
# ============================================================

def encoding_e1(quad_means_8, gammas_4, a_embed, c_embed,
                ancilla_wire=8, lam=None, omega_fixed=None):
    """E1 encoding on 9 qubits (wires 0..7 data, wire 8 ancilla).

    Steps (replicated from versione_1_test/qcnn_builder.py:165-220):
      1a. Local affine RY on each of wires 0..7:
            RY(a_embed[i] * pi * quad_means[i] + c_embed[i])
      1b. Global ancilla (wire 8): re-upload of gammas (=tanh-compressed gA4)
            RY-RZ-RX-RZ - RY(omega_fixed) - RZ-RX-RY-RZ
      1c. Fusion: for each local wire i in 0..7:
            CNOT(8, i) - RZ(lam) - CNOT(8, i)

    Inputs:
      quad_means_8: (B, 8) tensor — 8 quadrant means per image
      gammas_4:     (B, 4) tensor — pre-computed (use compute_gamma outside)
      a_embed:      (8,) trainable affine slope
      c_embed:      (8,) trainable affine bias
      ancilla_wire: int (default 8)
      lam:          fixed fusion angle (defaults to BASELINE.LAMBDA_FUSION)
      omega_fixed:  fixed re-upload separator (defaults to BASELINE.E1_OMEGA_FIXED)
    """
    if lam is None:
        lam = BASELINE.LAMBDA_FUSION
    if omega_fixed is None:
        omega_fixed = BASELINE.E1_OMEGA_FIXED

    # 1a. Local affine angle embedding on wires 0..7
    for i in range(8):
        qml.RY(
            a_embed[i] * (math.pi * quad_means_8[:, i]) + c_embed[i],
            wires=i,
        )

    # 1b. Global ancilla encoding on wire 8 — re-upload of gammas
    qml.RY(gammas_4[:, 0], wires=ancilla_wire)
    qml.RZ(gammas_4[:, 1], wires=ancilla_wire)
    qml.RX(gammas_4[:, 2], wires=ancilla_wire)
    qml.RZ(gammas_4[:, 3], wires=ancilla_wire)
    qml.RY(omega_fixed,    wires=ancilla_wire)
    qml.RZ(gammas_4[:, 0], wires=ancilla_wire)
    qml.RX(gammas_4[:, 1], wires=ancilla_wire)
    qml.RY(gammas_4[:, 2], wires=ancilla_wire)
    qml.RZ(gammas_4[:, 3], wires=ancilla_wire)

    # 1c. Fusion: entangle ancilla with each local wire via CNOT-RZ-CNOT
    for i in range(8):
        qml.CNOT(wires=[ancilla_wire, i])
        qml.RZ(lam, wires=i)
        qml.CNOT(wires=[ancilla_wire, i])


# ============================================================
# Dense — placeholder
# ============================================================

def encoding_dense(*args, **kwargs):
    """Placeholder for the dense encoding (paper Sec. II C 3)."""
    raise NotImplementedError(
        "dense encoding: da implementare in seguito"
    )
