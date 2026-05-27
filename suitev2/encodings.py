"""
encodings.py â€” Data encoding strategies for the suite.

  - encoding_e3: amplitude embedding of 256-pixel image on 8 qubits
                 (identical to pulito86 baseline, + optional norm injection).
  - encoding_e1: trainable affine angle embedding on 8 qubits + a global
                 ancilla (wire 8) re-uploading 4 global statistics, fused
                 to the local wires via CNOT-RZ-CNOT.
                 Replicates the E1 of versione_1_test/qcnn_builder.py
                 (wires 0..7 data, wire 8 ancilla; 9 qubits total).
  - encoding_custom: pairwise fragment encoding with trainable weights.

All encoding functions assume the QNode has wires properly allocated
(8 for E3, 9 for E1). Trainable parameters (a_embed, c_embed for E1)
are owned by the model and passed in.
"""

import math
import torch
import pennylane as qml

from suitev2.config import BASELINE


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
# E3 â€” amplitude embedding (256 amplitudes on 8 qubits)
# ============================================================

def encoding_e3(x_flat, norm_angle=None, n_qubits=8):
    """Apply amplitude embedding of a 256-d (or 2^n_qubits) input.

    PennyLane's AmplitudeEmbedding handles normalisation internally.  If
    `norm_angle` (shape (B,), in [0, pi]) is provided, a parameter-free
    RY rotation is injected on wire 0 after the embedding to re-introduce
    the pixel-intensity scale information that AmplitudeEmbedding(normalize=True)
    discards.  The angle is pre-computed outside the QNode as:

        norm_angle = pi * ||x_flat||_2 / sqrt(n_pixels)

    so that a blank image gives 0 and a fully-saturated image gives ~pi.
    Matches pulito86's encoding step (+ norm injection from suitev1).
    """
    qml.AmplitudeEmbedding(features=x_flat, wires=range(n_qubits), normalize=True)
    if norm_angle is not None:
        qml.RY(norm_angle, wires=0)


# ============================================================
# E1 â€” affine angle embedding + global ancilla fusion (9 qubits)
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

    Wire 8 is NOT measured by the QCNN (the conv/pool layers operate on
    wires 0..7 only); it is implicitly traced-out when qml.probs(wires=[0,4])
    is returned.  This partial-trace is intentional: the ancilla's entanglement
    with wires 0..7 (established via the fusion step) provides a soft global
    context that influences the marginal probabilities on the output wires.

    Inputs:
      quad_means_8: (B, 8) tensor â€” 8 quadrant means per image (clean, no aug)
      gammas_4:     (B, 4) tensor â€” pre-computed (use compute_gamma outside)
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

    # 1b. Global ancilla encoding on wire 8 â€” re-upload of gammas
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


# =============================================================================
# Custom â€” Pairwise Fragment Encoding (Re-uploading on patches)
# =============================================================================

def encoding_custom(patches, theta_enc):
    """
    Custom Pairwise Fragment Encoding (data re-uploading with trainable weights).

    Args:
      patches:   (B, 4, 16) tensor â€” four 8x8 patches compressed to 16 local
                 2x2 means each, already scaled by pi (from images_to_four_patches).
      theta_enc: (2, 4, 4) trainable parameter tensor â€” groups 0/1 shared across
                 top/bottom image halves (patches 0+1 share group 0,
                 patches 2+3 share group 1).

    16x16 images are split into 4 patches (8x8).
    Each 8x8 patch is compressed into 16 local 2x2 means (see images_to_four_patches).
    These 16 features are uploaded in 4 sequential steps per qubit-pair using
    trainable interleaved gates â€” this constitutes the data re-uploading.
    """
    N_ENCODING_STEPS = 4

    def encode_patch_on_qubit_pair(patch, theta_enc_group, wires):
        a, b = wires
        for s in range(N_ENCODING_STEPS):
            base = 4 * s
            x0 = patch[:, base + 0]
            x1 = patch[:, base + 1]
            x2 = patch[:, base + 2]
            x3 = patch[:, base + 3]

            qml.RY(x0, wires=a)
            qml.RZ(x1, wires=a)
            qml.RY(x2, wires=b)
            qml.RZ(x3, wires=b)

            qml.CNOT(wires=[a, b])
            qml.RY(theta_enc_group[s, 0], wires=a)
            qml.RY(theta_enc_group[s, 1], wires=b)

            qml.CNOT(wires=[b, a])
            qml.RZ(theta_enc_group[s, 2], wires=a)
            qml.RZ(theta_enc_group[s, 3], wires=b)

    pair_wires = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for patch_idx, wires in enumerate(pair_wires):
        group_idx = 0 if patch_idx in [0, 1] else 1
        encode_patch_on_qubit_pair(
            patches[:, patch_idx, :],
            theta_enc[group_idx],
            wires
        )

