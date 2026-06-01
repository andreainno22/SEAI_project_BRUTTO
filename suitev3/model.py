"""
model.py - HybridFlatQCNN: amplitude encoding (no norm injection) -> single
flat_no_pool convolution layer on 8 wires -> 8 single-qubit <Z> readout ->
classical head (LayerNorm + Linear).

This is the v1-style hybrid design (quantum feature map + classical head),
ported onto suitev2's 4-class / single-QNode / 16x16 task.

Two modes:
  - freeze_quantum=False (arm "trained"): theta_conv ~ 0.01*N(0,1), trainable.
  - freeze_quantum=True  (arm "frozen"):  theta_conv ~ U(0, 2*pi), FROZEN.
                                          Only the head trains.

The amplitude encoding and ansatz blocks are imported verbatim from suitev2 so
the quantum kernel is identical to the rest of the project; the only encoding
change is norm_angle=None (norm injection removed, by request).
"""

import math
import torch
import torch.nn as nn
import pennylane as qml

from suitev2.ansatz import hur_convolution_circuit8, convolution_layer_on_wires
from suitev3 import config as V3


WIRES_8 = list(range(V3.N_QUBITS))


class HybridFlatQCNN(nn.Module):
    """flat_no_pool QCNN + classical head.

    Args:
      freeze_quantum: if True, the conv params are drawn from U(0, 2*pi) and
                      frozen (requires_grad=False); only the head is trained.
      conv_fn:        two-qubit convolution block (default hur8, 10 params).
      n_conv:         number of conv parameters.
    """

    def __init__(self, freeze_quantum: bool = False,
                 conv_fn=hur_convolution_circuit8, n_conv: int = V3.N_CONV_PARAMS):
        super().__init__()
        self.freeze_quantum = freeze_quantum
        self.conv_fn = conv_fn
        self.n_conv = n_conv

        # --- quantum parameters (single flat conv layer) ---
        if freeze_quantum:
            theta = torch.empty(n_conv).uniform_(0.0, V3.FROZEN_INIT_HIGH)
        else:
            theta = V3.TRAINED_INIT_SCALE * torch.randn(n_conv)
        self.theta_conv = nn.Parameter(theta, requires_grad=not freeze_quantum)

        # --- classical head: LayerNorm(8) + Linear(8, 4) ---
        self.head = nn.Sequential(
            nn.LayerNorm(V3.N_READOUT),
            nn.Linear(V3.N_READOUT, V3.N_CLASSES),
        )

        # --- device + QNode ---
        self.dev = qml.device("default.qubit", wires=V3.N_QUBITS)
        self.qnode = self._build_qnode()

    def _build_qnode(self):
        conv_fn = self.conv_fn

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(x_flat, theta_conv):
            # Amplitude embedding of the 256-pixel image, NO norm injection
            # (inlined here so v3 does not depend on suitev2's encoding signature).
            qml.AmplitudeEmbedding(features=x_flat, wires=range(V3.N_QUBITS), normalize=True)
            # Single flat convolution layer on all 8 wires (no pooling).
            convolution_layer_on_wires(conv_fn, theta_conv, WIRES_8)
            # 8 single-qubit <Z> expectation values.
            return [qml.expval(qml.PauliZ(w)) for w in range(V3.N_QUBITS)]

        return qnode

    def features(self, x_flat):
        """Post-quantum readout features (B, 8), BEFORE the classical head.

        Exposed so the frozen arm can precompute its (fixed) feature map once and
        train only the head on the cache - exact, not an approximation, since a
        frozen circuit produces identical features at every epoch.
        """
        out = self.qnode(x_flat, self.theta_conv)
        # PennyLane returns a tuple of 8 (B,) tensors with input broadcasting.
        if isinstance(out, (list, tuple)):
            feats = torch.stack(out, dim=-1)        # (B, 8)
        else:
            feats = out
            if feats.dim() == 2 and feats.shape[0] == V3.N_READOUT:
                feats = feats.transpose(0, 1)       # (8, B) -> (B, 8) fallback
        return feats.float()

    def forward(self, x_flat, *args, **kwargs):
        """x_flat: (B, 256). Extra args (quad_means_8, gA4) are ignored so the
        same DataLoader as suitev2 can be reused unchanged."""
        return self.head(self.features(x_flat))      # logits (B, 4)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
