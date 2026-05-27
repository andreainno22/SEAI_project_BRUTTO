"""
model.py — FashionQCNN parametric on (ansatz_name, encoding_name).

Architecture (matches pulito86 baseline):
  - Encoding on wires 0..7 (+ ancilla wire 8 for E1)
  - Conv layer 1 on wires 0..7  (same ansatz across all pairs)
  - Pool 8 -> 4 on (1,0),(3,2),(5,4),(7,6); retained wires = [0,2,4,6]
  - Conv layer 2 on wires [0,2,4,6]
  - Pool 4 -> 2 on (2,0),(6,4); output wires = [0, 4]
  - Return: qml.probs(wires=[0, 4]) -> (B, 4) class probabilities

The ansatz function and parameter shape are looked up in ANSATZ_REGISTRY;
the encoding function and qubit count in ENCODING_REGISTRY (config.py).
"""

import torch
import torch.nn as nn
import pennylane as qml

from fashion_suite.config import (
    BASELINE,
    ANSATZ_REGISTRY,
    ENCODING_REGISTRY,
    populate_registries,
)
from fashion_suite.ansatz import (
    convolution_layer_on_wires,
    hur_pool_layer,
)
from fashion_suite.encodings import compute_gamma


# Pool pair definitions (same for all ansatz)
POOL_PAIRS_LAYER1 = [(1, 0), (3, 2), (5, 4), (7, 6)]   # 8 -> 4 retained [0,2,4,6]
POOL_PAIRS_LAYER2 = [(2, 0), (6, 4)]                    # 4 -> 2 retained [0, 4]

WIRES_8 = [0, 1, 2, 3, 4, 5, 6, 7]
WIRES_4 = [0, 2, 4, 6]
OUTPUT_WIRES = [0, 4]


class FashionQCNN(nn.Module):
    """QCNN whose conv ansatz and encoding are selected at construction.

    Args:
      ansatz_name: key in ANSATZ_REGISTRY ("hur6", "hur8", "hur9", ...)
      encoding_name: key in ENCODING_REGISTRY ("e1", "e3", ...)
      init_scale: std for near-zero random param init (default 0.01 like pulito86).
    """

    def __init__(self, ansatz_name: str, encoding_name: str, init_scale: float = 0.01):
        super().__init__()
        if not ANSATZ_REGISTRY or not ENCODING_REGISTRY:
            populate_registries()

        conv_fn, n_conv = ANSATZ_REGISTRY[ansatz_name]
        if conv_fn is None:
            raise ValueError(f"ansatz '{ansatz_name}' is a placeholder, cannot instantiate.")

        enc_fn, n_qubits, has_trainable = ENCODING_REGISTRY[encoding_name]
        if enc_fn is None:
            raise ValueError(f"encoding '{encoding_name}' is a placeholder, cannot instantiate.")

        self.ansatz_name = ansatz_name
        self.encoding_name = encoding_name
        self.conv_fn = conv_fn
        self.enc_fn = enc_fn
        self.n_conv = n_conv
        self.n_qubits = n_qubits

        # Trainable conv + pool parameters
        self.theta_conv1 = nn.Parameter(init_scale * torch.randn(n_conv))
        self.theta_pool1 = nn.Parameter(init_scale * torch.randn(2))
        self.theta_conv2 = nn.Parameter(init_scale * torch.randn(n_conv))
        self.theta_pool2 = nn.Parameter(init_scale * torch.randn(2))

        # E1-only trainable embedding parameters
        if encoding_name == "e1":
            self.a_embed = nn.Parameter(torch.full((8,), BASELINE.E1_INIT_A, dtype=torch.float32))
            self.c_embed = nn.Parameter(torch.zeros(8, dtype=torch.float32))
        else:
            self.a_embed = None
            self.c_embed = None

        # Device + QNode
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = self._build_qnode()

    # ----------------------------------------------------------
    def _build_qnode(self):
        encoding_name = self.encoding_name
        conv_fn = self.conv_fn
        enc_fn = self.enc_fn

        if encoding_name == "e3":
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def qnode_e3(x_flat, theta_c1, theta_p1, theta_c2, theta_p2):
                enc_fn(x_flat, n_qubits=8)

                convolution_layer_on_wires(conv_fn, theta_c1, WIRES_8)
                hur_pool_layer(theta_p1, POOL_PAIRS_LAYER1)

                convolution_layer_on_wires(conv_fn, theta_c2, WIRES_4)
                hur_pool_layer(theta_p2, POOL_PAIRS_LAYER2)

                return qml.probs(wires=OUTPUT_WIRES)
            return qnode_e3

        elif encoding_name == "e1":
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def qnode_e1(quad_means_8, gammas_4, a_embed, c_embed,
                         theta_c1, theta_p1, theta_c2, theta_p2):
                # E1 encoding spans wires 0..8 (ancilla on wire 8)
                enc_fn(quad_means_8, gammas_4, a_embed, c_embed, ancilla_wire=8)

                # QCNN proceeds on wires 0..7 only (ancilla is "discarded")
                convolution_layer_on_wires(conv_fn, theta_c1, WIRES_8)
                hur_pool_layer(theta_p1, POOL_PAIRS_LAYER1)

                convolution_layer_on_wires(conv_fn, theta_c2, WIRES_4)
                hur_pool_layer(theta_p2, POOL_PAIRS_LAYER2)

                return qml.probs(wires=OUTPUT_WIRES)
            return qnode_e1

        else:
            raise ValueError(f"Unknown encoding: {encoding_name!r}")

    # ----------------------------------------------------------
    def forward(self, x_flat, quad_means_8=None, gA4=None):
        """Forward pass returning (B, 4) class probabilities.

        For E3: only x_flat (B, 256) is used.
        For E1: only quad_means_8 (B, 8) and gA4 (B, 4) are used.
        """
        if self.encoding_name == "e3":
            probs = self.qnode(
                x_flat,
                self.theta_conv1, self.theta_pool1,
                self.theta_conv2, self.theta_pool2,
            )
        elif self.encoding_name == "e1":
            gammas = compute_gamma(gA4)   # (B, 4) — torch op OUTSIDE the QNode
            probs = self.qnode(
                quad_means_8, gammas,
                self.a_embed, self.c_embed,
                self.theta_conv1, self.theta_pool1,
                self.theta_conv2, self.theta_pool2,
            )
        else:
            raise ValueError(f"Unknown encoding: {self.encoding_name!r}")

        return probs.float()

    # ----------------------------------------------------------
    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
