"""
model.py â€” FashionQCNN parametric on (ansatz_name, encoding_name).

Architecture (matches pulito86 baseline):
  - Encoding on wires 0..7 (+ ancilla wire 8 for E1)
  - Conv layer 1 on wires 0..7  (same ansatz across all pairs)
  - Pool 8 -> 4 on (1,0),(3,2),(5,4),(7,6); retained wires = [0,2,4,6]
  - Conv layer 2 on wires [0,2,4,6]
  - Pool 4 -> 2 on (2,0),(6,4); output wires = [0, 4]
  - custom_ansatz adds its final 2-qubit Cartan classifier before readout
  - Return: qml.probs(wires=[0, 4]) -> (B, 4) class probabilities

For hur9 we follow Hur et al.'s "9b" variant: pooling is trace-out only,
because circuit 9 is already an arbitrary SU(4) gate and the extra controlled
pooling rotations are potentially redundant.

The ansatz, pooling function, and parameter shapes are looked up in ANSATZ_REGISTRY;
the encoding function and qubit count in ENCODING_REGISTRY (config.py).
"""

import math
import torch
import torch.nn as nn
import pennylane as qml

from suitev2.config import (
    BASELINE,
    ANSATZ_REGISTRY,
    ENCODING_REGISTRY,
    populate_registries,
)
from suitev2.ansatz import (
    convolution_layer_on_wires,
    pooling_layer,
)
from suitev2.encodings import compute_gamma


# Pool pair definitions (same for all ansatz)
POOL_PAIRS_LAYER1 = [(1, 0), (3, 2), (5, 4), (7, 6)]   # 8 -> 4 retained [0,2,4,6]
POOL_PAIRS_LAYER2 = [(2, 0), (6, 4)]                    # 4 -> 2 retained [0, 4]

WIRES_8 = [0, 1, 2, 3, 4, 5, 6, 7]
WIRES_4 = [0, 2, 4, 6]
OUTPUT_WIRES = [0, 4]
TRACE_POOL_ANSATZ = {"hur9"}  # Hur et al. circuit 9b: trace-out only, no pool gates.
FINAL_CLASSIFIER_ANSATZ = {"custom_ansatz"}  # Matches ciruito_pazzo.ipynb.


def patch_8x8_to_2x2_means(patch):
    """Convert one 8x8 patch into 16 local 2x2 means."""
    B = patch.shape[0]
    patch = patch.reshape(B, 4, 2, 4, 2)
    means = patch.mean(dim=(2, 4))
    return means.reshape(B, -1)


def images_to_four_patches(x_flat):
    """Split flattened 16x16 images into 4 patches (shape: B, 4, 16), scaled by pi."""
    B = x_flat.shape[0]
    img = x_flat.reshape(B, 16, 16)

    f0 = patch_8x8_to_2x2_means(img[:, 0:8, 0:8])
    f1 = patch_8x8_to_2x2_means(img[:, 0:8, 8:16])
    f2 = patch_8x8_to_2x2_means(img[:, 8:16, 0:8])
    f3 = patch_8x8_to_2x2_means(img[:, 8:16, 8:16])

    patches = torch.stack([f0, f1, f2, f3], dim=1)
    return math.pi * patches


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

        if ansatz_name not in ANSATZ_REGISTRY:
            raise ValueError(f"Unknown ansatz: {ansatz_name!r}")
        if encoding_name not in ENCODING_REGISTRY:
            raise ValueError(f"Unknown encoding: {encoding_name!r}")

        conv_fn, n_conv, pool_fn, n_pool = ANSATZ_REGISTRY[ansatz_name]
        if conv_fn is None:
            raise ValueError(f"ansatz '{ansatz_name}' is a placeholder, cannot instantiate.")

        enc_fn, n_qubits, has_trainable = ENCODING_REGISTRY[encoding_name]
        if enc_fn is None:
            raise ValueError(f"encoding '{encoding_name}' is a placeholder, cannot instantiate.")

        self.ansatz_name = ansatz_name
        self.encoding_name = encoding_name
        self.conv_fn = conv_fn
        self.pool_fn = pool_fn
        self.enc_fn = enc_fn
        self.n_conv = n_conv
        self.n_qubits = n_qubits
        self.use_param_pool = ansatz_name not in TRACE_POOL_ANSATZ
        self.use_final_classifier = ansatz_name in FINAL_CLASSIFIER_ANSATZ
        self.n_pool = n_pool if self.use_param_pool else 0

        # Trainable conv + pool parameters
        self.theta_conv1 = nn.Parameter(init_scale * torch.randn(n_conv))
        self.theta_conv2 = nn.Parameter(init_scale * torch.randn(n_conv))
        if self.use_param_pool:
            self.theta_pool1 = nn.Parameter(init_scale * torch.randn(self.n_pool))
            self.theta_pool2 = nn.Parameter(init_scale * torch.randn(self.n_pool))
        else:
            self.register_buffer("theta_pool1", torch.empty(0))
            self.register_buffer("theta_pool2", torch.empty(0))
        if self.use_final_classifier:
            self.theta_final = nn.Parameter(init_scale * torch.randn(n_conv))
        else:
            self.theta_final = None

        # Trainable embedding parameters
        if encoding_name == "e1":
            self.a_embed = nn.Parameter(torch.full((8,), BASELINE.E1_INIT_A, dtype=torch.float32))
            self.c_embed = nn.Parameter(torch.zeros(8, dtype=torch.float32))
            self.theta_enc = None
        elif encoding_name == "custom":
            self.a_embed = None
            self.c_embed = None
            self.theta_enc = nn.Parameter(init_scale * torch.randn(2, 4, 4))
        else:
            self.a_embed = None
            self.c_embed = None
            self.theta_enc = None

        # Device + QNode
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = self._build_qnode()

    # ----------------------------------------------------------
    def _build_qnode(self):
        encoding_name = self.encoding_name
        conv_fn = self.conv_fn
        enc_fn = self.enc_fn
        pool_fn = self.pool_fn
        use_param_pool = self.use_param_pool
        use_final_classifier = self.use_final_classifier

        def apply_pool(theta_pool, pool_pairs):
            if use_param_pool:
                pooling_layer(pool_fn, theta_pool, pool_pairs)

        def apply_final(theta_final):
            if use_final_classifier:
                conv_fn(theta_final, OUTPUT_WIRES)

        if encoding_name == "e3":
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def qnode_e3(x_flat, norm_angle, theta_c1, theta_p1,
                         theta_c2, theta_p2, theta_final):
                # norm_angle: (B,) tensor in [0, pi] â€” pre-computed L2-norm angle.
                # Passed directly to enc_fn so the QNode tape stays static.
                enc_fn(x_flat, norm_angle=norm_angle, n_qubits=8)

                convolution_layer_on_wires(conv_fn, theta_c1, WIRES_8)
                apply_pool(theta_p1, POOL_PAIRS_LAYER1)

                convolution_layer_on_wires(conv_fn, theta_c2, WIRES_4)
                apply_pool(theta_p2, POOL_PAIRS_LAYER2)
                apply_final(theta_final)

                return qml.probs(wires=OUTPUT_WIRES)
            return qnode_e3

        elif encoding_name == "e1":
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def qnode_e1(quad_means_8, gammas_4, a_embed, c_embed,
                         theta_c1, theta_p1, theta_c2, theta_p2, theta_final):
                # E1 encoding spans wires 0..8 (ancilla on wire 8)
                enc_fn(quad_means_8, gammas_4, a_embed, c_embed, ancilla_wire=8)

                # QCNN proceeds on wires 0..7 only (ancilla is "discarded")
                convolution_layer_on_wires(conv_fn, theta_c1, WIRES_8)
                apply_pool(theta_p1, POOL_PAIRS_LAYER1)

                convolution_layer_on_wires(conv_fn, theta_c2, WIRES_4)
                apply_pool(theta_p2, POOL_PAIRS_LAYER2)
                apply_final(theta_final)

                return qml.probs(wires=OUTPUT_WIRES)
            return qnode_e1

        elif encoding_name == "custom":
            @qml.qnode(self.dev, interface="torch", diff_method="backprop")
            def qnode_custom(patches, theta_enc, theta_c1, theta_p1,
                             theta_c2, theta_p2, theta_final):
                enc_fn(patches, theta_enc)

                convolution_layer_on_wires(conv_fn, theta_c1, WIRES_8)
                apply_pool(theta_p1, POOL_PAIRS_LAYER1)

                convolution_layer_on_wires(conv_fn, theta_c2, WIRES_4)
                apply_pool(theta_p2, POOL_PAIRS_LAYER2)
                apply_final(theta_final)

                return qml.probs(wires=OUTPUT_WIRES)
            return qnode_custom

        else:
            raise ValueError(f"Unknown encoding: {encoding_name!r}")

    # ----------------------------------------------------------
    def forward(self, x_flat, quad_means_8=None, gA4=None):
        """Forward pass returning (B, 4) class probabilities.

        For E3: only x_flat (B, 256) is used.
        For E1: only quad_means_8 (B, 8) and gA4 (B, 4) are used.
        """
        if self.encoding_name == "e3":
            # Compute the L2-norm angle outside the QNode (torch op, not on the tape).
            # norm_angle in [0, pi]: blank image -> 0, fully-saturated image -> ~pi.
            # Dividing by sqrt(n_pixels) normalises across different image sizes.
            norm_angle = math.pi * x_flat.norm(dim=-1) / math.sqrt(float(x_flat.shape[1]))
            probs = self.qnode(
                x_flat, norm_angle,
                self.theta_conv1, self.theta_pool1,
                self.theta_conv2, self.theta_pool2,
                self.theta_final if self.theta_final is not None else x_flat.new_empty(0),
            )
        elif self.encoding_name == "e1":
            gammas = compute_gamma(gA4)   # (B, 4) â€” torch op OUTSIDE the QNode
            probs = self.qnode(
                quad_means_8, gammas,
                self.a_embed, self.c_embed,
                self.theta_conv1, self.theta_pool1,
                self.theta_conv2, self.theta_pool2,
                self.theta_final if self.theta_final is not None else x_flat.new_empty(0),
            )
        elif self.encoding_name == "custom":
            patches = images_to_four_patches(x_flat)
            probs = self.qnode(
                patches, self.theta_enc,
                self.theta_conv1, self.theta_pool1,
                self.theta_conv2, self.theta_pool2,
                self.theta_final if self.theta_final is not None else x_flat.new_empty(0),
            )
        else:
            raise ValueError(f"Unknown encoding: {self.encoding_name!r}")

        return probs.float()

    # ----------------------------------------------------------
    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

