"""
baselines.py - Classical reference models for the v3 comparison.

  - LogisticBaseline   (C1): plain logistic regression on the 256 raw pixels.
                             A trivial classical model; if the quantum arms do
                             not beat it, there is no practical case for the
                             quantum circuit on this task.
  - RandomFeatureBaseline (C2): a FIXED random linear projection of the 256
                             pixels into 8 features, passed through cos(.) (so
                             features land in [-1, 1] like <Z> values), then a
                             trainable head identical to the quantum model's.
                             This is the classical analogue of the "frozen"
                             quantum arm (8 random features + same head), and
                             isolates whether the quantum random feature map is
                             any better than a classical one.

Both return logits (B, 4) and are trained with the SAME optimizer / loss /
schedule as the quantum arms (see train.py).
"""

import math
import torch
import torch.nn as nn

from suitev3 import config as V3


class LogisticBaseline(nn.Module):
    def __init__(self, in_dim: int = V3.PIXEL_DIM, n_classes: int = V3.N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x_flat, *args, **kwargs):
        return self.fc(x_flat.float())

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RandomFeatureBaseline(nn.Module):
    """Fixed random features (Rahimi-Recht style) + trainable head.

    The projection W and phase b are FIXED random buffers (not trained), so the
    only trainable parameters live in the head - exactly mirroring the frozen
    quantum arm, where only the head learns on top of a fixed feature map.
    """

    def __init__(self, in_dim: int = V3.PIXEL_DIM,
                 n_features: int = V3.RANDFEAT_DIM, n_classes: int = V3.N_CLASSES):
        super().__init__()
        # Fixed random projection (seeded globally via set_seed before construction).
        W = torch.randn(in_dim, n_features) / math.sqrt(in_dim)
        b = torch.empty(n_features).uniform_(0.0, 2.0 * math.pi)
        self.register_buffer("W", W)
        self.register_buffer("b", b)

        self.head = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, n_classes),
        )

    def forward(self, x_flat, *args, **kwargs):
        z = torch.cos(x_flat.float() @ self.W + self.b)   # (B, K) in [-1, 1]
        return self.head(z)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLP8Baseline(nn.Module):
    """Classical TRAINABLE feature extractor (256 -> 8) + the SAME head as the QCNN.

    The iso-architecture classical twin of the 'trained' quantum arm: identical
    8-feature bottleneck and identical head; the ONLY difference is that the
    256 -> 8 feature map is a trainable classical Linear + tanh instead of the
    trainable quantum circuit. tanh keeps the features in (-1, 1), the range of
    <Z> expectation values, so the head sees comparable inputs.

    This is THE control that closes the question: trained-quantum vs
    trained-classical at the same bottleneck + same head + same training.

    Parameter note: the classical extractor uses 256*8 + 8 = 2056 params vs the
    quantum circuit's 10. The quantum side is far more parameter-efficient
    (amplitude encoding injects all 256 pixels "for free"). The comparison is
    deliberately iso-ARCHITECTURE (same bottleneck + head), not iso-parameter;
    the asymmetry is reported as-is.
    """

    def __init__(self, in_dim: int = V3.PIXEL_DIM,
                 n_features: int = V3.RANDFEAT_DIM, n_classes: int = V3.N_CLASSES):
        super().__init__()
        self.extractor = nn.Linear(in_dim, n_features)
        self.head = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, n_classes),
        )

    def forward(self, x_flat, *args, **kwargs):
        z = torch.tanh(self.extractor(x_flat.float()))   # (B, 8) in (-1, 1)
        return self.head(z)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------------------------------------------
# Arm -> model factory
# ------------------------------------------------------------------

def build_model(arm: str):
    """Return a fresh model for the given arm name.

    Quantum arms are imported lazily so the classical baselines don't pull in
    PennyLane unless needed.
    """
    if arm == "logistic":
        return LogisticBaseline()
    if arm == "randfeat":
        return RandomFeatureBaseline()
    if arm == "mlp8":
        return MLP8Baseline()
    if arm in ("trained", "frozen"):
        from suitev3.model import HybridFlatQCNN
        return HybridFlatQCNN(freeze_quantum=(arm == "frozen"))
    raise ValueError(f"Unknown arm: {arm!r}")
