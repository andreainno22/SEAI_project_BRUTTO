"""
config.py - suitev3 configuration.

Almost everything is inherited from suitev2.config.BASELINE (optimizer LRs,
epochs, batch size, split sizes, label smoothing, early stop, scheduler eta_min,
class map). Only the v3-specific knobs live here.
"""

import math

# Inherit ALL training hyperparameters from suitev2 (single source of truth).
from suitev2.config import BASELINE  # noqa: F401  (re-exported for convenience)


# Seeds: paired across arms (same seed -> identical data split).
SUITE_SEEDS = [42, 43, 44]

# The experimental arms (2x2 quantum/classical x trainable/fixed, + logistic).
#   trained  = quantum trainable extractor (256->8) + head
#   frozen   = quantum FIXED random extractor (256->8) + head
#   mlp8     = classical TRAINABLE extractor (256->8) + head   (iso-arch twin of trained)
#   randfeat = classical FIXED random extractor (256->8) + head
#   logistic = linear on raw pixels (no bottleneck) - trivial classical reference
ARMS = ["trained", "frozen", "mlp8", "randfeat", "logistic"]

# ------------------------------------------------------------------
# Quantum backbone (flat_no_pool, faithful to suitev1)
# ------------------------------------------------------------------
N_QUBITS = 8
N_READOUT = 8            # 8 single-qubit <Z> features (no ZZ, by request)
N_CLASSES = BASELINE.N_CLASSES

# hur8 convolution block: 10 parameters, applied ONCE on all 8 wires, no pooling.
CONV_NAME = "hur8"
N_CONV_PARAMS = 10

# Init scales.
TRAINED_INIT_SCALE = 0.01   # near-identity (matches suitev2) - best shot for the trained arm.
# Frozen arm: meaningful random unitary so it is a genuine "random circuit",
# not a near-identity no-op. Angles drawn ~ U(0, 2*pi).
FROZEN_INIT_HIGH = 2.0 * math.pi

# ------------------------------------------------------------------
# Classical baselines
# ------------------------------------------------------------------
PIXEL_DIM = N_QUBITS_SQUARED = BASELINE.IMG_SIZE * BASELINE.IMG_SIZE  # 256
RANDFEAT_DIM = N_READOUT     # 8 - matched to the quantum readout dimension.

# ------------------------------------------------------------------
# Optimizer (single group; suitev2 used 1e-3 / WD=0 for every group).
# v3 has no E1 embedding params, so one AdamW group covers quantum + head.
# ------------------------------------------------------------------
LR = BASELINE.LR_QKERNEL     # 1e-3
WEIGHT_DECAY = BASELINE.WD_QKERNEL  # 0.0
