"""
config.py â€” Suite-wide configuration: hyperparameters, registries, seeds.

The suite explores a 4-ansatz x 3-encoding matrix on Fashion-MNIST 4-class.
  encodings: e3 (amplitude), e1 (affine angle + global ancilla), custom (pairwise fragment)
Placeholder slots (custom4q) raise NotImplementedError and are skipped
by run_suite.py.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional


# ============================================================
# Baseline hyperparameters (from pulito86.ipynb)
# ============================================================

@dataclass
class BaselineConfig:
    # Data
    IMG_SIZE: int = 16
    N_CLASSES: int = 4
    FASHION_CLASSES: tuple = (0, 1, 7, 8)   # T-shirt/top, Trouser, Sneaker, Bag
    CLASS_MAP: dict = field(default_factory=lambda: {0: 0, 1: 1, 7: 2, 8: 3})

    # Split allineato esattamente al paper Hur 2022 (Sec. IV A):
    #   12000 train / 2000 test per Fashion-MNIST (ratio 6:1).
    # Diviso bilanciato sulle 4 classi:
    #   train: 12000 / 4 = 3000 per classe
    #   test : 2000  / 4 =  500 per classe
    # Val: il paper non specifica un val set; usiamo val=test in size
    # (500/classe), pratica standard per early stopping/best checkpoint.
    TRAIN_PER_CLASS: int = 3000     # 4 * 3000 = 12000 total
    VAL_PER_CLASS:   int = 500       # 4 * 500  =  2000 total
    TEST_PER_CLASS:  int = 500       # 4 * 500  =  2000 total

    # Training
    BATCH_SIZE: int = 24
    EPOCHS: int = 15
    LR: float = 1e-3            # kept for config.json logging / backward compat
    EARLY_STOP_PATIENCE: int = 10
    LABEL_SMOOTHING: float = 0.1

    # Optimizer â€” per-group AdamW learning rates & weight decays
    # LR_QKERNEL applies to theta_conv + theta_pool (quantum circuit params).
    # LR_EMBED   applies to a_embed/c_embed (E1) and theta_enc (custom);
    #            kept deliberately lower so the encoding doesn't blow up early.
    LR_QKERNEL:  float = 1e-3
    LR_EMBED:    float = 1e-4
    WD_QKERNEL:  float = 1e-5
    WD_EMBED:    float = 1e-5

    # E1-specific (replicated from versione_1_test/qcnn_builder.py)
    E1_INIT_A: float = 0.2
    LAMBDA_FUSION: float = math.pi / 4
    E1_OMEGA_FIXED: float = math.pi / 2
    BETA_GLOBAL: tuple = (1.0, 10.0, 10.0, 1.0)


BASELINE = BaselineConfig()


# ============================================================
# Suite seeds
# ============================================================

SUITE_SEEDS = [42, 43, 44]


# ============================================================
# Ansatz registry
#   key   -> (conv_fn, n_conv_params)
#   conv_fn=None means placeholder -> skipped at runtime.
# ============================================================

# Filled in __init__ via late import to avoid circular dep
ANSATZ_REGISTRY: dict = {}


# ============================================================
# Encoding registry
#   key   -> (enc_fn, n_qubits_total, has_trainable_params)
#   enc_fn=None means placeholder -> skipped at runtime.
# ============================================================

ENCODING_REGISTRY: dict = {}


def populate_registries():
    """Populate the registries from the implementation modules.

    Called lazily by run_suite.py to avoid import-time circular deps.
    Placeholders (custom4q) are registered as (None, None, ...)
    to be explicitly skipped by the runner. the combo matrix and log a SKIPPED row.
    """
    from suitev2 import ansatz as _ansatz
    from suitev2 import encodings as _encodings

    ANSATZ_REGISTRY.clear()
    ANSATZ_REGISTRY.update({
        "hur6":     (_ansatz.hur_convolution_circuit6, 6),
        "hur8":     (_ansatz.hur_convolution_circuit8, 10),
        "hur9":     (_ansatz.hur_convolution_circuit9, 15),
        "custom4q": (None, None),   # placeholder
    })

    ENCODING_REGISTRY.clear()
    ENCODING_REGISTRY.update({
        "e1":    (_encodings.encoding_e1, 9, True),    # 9 wires, has trainable params
        "e3":    (_encodings.encoding_e3, 8, False),   # 8 wires, no trainable params
        "custom": (_encodings.encoding_custom, 8, True),
    })


# ============================================================
# Suite combinations
# ============================================================

def all_combos():
    """Return list of all (ansatz_name, encoding_name) combinations.

    The order is fixed for reproducibility: outer loop = ansatz, inner = encoding.
    Placeholder combos are included in this list; run_suite.py is responsible
    for skipping them with a log line.
    """
    if not ANSATZ_REGISTRY or not ENCODING_REGISTRY:
        populate_registries()
    return [(a, e) for a in ANSATZ_REGISTRY for e in ENCODING_REGISTRY]


def combo_is_runnable(ansatz_name: str, encoding_name: str) -> tuple:
    """Return (is_runnable: bool, skip_reason: str or None)."""
    if not ANSATZ_REGISTRY or not ENCODING_REGISTRY:
        populate_registries()
    if ANSATZ_REGISTRY[ansatz_name][0] is None:
        return False, f"ansatz '{ansatz_name}' is a placeholder"
    if ENCODING_REGISTRY[encoding_name][0] is None:
        return False, f"encoding '{encoding_name}' is a placeholder"
    return True, None

