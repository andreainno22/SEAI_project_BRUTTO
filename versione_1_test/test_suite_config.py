"""
test_suite_config.py — Configuration for the QCNN ablation test suite.

BASELINE_CONFIG defines the "best known" configuration (the SOTA baseline
derived from v2.2 — see Latest/CHANGES_v2.md — with amplitude encoding,
v2_deep ansatz, Z+ZZ readout, trainable quantum kernel). Each entry in
ABLATION_TESTS is a dict of key overrides applied on top of the baseline —
only the keys that differ from the baseline need to be specified.

To add a new test, add an entry to ABLATION_TESTS:
    "my_new_test": {"KEY_TO_OVERRIDE": new_value, ...}

To run a specific test from the command line:
    python ablation_suite.py my_new_test
"""

import math

# ---------------------------------------------------------------------------
# Baseline configuration — mirrors v2.2 (amplitude, v2_deep, Z+ZZ, trainable)
# ---------------------------------------------------------------------------
BASELINE_CONFIG = {
    # --- Data ---
    "TRAIN_SAMPLES":        10000,   # Reduced for fast ablation sweeps; increase for full runs
    "VAL_SAMPLES":           2500,
    "TEST_SAMPLES":          2500,
    "BATCH_SIZE":            128,

    # --- Training ---
    "EPOCHS":                 120,   # Reduced for ablation; use 120 for production runs
    "EARLY_STOP_PATIENCE":    10,
    "PRINT_EVERY":             5,

    # --- Per-group learning rates ---
    "LR_HEAD":             5e-4,
    "LR_QKERNEL":          1e-3,
    "LR_EMBED":            1e-4,   # E1 embedding params (a, c) only

    # --- Regularisation ---
    "WEIGHT_DECAY_HEAD":   1e-4,
    "WEIGHT_DECAY_QKERNEL":1e-5,
    "WEIGHT_DECAY_EMBED":  1e-5,
    "CLIP_NORM":           1.0,
    "LABEL_SMOOTHING":     0.1,    # Prevents overconfident logits (T9)

    # --- Quantum architecture ---
    "ENCODING":      "amplitude",  # "amplitude" | "E1"
    "INJECT_NORM":   True,         # Amplitude only: inject patch L2 norm via RY on wire 0 (C9)
    "ANSATZ_TYPE":   "v2_deep",    # "v2_deep" | "qfix_shallow" | "flat_no_pool"
    "READOUT":       "Z_ZZ",       # "Z" | "Z_ZZ"
    "QCNN_TRAINABLE": True,        # True = end-to-end, False = frozen quantum params

    # --- Initialisation ---
    "INIT_NOISE_SCALE": 1e-3,      # Near-zero random init to break circuit symmetry (B3)

    # --- E1-specific ---
    "E1_INIT_A":    0.2,           # Initial scale of trainable affine amplitude a_embed
    "LAMBDA_FUSION": math.pi / 4,  # Fixed angle for global-to-local fusion gate
    "E1_OMEGA_FIXED": math.pi / 2, # Fixed angle for global reupload separator (omega)
}

# ---------------------------------------------------------------------------
# Ablation tests — keys override BASELINE_CONFIG
# ---------------------------------------------------------------------------
ABLATION_TESTS = {

    # 1. Encoding axis -------------------------------------------------------
    "baseline_amplitude": {},
    # Baseline (Amplitude E3 + norm injection, v2_deep, Z+ZZ, trainable).
    # No overrides — serves as the reference point for all comparisons.

    "test_E1_custom": {
        "ENCODING":    "E1",
        "INJECT_NORM": False,   # norm injection is an amplitude-only concept
    },
    # E1: Custom global-local encoding. It encodes 8 local features (quadrant means)
    # on wires 0-7 using a trainable affine mapping, plus 4 global image statistics
    # on a 9th ancilla wire. The global ancilla is fused into the local wires via
    # CNOT-RZ-CNOT gates before the ansatz.

    # 2. Readout axis --------------------------------------------------------
    "test_readout_Z_only": {
        "READOUT": "Z",
    },
    # Measures only Pauli-Z expectation values on the retained wires.
    # Compared against the baseline (Z + ZZ) to quantify the contribution
    # of two-qubit ZZ correlations to classification accuracy.

    # 3. Trainability axis ---------------------------------------------------
    "test_frozen_qcnn": {
        "QCNN_TRAINABLE": False,
    },
    # Freezes all quantum parameters at initialisation; only the classical
    # head (LayerNorm + Linear) is trained. Measures how much of the accuracy
    # comes from random quantum feature maps vs. learned quantum kernels.

    # 4. Ansatz axis ---------------------------------------------------------
    "test_ansatz_qfix_shallow": {
        "ANSATZ_TYPE": "qfix_shallow",
    },
    # conv8 -> pool(8->4) -> readout (no second conv layer).
    # The "qfix" strategy: one convolutional layer, one pooling step, then
    # direct measurement. Tests whether conv4_retained adds real value.

    "test_ansatz_flat_no_pool": {
        "ANSATZ_TYPE": "flat_no_pool",
    },
    # conv8 -> readout on all 8 wires (no pooling).
    # Removes the hierarchical bottleneck entirely. With Z+ZZ readout this
    # yields 8Z + 8ZZ = 16 features/patch × 4 = 64 total head input features.
    # Tests the hypothesis that pooling distils rather than discards information.
}


def get_config(test_name: str) -> dict:
    """Return the full config for `test_name` (baseline + overrides).

    Raises ValueError if `test_name` is not in ABLATION_TESTS.

    Post-merge normalisation: INJECT_NORM is amplitude-only — for any non-amplitude
    encoding the flag is forced to False so prints/logs reflect what actually runs.
    """
    if test_name not in ABLATION_TESTS:
        raise ValueError(
            f"Unknown test {test_name!r}. "
            f"Available: {list(ABLATION_TESTS.keys())}"
        )
    cfg = BASELINE_CONFIG.copy()
    cfg.update(ABLATION_TESTS[test_name])

    if cfg.get("ENCODING") != "amplitude":
        cfg["INJECT_NORM"] = False

    return cfg
