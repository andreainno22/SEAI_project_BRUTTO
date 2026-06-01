"""
cost.py - Resource accounting that counts the ENCODING as part of the model cost.

Rationale (the kernel-methods view: Schuld 2021 "Supervised quantum ML models are
kernel methods"; Havlicek et al. Nature 2019): the data-encoding feature map IS the
model, so its circuit must be counted. For amplitude embedding of 256 amplitudes on
8 qubits the state-preparation circuit dominates everything; the trainable conv adds
only ~7 CNOTs, so the apparent "10 trainable params" advantage disappears once the
encoding is counted.

Real numbers are obtained by decomposing the circuit into a hardware gate set
{RX, RY, RZ, CNOT} with PennyLane's `decompose` transform (qml.specs does NOT
decompose AmplitudeEmbedding, which default.qubit accepts natively).

Pure circuit/graph introspection - does NOT run the simulation/training.

Usage:  python -m suitev3.cost
"""

import math
from collections import Counter

import torch
import pennylane as qml
from pennylane.transforms import decompose

from suitev2.ansatz import hur_convolution_circuit8, convolution_layer_on_wires
from suitev3 import config as V3
from suitev3.model import HybridFlatQCNN
from suitev3.baselines import MLP8Baseline, LogisticBaseline, RandomFeatureBaseline


N = V3.N_QUBITS  # 8
GATE_SET = {qml.RX, qml.RY, qml.RZ, qml.CNOT}
TWO_QUBIT = {"CNOT", "CZ", "CY", "Toffoli"}
ROTATIONS = {"RX", "RY", "RZ"}


def _count(ops):
    """Decompose a list of ops to {RX,RY,RZ,CNOT} and tally gates."""
    tape = qml.tape.QuantumScript(ops)
    (dtape,), _ = decompose(tape, gate_set=GATE_SET)
    cnt = Counter(op.name for op in dtape.operations)
    total = sum(cnt.values())
    two_q = sum(c for g, c in cnt.items() if g in TWO_QUBIT)
    rot = sum(c for g, c in cnt.items() if g in ROTATIONS)
    return total, two_q, rot, dict(cnt)


def quantum_cost(t_eps=(1e-3, 1e-6)):
    x = torch.rand(2 ** N)
    x = x / x.norm()
    theta = torch.rand(V3.N_CONV_PARAMS)

    # encoding only
    enc = _count([qml.AmplitudeEmbedding(x, wires=range(N), normalize=True)])

    # encoding + one flat_no_pool conv layer (hur8)
    with qml.tape.QuantumTape() as t:
        qml.AmplitudeEmbedding(x, wires=range(N), normalize=True)
        convolution_layer_on_wires(hur_convolution_circuit8, theta, list(range(N)))
    full = _count(list(t.operations))

    # Ross-Selinger T-count: ~3*log2(1/eps) T gates per arbitrary single-qubit
    # rotation (Ross & Selinger 2016). Clifford gates are ~free in the FT cost
    # model, so the (non-Clifford) rotations dominate the T-count.
    full_rot = full[2]
    tcount = {eps: int(full_rot * 3.0 * math.log2(1.0 / eps)) for eps in t_eps}

    n_trainable = HybridFlatQCNN(freeze_quantum=False).n_trainable_params()
    return {"enc": enc, "full": full, "tcount": tcount, "n_trainable": n_trainable}


def _linear_flops(in_dim, out_dim):
    return 2 * in_dim * out_dim  # MAC = 2 FLOPs


def classical_costs():
    out = {}
    lg = LogisticBaseline()
    out["logistic"] = {"params": lg.n_trainable_params(),
                       "flops": _linear_flops(V3.PIXEL_DIM, V3.N_CLASSES)}
    m = MLP8Baseline()
    out["mlp8"] = {"params": m.n_trainable_params(),
                   "flops": (_linear_flops(V3.PIXEL_DIM, V3.RANDFEAT_DIM)
                             + 11 * V3.RANDFEAT_DIM                 # tanh + layernorm
                             + _linear_flops(V3.RANDFEAT_DIM, V3.N_CLASSES))}
    rf = RandomFeatureBaseline()
    out["randfeat"] = {"params": rf.n_trainable_params(),
                       "flops": (_linear_flops(V3.PIXEL_DIM, V3.RANDFEAT_DIM)
                                 + 8 * V3.RANDFEAT_DIM
                                 + _linear_flops(V3.RANDFEAT_DIM, V3.N_CLASSES))}
    return out


def main():
    print("=" * 74)
    print("suitev3 cost accounting - ENCODING COUNTED AS PART OF THE MODEL")
    print("  kernel-methods view: the encoding feature map IS the model")
    print("=" * 74)

    q = quantum_cost()
    e_tot, e_2q, e_rot, e_types = q["enc"]
    f_tot, f_2q, f_rot, f_types = q["full"]
    a_2q, a_rot = f_2q - e_2q, f_rot - e_rot

    print("\nQUANTUM trained model  (8 qubits; amplitude encoding + 1 flat conv hur8)")
    print(f"  trainable params                : {q['n_trainable']}   (conv {V3.N_CONV_PARAMS} + head 52)")
    print(f"  total gates (decomposed)        : {f_tot}")
    print(f"  two-qubit (CNOT) gates          : {f_2q}")
    print(f"  single-qubit rotations          : {f_rot}")
    print(f"     - ENCODING (state prep)      : {e_2q} two-qubit, {e_rot} rotations  ({e_tot} gates)")
    print(f"     - ANSATZ (trainable conv)    : {a_2q} two-qubit, {a_rot} rotations")
    print(f"  encoding share of two-qubit gates: {e_2q / f_2q * 100:.1f}%")
    print(f"  Clifford+T T-count (Ross-Selinger ~3*log2(1/eps)/rotation):")
    for eps, tc in q["tcount"].items():
        print(f"     eps={eps:g}  ->  ~{tc:,} T gates")
    print(f"  full gate_types: {f_types}")

    print("\nCLASSICAL baselines  (forward-pass FLOPs; MAC = 2 FLOPs)")
    for name, c in classical_costs().items():
        print(f"  {name:9s} | trainable params {c['params']:5d} | ~{c['flops']:,} FLOPs / inference")

    print("\n" + "=" * 74)
    print("Takeaway: counting the encoding, the quantum model needs ~%d two-qubit"
          % f_2q)
    print("gates and ~10^4 T gates per inference, vs ~10^3 FLOPs for the classical")
    print("twin. The '10 trainable params' figure is not a fair cost proxy: ~%.0f%%"
          % (e_2q / f_2q * 100))
    print("of the two-qubit cost is the fixed amplitude-encoding state preparation.")
    print("=" * 74)


if __name__ == "__main__":
    main()
