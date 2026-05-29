"""
ansatz.py - Convolutional and pooling building blocks for the Hur 2022 QCNN.

Three convolutional ansatz from the paper (Fig. 2):
  - hur_convolution_circuit6 (6 params)  = U_SO4 in the reference repo
  - hur_convolution_circuit8 (10 params) = hur8_two_pool_experiment baseline implementation
  - hur_convolution_circuit9 (15 params) = U_SU4 (arbitrary SU(4)) in the repo
  - custom_ansatz (11 params)            = Cartan-inspired block from
                                           cartan_pooling_experiment.ipynb

The Hur pooling block (hur_pool_pair) is shared across the Hur ansatz,
matching the paper's setup. The custom ansatz uses the controlled-transfer
pooling block from cartan_pooling_experiment.ipynb. Pool pairs: control/discard qubit is
traced out (in the sense of being discarded from subsequent layers), target/
keep qubit is retained.

Reference repository: https://github.com/takh04/QCNN (unitary.py).
"""

import pennylane as qml


# ============================================================
# Convolutional circuits
# ============================================================

def hur_convolution_circuit6(theta, wires):
    """Hur paper Circuit 6 = U_SO4 in the reference repo.

    Structure (6 params):
      RY(t0) -- CNOT -- RY(t2) -- CNOT -- RY(t4)
      RY(t1)            RY(t3)            RY(t5)

    Both CNOTs have wires[0] as control, wires[1] as target.
    Implements an arbitrary SO(4) gate on the two qubits.
    """
    a, b = wires
    qml.RY(theta[0], wires=a)
    qml.RY(theta[1], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RY(theta[2], wires=a)
    qml.RY(theta[3], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RY(theta[4], wires=a)
    qml.RY(theta[5], wires=b)


def hur_convolution_circuit8(theta, wires):
    """Hur paper Circuit 8 - identical to hur8_two_pool_experiment baseline.

    Structure (10 params):
      RX(t0) -- RZ(t2) -- RX(t4) -- CNOT -- RX(t6) -- RZ(t8)
      RX(t1)    RZ(t3)    RX(t5)            RX(t7)    RZ(t9)
    """
    a, b = wires
    qml.RX(theta[0], wires=a)
    qml.RX(theta[1], wires=b)
    qml.RZ(theta[2], wires=a)
    qml.RZ(theta[3], wires=b)
    qml.RX(theta[4], wires=a)
    qml.RX(theta[5], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RX(theta[6], wires=a)
    qml.RX(theta[7], wires=b)
    qml.RZ(theta[8], wires=a)
    qml.RZ(theta[9], wires=b)


def hur_convolution_circuit9(theta, wires):
    """Hur paper Circuit 9 = U_SU4 in the reference repo.

    Structure (15 params) - KAK-style parameterization of arbitrary SU(4):
      U3(t0,t1,t2)  -- CNOT(a,b) -- RY(t6) -- CNOT(b,a) -- RY(t8) -- CNOT(a,b) -- U3(t9,t10,t11)
      U3(t3,t4,t5)               -- RZ(t7) --                                     U3(t12,t13,t14)
    """
    a, b = wires
    qml.U3(theta[0], theta[1], theta[2], wires=a)
    qml.U3(theta[3], theta[4], theta[5], wires=b)
    qml.CNOT(wires=[a, b])
    qml.RY(theta[6], wires=a)
    qml.RZ(theta[7], wires=b)
    qml.CNOT(wires=[b, a])
    qml.RY(theta[8], wires=a)
    qml.CNOT(wires=[a, b])
    qml.U3(theta[9],  theta[10], theta[11], wires=a)
    qml.U3(theta[12], theta[13], theta[14], wires=b)


def custom_ansatz(theta, wires):
    """Cartan-inspired two-qubit ansatz from cartan_pooling_experiment.ipynb.

    Structure (11 params):
      local RZ/RY rotations -> Ising XX/YY/ZZ core -> local RY/RZ rotations

    This is used as a translationally shared QCNN convolution block, exactly
    like the Hur two-qubit ansatz functions above.
    """
    a, b = wires

    qml.RZ(theta[0], wires=a)
    qml.RY(theta[1], wires=a)
    qml.RZ(theta[2], wires=b)
    qml.RY(theta[3], wires=b)

    qml.IsingXX(theta[4], wires=[a, b])
    qml.IsingYY(theta[5], wires=[a, b])
    qml.IsingZZ(theta[6], wires=[a, b])

    qml.RY(theta[7], wires=a)
    qml.RZ(theta[8], wires=a)
    qml.RY(theta[9], wires=b)
    qml.RZ(theta[10], wires=b)


# ============================================================
# Pooling block (shared across ansatz)
# ============================================================

def _controlled_rx_on_zero(angle, control, target):
    """Controlled-RX activated when the control qubit is |0> (open control).

    Hur pooling uses one such open-controlled rotation.
    """
    qml.PauliX(wires=control)
    qml.CRX(angle, wires=[control, target])
    qml.PauliX(wires=control)


def hur_pool_pair(theta_pool, control, target):
    """Hur paper pooling block (Fig. 3): 2 parameters.

      CRZ(theta[0]), activated on control=|1>
      CRX(theta[1]), activated on control=|0> (open control)

    After this gate, the `control` qubit is conceptually discarded.
    """
    qml.CRZ(theta_pool[0], wires=[control, target])
    _controlled_rx_on_zero(theta_pool[1], control, target)


def hur_pool_layer(theta_pool, pool_pairs):
    """Apply hur_pool_pair on a list of (control, target) pairs.

    The same 2 pool parameters are shared across all pairs in the layer.
    """
    for control, target in pool_pairs:
        hur_pool_pair(theta_pool, control, target)


def transfer_pooling_pair(theta_pool, discard, keep):
    """Controlled-transfer pooling block from cartan_pooling_experiment.ipynb.

    Total: 3 parameters. Before `discard` is ignored by the next QCNN layer,
    part of its information is transferred to `keep`.
    """
    qml.CNOT(wires=[discard, keep])
    qml.CRY(theta_pool[0], wires=[discard, keep])
    qml.CRZ(theta_pool[1], wires=[discard, keep])
    qml.RY(theta_pool[2], wires=keep)


def pooling_layer(pool_fn, theta_pool, pool_pairs):
    """Apply a pool function on all pooling pairs with shared parameters."""
    for discard, keep in pool_pairs:
        pool_fn(theta_pool, discard, keep)


# ============================================================
# Convolutional layer driver
# ============================================================

def convolution_layer_on_wires(conv_fn, theta_conv, active_wires, periodic=False):
    """Apply `conv_fn` on even+shifted pairs of `active_wires`.

    For active_wires = [0,1,2,3,4,5,6,7]:
      even pairs   = (0,1), (2,3), (4,5), (6,7)
      shifted pairs= (1,2), (3,4), (5,6)

    The same parameter vector is shared across all pairs (translational
    invariance, as required by the QCNN architecture).
    """
    even_pairs = [
        (active_wires[i], active_wires[i + 1])
        for i in range(0, len(active_wires) - 1, 2)
    ]
    shifted_pairs = [
        (active_wires[i], active_wires[i + 1])
        for i in range(1, len(active_wires) - 1, 2)
    ]
    if periodic and len(active_wires) > 2:
        shifted_pairs.append((active_wires[-1], active_wires[0]))

    for pair in even_pairs:
        conv_fn(theta_conv, pair)
    for pair in shifted_pairs:
        conv_fn(theta_conv, pair)

