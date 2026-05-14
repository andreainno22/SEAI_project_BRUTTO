"""
draw_circuits.py — Visualizzazione dei circuiti quantistici 8-qubit.

Genera immagini dei circuiti conv8, pool8 e dei QNode completi (E1-E4)
sia in formato grafico (matplotlib) che testuale (ASCII).
Output salvato in ./circuit_diagrams/

Esegui con:
    conda run -n seai_env python draw_circuits.py
"""

import sys
import math
import os
import numpy as np
import torch
import pennylane as qml
import matplotlib
matplotlib.use("Agg")   # backend non-interattivo, funziona senza display
import matplotlib.pyplot as plt

# Su Windows il terminale usa cp1252 che non supporta i caratteri grafici
# dell'ASCII art di PennyLane (─, │, ╰, ecc.). Forziamo utf-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Costanti (specchiano Test_8_qubit_qfix.py)
# ---------------------------------------------------------------------------
N_QUBITS     = 8
N_MEAS_QUBITS = N_QUBITS // 2          # 4
MEAS_WIRES   = list(range(0, N_QUBITS, 2))  # [0, 2, 4, 6]
LAMBDA_FUSION = math.pi / 4
BETA_GLOBAL  = np.array([1.0, 10.0, 10.0, 1.0], dtype=np.float32)

OUT_DIR = "./circuit_diagrams"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Devices dedicati alla visualizzazione (default.qubit è più robusto per draw)
# ---------------------------------------------------------------------------
dev8 = qml.device("default.qubit", wires=N_QUBITS)
dev9 = qml.device("default.qubit", wires=N_QUBITS + 1)  # E1: 8 locali + 1 globale

# ---------------------------------------------------------------------------
# Circuiti kernel
# ---------------------------------------------------------------------------
def conv8(theta, wires):
    q = wires
    for i in range(N_QUBITS):
        qml.RY(theta[i], wires=q[i])
    qml.CNOT(wires=[q[0], q[1]]); qml.RZ(theta[8],  wires=q[1])
    qml.CNOT(wires=[q[2], q[3]]); qml.RZ(theta[9],  wires=q[3])
    qml.CNOT(wires=[q[4], q[5]]); qml.RZ(theta[10], wires=q[5])
    qml.CNOT(wires=[q[6], q[7]]); qml.RZ(theta[11], wires=q[7])
    qml.CNOT(wires=[q[1], q[2]]); qml.RZ(theta[12], wires=q[2])
    qml.CNOT(wires=[q[3], q[4]]); qml.RZ(theta[13], wires=q[4])
    qml.CNOT(wires=[q[5], q[6]]); qml.RZ(theta[14], wires=q[6])
    qml.CNOT(wires=[q[7], q[0]]); qml.RZ(theta[15], wires=q[0])

def pool8(phi, wires):
    q = wires
    n = N_QUBITS
    half = n // 2
    for i in range(half):
        qml.CRZ(phi[i], wires=[q[2*i+1], q[2*i]])
    for i in range(half):
        qml.CRX(phi[half+i], wires=[q[(2*i+2) % n], q[(2*i+1) % n]])

# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------
def embed_E2_local_8(quad_means_8):
    for i in range(N_QUBITS):
        qml.RY(math.pi * float(quad_means_8[i]), wires=i)

def embed_E4_local_8(quad_means_8, a8, c8):
    for i in range(N_QUBITS):
        qml.RY(float(a8[i]) * (math.pi * float(quad_means_8[i])) + float(c8[i]), wires=i)

def inject_global_on_wire_8(gA4_vec):
    beta = torch.tensor(BETA_GLOBAL)
    gammas = math.pi * torch.tanh(beta * torch.tensor(gA4_vec, dtype=torch.float32))
    g = [float(gammas[i]) for i in range(4)]
    qml.RY(g[0], wires=8); qml.RZ(g[1], wires=8)
    qml.RX(g[2], wires=8); qml.RZ(g[3], wires=8)
    # re-upload (omega fisso = pi/2)
    qml.RY(math.pi/2, wires=8)
    qml.RZ(g[0], wires=8); qml.RX(g[1], wires=8)
    qml.RY(g[2], wires=8); qml.RZ(g[3], wires=8)

def fuse_global_to_locals_8(lam=LAMBDA_FUSION):
    for i in range(N_QUBITS):
        qml.CNOT(wires=[N_QUBITS, i])
        qml.RZ(lam, wires=i)
        qml.CNOT(wires=[N_QUBITS, i])

# ---------------------------------------------------------------------------
# QNodes per visualizzazione (interface numpy, diff_method=best)
# ---------------------------------------------------------------------------
@qml.qnode(dev8, interface="numpy")
def qnode_conv8_only(theta):
    """Solo conv8 — per vedere il kernel convoluzionale isolato."""
    conv8(theta, wires=list(range(N_QUBITS)))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

@qml.qnode(dev8, interface="numpy")
def qnode_pool8_only(phi):
    """Solo pool8 — per vedere il kernel di pooling isolato."""
    pool8(phi, wires=list(range(N_QUBITS)))
    return [qml.expval(qml.PauliZ(i)) for i in MEAS_WIRES]

@qml.qnode(dev8, interface="numpy")
def qnode_E2_draw(quad_means_8, theta_conv, phi_pool):
    """E2: Angle encoding (RY fisso) + conv8 + pool8."""
    embed_E2_local_8(quad_means_8)
    conv8(theta_conv, wires=list(range(N_QUBITS)))
    pool8(phi_pool, wires=list(range(N_QUBITS)))
    return [qml.expval(qml.PauliZ(i)) for i in MEAS_WIRES]

@qml.qnode(dev8, interface="numpy")
def qnode_E3_draw(quad_amp_64, theta_conv, phi_pool):
    """E3: AmplitudeEmbedding (64→256 con padding) + conv8 + pool8."""
    qml.AmplitudeEmbedding(quad_amp_64, wires=range(N_QUBITS), pad_with=0.0, normalize=True)
    conv8(theta_conv, wires=list(range(N_QUBITS)))
    pool8(phi_pool, wires=list(range(N_QUBITS)))
    return [qml.expval(qml.PauliZ(i)) for i in MEAS_WIRES]

@qml.qnode(dev8, interface="numpy")
def qnode_E4_draw(quad_means_8, a8, c8, theta_conv, phi_pool):
    """E4: Angle encoding trainable (a·π·x + c) + conv8 + pool8."""
    embed_E4_local_8(quad_means_8, a8, c8)
    conv8(theta_conv, wires=list(range(N_QUBITS)))
    pool8(phi_pool, wires=list(range(N_QUBITS)))
    return [qml.expval(qml.PauliZ(i)) for i in MEAS_WIRES]

@qml.qnode(dev9, interface="numpy")
def qnode_E1_draw(quad_means_8, gA4_vec, a8, c8, theta_conv, phi_pool):
    """E1: Trainable angle + global ancilla (wire 8) + CNOT fusion + conv8 + pool8."""
    embed_E4_local_8(quad_means_8, a8, c8)
    inject_global_on_wire_8(gA4_vec)
    fuse_global_to_locals_8(lam=LAMBDA_FUSION)
    conv8(theta_conv, wires=list(range(N_QUBITS)))
    pool8(phi_pool, wires=list(range(N_QUBITS)))
    return [qml.expval(qml.PauliZ(i)) for i in MEAS_WIRES]

# ---------------------------------------------------------------------------
# Argomenti dummy (valori neutri ma non tutti zero per evitare gate banali)
# ---------------------------------------------------------------------------
theta_dummy  = np.zeros(16)
phi_dummy    = np.zeros(8)
means_dummy  = np.full(8, 0.5)
a_dummy      = np.ones(8)
c_dummy      = np.zeros(8)
gA4_dummy    = np.array([0.3, 0.1, 0.2, 0.4])
amp_dummy    = np.ones(64) / math.sqrt(64)   # vettore normalizzato uniforme

# ---------------------------------------------------------------------------
# Helper: salva figura e stampa ASCII
# ---------------------------------------------------------------------------
def draw_and_save(qnode, args, name, title, figsize=(14, 5)):
    # ASCII art → file (non a stdout: Windows cp1252 non supporta i caratteri grafici)
    ascii_str = qml.draw(qnode, decimals=2)(*args)
    txt_path = os.path.join(OUT_DIR, f"{name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n{'='*60}\n")
        f.write(ascii_str)

    # Grafico matplotlib → PNG
    fig, ax = qml.draw_mpl(qnode, decimals=2, style="pennylane")(*args)
    ax.set_title(title, fontsize=11, pad=10)
    fig.set_size_inches(*figsize)
    img_path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Solo ASCII safe su stdout
    print(f"  [{name}] OK  ->  {name}.png + {name}.txt")

# ---------------------------------------------------------------------------
# Disegna tutti i circuiti
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Output directory: {os.path.abspath(OUT_DIR)}")

    draw_and_save(
        qnode_conv8_only,
        (theta_dummy,),
        name="1_conv8",
        title="conv8 — kernel convoluzionale (8 RY + 2 layer CNOT-RZ)",
        figsize=(16, 5),
    )

    draw_and_save(
        qnode_pool8_only,
        (phi_dummy,),
        name="2_pool8",
        title="pool8 — kernel di pooling (4 CRZ + 4 CRX)",
        figsize=(12, 5),
    )

    draw_and_save(
        qnode_E2_draw,
        (means_dummy, theta_dummy, phi_dummy),
        name="3_E2_full",
        title="E2 completo — RY(π·x) + conv8 + pool8",
        figsize=(18, 5),
    )

    draw_and_save(
        qnode_E3_draw,
        (amp_dummy, theta_dummy, phi_dummy),
        name="4_E3_full",
        title="E3 completo — AmplitudeEmbedding(64→256) + conv8 + pool8",
        figsize=(18, 5),
    )

    draw_and_save(
        qnode_E4_draw,
        (means_dummy, a_dummy, c_dummy, theta_dummy, phi_dummy),
        name="5_E4_full",
        title="E4 completo — RY(a·π·x + c) trainable + conv8 + pool8",
        figsize=(18, 5),
    )

    draw_and_save(
        qnode_E1_draw,
        (means_dummy, gA4_dummy, a_dummy, c_dummy, theta_dummy, phi_dummy),
        name="6_E1_full",
        title="E1 completo — trainable affine + global ancilla (wire 8) + fusion + conv8 + pool8",
        figsize=(22, 6),
    )

    print(f"\nDone. Tutti i circuiti salvati in: {os.path.abspath(OUT_DIR)}/")
    print("File generati:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")
