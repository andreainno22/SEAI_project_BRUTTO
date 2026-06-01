"""
draw_circuits.py — Visualizzazione dei 3 ansatz QCNN (Phase 1, amplitude encoding).

Disegna i circuiti completi (encoding + ansatz + readout) per le 3 varianti:
  1. no_pooling_ansatz   : AmplitudeEmbedding + conv8 → readout su tutti e 8 i wire
  2. pool_8->4_ansatz    : AmplitudeEmbedding + conv8 + pool_8_to_4 → readout su [0,2,4,6]
  3. pool_8->4_conv4_ansatz : stessa + conv4_retained → readout su [0,2,4,6]

Usa le funzioni di circuito definite in suitev1/qcnn_builder.py.
Output salvato in ./circuit_diagrams/

Esegui con:
    conda run -n seai_env python draw_circuits.py
"""

import sys, os, math
import numpy as np
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Su Windows, forza utf-8 per l'ASCII art di PennyLane (─ │ ╰ ecc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Aggiungi la root al path per importare suitev1
sys.path.insert(0, os.path.dirname(__file__))
from suitev1.qcnn_builder import conv8, pool_8_to_4, conv4_retained

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
KEEP_WIRES_POOL = [0, 2, 4, 6]   # wire trattenuti dopo il pooling
KEEP_WIRES_FLAT = list(range(8)) # tutti i wire per no_pooling
ZZ_PAIRS_POOL   = [(0, 2), (2, 4), (4, 6), (6, 0)]
ZZ_PAIRS_FLAT   = [(i, (i + 1) % 8) for i in range(8)]

OUT_DIR = "./circuit_diagrams"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Argomenti dummy (forma corretta per ogni QNode)
# ---------------------------------------------------------------------------
amp256_dummy  = np.ones(256) / math.sqrt(256)   # vettore normalizzato uniforme
norm_dummy    = np.array(math.pi / 2)           # angolo di norma medio
theta16_dummy = np.zeros(16)                    # conv8: 16 parametri
phi8_dummy    = np.zeros(8)                     # pool_8_to_4: 8 parametri
theta8_dummy  = np.zeros(8)                     # conv4_retained: 8 parametri

# ---------------------------------------------------------------------------
# Devices (default.qubit: più robusto per draw_mpl)
# ---------------------------------------------------------------------------
dev8 = qml.device("default.qubit", wires=8)

# ---------------------------------------------------------------------------
# QNode 1 — no_pooling_ansatz
# ---------------------------------------------------------------------------
@qml.qnode(dev8, interface="numpy")
def qnode_no_pool(amp256, norm_angle, theta_conv):
    qml.AmplitudeEmbedding(amp256, wires=range(8), normalize=True)
    qml.RY(norm_angle, wires=0)
    conv8(theta_conv)
    return (
        [qml.expval(qml.PauliZ(i)) for i in KEEP_WIRES_FLAT]
        + [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in ZZ_PAIRS_FLAT]
    )

# ---------------------------------------------------------------------------
# QNode 2 — pool_8->4_ansatz
# ---------------------------------------------------------------------------
@qml.qnode(dev8, interface="numpy")
def qnode_pool_shallow(amp256, norm_angle, theta_conv, phi_pool):
    qml.AmplitudeEmbedding(amp256, wires=range(8), normalize=True)
    qml.RY(norm_angle, wires=0)
    conv8(theta_conv)
    pool_8_to_4(phi_pool)
    return (
        [qml.expval(qml.PauliZ(i)) for i in KEEP_WIRES_POOL]
        + [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in ZZ_PAIRS_POOL]
    )

# ---------------------------------------------------------------------------
# QNode 3 — pool_8->4_conv4_ansatz
# ---------------------------------------------------------------------------
@qml.qnode(dev8, interface="numpy")
def qnode_pool_deep(amp256, norm_angle, theta_conv, phi_pool, theta_conv2):
    qml.AmplitudeEmbedding(amp256, wires=range(8), normalize=True)
    qml.RY(norm_angle, wires=0)
    conv8(theta_conv)
    pool_8_to_4(phi_pool)
    conv4_retained(theta_conv2)
    return (
        [qml.expval(qml.PauliZ(i)) for i in KEEP_WIRES_POOL]
        + [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in ZZ_PAIRS_POOL]
    )

# ---------------------------------------------------------------------------
# Helper: salva PNG + TXT
# ---------------------------------------------------------------------------
def draw_and_save(qnode, args, name, title, figsize=(18, 6)):
    # ASCII art → file UTF-8
    ascii_str = qml.draw(qnode, decimals=2)(*args)
    txt_path = os.path.join(OUT_DIR, f"{name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n{'=' * 60}\n")
        f.write(ascii_str)

    # Grafico matplotlib → PNG
    fig, ax = qml.draw_mpl(qnode, decimals=2, style="pennylane")(*args)
    ax.set_title(title, fontsize=11, pad=10)
    fig.set_size_inches(*figsize)
    img_path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  [{name}]  ->  {name}.png + {name}.txt")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Output directory: {os.path.abspath(OUT_DIR)}\n")

    draw_and_save(
        qnode_no_pool,
        (amp256_dummy, norm_dummy, theta16_dummy),
        name="1_no_pooling_ansatz",
        title="no_pooling_ansatz  —  AmplitudeEmbedding + RY(norm) + conv8  →  Z+ZZ su 8 wire  [16 param q]",
        figsize=(22, 6),
    )

    draw_and_save(
        qnode_pool_shallow,
        (amp256_dummy, norm_dummy, theta16_dummy, phi8_dummy),
        name="2_pool_8to4_ansatz",
        title="pool_8→4_ansatz  —  AmplitudeEmbedding + RY(norm) + conv8 + pool_8_to_4  →  Z+ZZ su [0,2,4,6]  [24 param q]",
        figsize=(22, 6),
    )

    draw_and_save(
        qnode_pool_deep,
        (amp256_dummy, norm_dummy, theta16_dummy, phi8_dummy, theta8_dummy),
        name="3_pool_8to4_conv4_ansatz",
        title="pool_8→4_conv4_ansatz  —  AmplitudeEmbedding + RY(norm) + conv8 + pool_8_to_4 + conv4_retained  →  Z+ZZ su [0,2,4,6]  [32 param q]",
        figsize=(26, 6),
    )

    print(f"\nDone. File generati in: {os.path.abspath(OUT_DIR)}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")
