# Guida alle modifiche del codice QNN di Vittori — Scaling a N Qubit

> **Scopo del documento:** analisi tecnica dettagliata di ogni modifica necessaria per portare il circuito QCNN di Lorenzo Vittori da 4 qubit fissi a N qubit configurabili, con supporto per immagini più grandi e nuovi dataset.

---

## Indice

1. [Panoramica dell'architettura originale](#1-panoramica-dellarchitettura-originale)
2. [Filosofia delle modifiche](#2-filosofia-delle-modifiche)
3. [Blocco 1 — CONFIG](#3-blocco-1--config)
4. [Blocco 2 — Feature Extraction](#4-blocco-2--feature-extraction)
5. [Blocco 3 — Circuito quantistico: `convN` e `poolN`](#5-blocco-3--circuito-quantistico-convn-e-pooln)
6. [Blocco 4 — Embeddings (E2, E3, E4, E1)](#6-blocco-4--embeddings-e2-e3-e4-e1)
7. [Blocco 5 — Devices e QNodes](#7-blocco-5--devices-e-qnodes)
8. [Blocco 6 — `QuanvEmbedModel`](#8-blocco-6--quanvembedmodel)
9. [Blocco 7 — Training loop e diagnostica](#9-blocco-7--training-loop-e-diagnostica)
10. [Gestione speciale di E3 (Amplitude Encoding)](#10-gestione-speciale-di-e3-amplitude-encoding)
11. [Nuovo dataset — Fashion-MNIST](#11-nuovo-dataset--fashion-mnist)
12. [Checklist delle modifiche](#12-checklist-delle-modifiche)
13. [Tabella riepilogativa dei parametri](#13-tabella-riepilogativa-dei-parametri)
14. [Rischi e mitigazioni](#14-rischi-e-mitigazioni)

---

## 1. Panoramica dell'architettura originale

Il codice di Vittori implementa una **Quantum Convolutional Neural Network (QCNN)** a architettura ibrida quantistica-classica. Il flusso completo è:

```
Immagine MNIST 8×8 (1 canale)
    │
    ├─ Divisione in 4 quadranti 4×4
    │
    ├─ Per ogni quadrante:
    │       │
    │       ├─ Feature extraction → 4 scalari (patch-means o pixel raw)
    │       │
    │       └─ Circuito quantistico a 4 qubit:
    │               ├─ Encoding (E1/E2/E3/E4)
    │               ├─ conv4 (RY + CNOT ring + RZ)
    │               ├─ pool4 (CRZ + CRX)
    │               └─ 4× ⟨Z_i⟩  →  4 scalari di output
    │
    ├─ Concatenazione: 4 quadranti × 4 valori = vettore da 16 dim
    │
    └─ Testa classica: LayerNorm → Linear(16,32) → GELU → Dropout → Linear(32, N_CLASSES)
```

### Parametri del circuito originale (4 qubit)

| Parametro | Dimensione | Descrizione |
|---|---|---|
| `theta_conv` | 8 | RY (×4) + RZ ring (×4) |
| `phi_pool` | 4 | CRZ (×2) + CRX (×2) |
| `e4_a16`, `e4_c16` | 16 ciascuno | Learned angle encoding E4 |
| `e1_a16`, `e1_c16` | 16 ciascuno | Learned angle encoding E1 |

---

## 2. Filosofia delle modifiche

L'obiettivo è rendere `N_QUBITS` un **parametro di configurazione** a singolo punto di modifica: cambiare `CONFIG["N_QUBITS"]` deve propagare automaticamente a tutto il codice senza ulteriori interventi.

Le modifiche seguono la catena di dipendenze:

```
CONFIG["N_QUBITS"]
    │
    ├──► transform (resize dell'immagine)
    ├──► extract_quad_features (numero di feature per quadrante)
    ├──► LOCAL_WIRES / GLOBAL_WIRE (liste di wire)
    ├──► convN / poolN (circuito, numero parametri)
    ├──► embeddings E2/E3/E4/E1 (gate RY su N qubit)
    ├──► QNodes (device, wire, chiamate a convN/poolN)
    └──► QuanvEmbedModel (dimensioni parametri, head MLP)
```

**Regola d'oro:** nessuna costante `4` hardcoded nel codice dopo le modifiche. Ogni riferimento alla cardinalità dei qubit deve derivare da `N_QUBITS`.

---

## 3. Blocco 1 — CONFIG

### Dove si trova nel file

Sezione `# CONFIG (v2-uniform)` all'inizio del notebook, circa riga 20.

### Cosa aggiungere

```python
CONFIG = {
    # --- tutto il contenuto originale rimane invariato ---
    "BASE_DIR":       "./qnn_Nqubit_results",
    "VARIANTS":       ["E2", "E3", "E4"],   # E1 opzionale, vedi §6.4
    "SEEDS":          [0, 1, 2],
    "TRAIN_SAMPLES":  1000,
    "VAL_SAMPLES":    500,
    "TEST_SAMPLES":   500,
    "EPOCHS":         15,
    "LR":             1e-2,
    "WEIGHT_DECAY":   0.0,
    "CLIP_NORM":      0.5,
    "ACC_STEPS":      5,
    "PRINT_EVERY":    3,
    "LR_HEAD":        1e-2,
    "LR_QKERNEL":     1e-3,
    "LR_EMBED":       5e-4,
    "E1_INIT_A":      0.2,
    "FAIR_DIM_MATCH": True,
    "SAVE_FIGS":      True,
    "DO_DRAW":        False,
    "DO_CONFUSION_SEED0": True,

    # --- NUOVE CHIAVI ---
    "N_QUBITS":   8,    # qubit per circuito (era implicitamente 4)
    "IMAGE_SIZE": 8,    # lato dell'immagine in pixel (era 8)
                        # per N_QUBITS=8 con E3 usare IMAGE_SIZE=32
}
```

### Costanti derivate (aggiungere subito dopo CONFIG)

```python
N_QUBITS        = CONFIG["N_QUBITS"]
IMAGE_SIZE      = CONFIG["IMAGE_SIZE"]
QUAD_SIZE       = IMAGE_SIZE // 2          # lato quadrante in pixel
PIXELS_PER_QUAD = QUAD_SIZE * QUAD_SIZE    # pixel per quadrante
AMP_SIZE        = 2 ** N_QUBITS           # ampiezze richieste da E3

# Sanity checks
assert N_QUBITS >= 4,    "N_QUBITS minimo 4 (sotto non ha senso pratico)"
assert N_QUBITS % 2 == 0, "N_QUBITS deve essere pari (pooling simmetrico)"
assert IMAGE_SIZE % 2 == 0, "IMAGE_SIZE deve essere pari"
assert PIXELS_PER_QUAD % N_QUBITS == 0, (
    f"PIXELS_PER_QUAD ({PIXELS_PER_QUAD}) non divisibile per N_QUBITS ({N_QUBITS}). "
    f"Aumentare IMAGE_SIZE o ridurre N_QUBITS."
)

# Warning E3
if PIXELS_PER_QUAD < AMP_SIZE:
    print(f"[WARN] E3 richiede {AMP_SIZE} ampiezze ma il quadrante ha solo "
          f"{PIXELS_PER_QUAD} pixel. E3 sarà disabilitata automaticamente.")
    CONFIG["VARIANTS"] = [v for v in CONFIG["VARIANTS"] if v != "E3"]
```

### Tabella di compatibilità IMAGE_SIZE / N_QUBITS

| N_QUBITS | IMAGE_SIZE minimo (E2/E4) | IMAGE_SIZE per E3 | Pixel/quadrante | Ampiezze E3 |
|---|---|---|---|---|
| 4 | 8 | 8 | 16 | 16 ✓ |
| 8 | 8 | 32 | 16 (no E3) / 256 (sì E3) | 256 |
| 16 | 16 | — (non pratico) | 64 | 65.536 ✗ |

---

## 4. Blocco 2 — Feature Extraction

### Dove si trova nel file

Sezione `# Feature extraction (cached)`, funzioni `extract_patches_2x2`, `quadrants_4x4_flat`, `QUAD_PATCH_IDXS`, `get_features`.

### Cosa cambia

`extract_patches_2x2` è hardcoded per 4 feature (patch 2×2). Con N qubit servono N feature per quadrante. La nuova funzione divide ogni quadrante in N gruppi di pixel contigui e ne calcola la media (stesso principio, generalizzato).

`QUAD_PATCH_IDXS` (la lista degli indici dei patch per quadrante) non è più necessaria: il nuovo codice lavora direttamente sugli array numpy.

### Codice da sostituire

**Eliminare:** `extract_patches_2x2`, `quadrants_4x4_flat`, `QUAD_PATCH_IDXS`

**Aggiungere:**

```python
def extract_quad_features(X: np.ndarray, n_qubits: int):
    """
    Estrae N feature per quadrante da un'immagine X di forma (IMAGE_SIZE, IMAGE_SIZE).

    Strategia:
      - Divide ogni quadrante in n_qubits gruppi contigui di pixel
      - Calcola la media di ciascun gruppo → n_qubits scalari per quadrante
      - Restituisce anche i pixel grezzi del quadrante (per E3)

    Args:
        X:        np.ndarray (H, W) float32, valori in [0,1]
        n_qubits: numero di qubit = numero di feature per quadrante

    Returns:
        quad_feats: (4, n_qubits) float32  — feature medie (per E2/E4/E1)
        quad_full:  (4, 2^n_qubits) float32 — pixel per amplitude encoding (E3)
    """
    H, W = X.shape
    qs = H // 2  # QUAD_SIZE

    # Estrarre i 4 quadranti come array 1D
    quads_raw = [
        X[0:qs, 0:qs].reshape(-1),   # top-left
        X[0:qs, qs:W].reshape(-1),   # top-right
        X[qs:H, 0:qs].reshape(-1),   # bottom-left
        X[qs:H, qs:W].reshape(-1),   # bottom-right
    ]

    # Feature medie: n_qubits gruppi di chunk pixel ciascuno
    chunk = len(quads_raw[0]) // n_qubits  # pixel per gruppo
    quad_feats = np.stack([
        np.array([q[i*chunk:(i+1)*chunk].mean() for i in range(n_qubits)])
        for q in quads_raw
    ], axis=0).astype(np.float32)           # (4, n_qubits)

    # Pixel completi per E3 (amplitude encoding)
    amp_size = 2 ** n_qubits
    quad_full = np.stack([
        q[:amp_size] if len(q) >= amp_size
        else np.pad(q, (0, amp_size - len(q)))  # zero-padding se necessario
        for q in quads_raw
    ], axis=0).astype(np.float32)               # (4, 2^n_qubits)

    return quad_feats, quad_full
```

### Aggiornamento di `get_features`

```python
def get_features(dataset_list, cache, idx: int):
    if idx in cache:
        return cache[idx]
    x, y = dataset_list[idx]
    X = image_to_numpy(x)  # invariata

    quad_feats, quad_full = extract_quad_features(X, N_QUBITS)
    gA4 = global_set_A_features(X)  # invariata (usa l'immagine intera)

    sample = {
        "quad_means": torch.tensor(quad_feats, dtype=torch.float32),  # (4, N_QUBITS)
        "quads_amp":  torch.tensor(quad_full,  dtype=torch.float32),  # (4, 2^N_QUBITS)
        "gA4":        torch.tensor(gA4,        dtype=torch.float32),  # (4,) — invariato
        "y":          int(y),
    }
    cache[idx] = sample
    return sample
```

> **Nota:** la chiave `"quads16"` diventa `"quads_amp"` (dimensione variabile). Aggiornare tutti i riferimenti nel model.

---

## 5. Blocco 3 — Circuito quantistico: `convN` e `poolN`

### Dove si trova nel file

Sezione `# Shared quantum kernel (conv + pool)`, funzioni `conv4` e `pool4`.

### Analisi delle funzioni originali

**`conv4`** (8 parametri):
- Layer 1: `RY` su tutti e 4 i qubit → 4 parametri
- Layer 2: CNOT ring (4 coppie) + `RZ` su target → 4 parametri
- Struttura: `(0→1), (2→3), (1→2), (3→0)` — ring chiuso

**`pool4`** (4 parametri):
- 2 × `CRZ(ctrl=odd, tgt=even)`
- 2 × `CRX(ctrl=shifted, tgt=...)`

### Generalizzazione a N qubit

La struttura del ring si generalizza naturalmente. Con N qubit il ring ha N archi. Il pattern "coppie pari poi coppie dispari" è il **brick-wall entanglement**, che offre il miglior compromesso expressivity/trainabilità per QCNN.

```python
def convN(theta, wires):
    """
    Layer conv-like per N qubit con brick-wall entanglement.

    Struttura:
      - RY layer:         N parametri  (theta[0..N-1])
      - CNOT pari + RZ:   N//2 parametri  (theta[N..N + N//2 - 1])
      - CNOT dispari + RZ: (N-1)//2 parametri
      - CNOT ring-close + RZ: 1 parametro
    Totale parametri: 2*N + 1  (ring chiuso)

    Args:
        theta: tensor di dimensione 2*N+1
        wires:  lista di N indici di wire
    """
    N = len(wires)

    # RY su tutti i qubit
    for i in range(N):
        qml.RY(theta[i], wires=wires[i])

    # Entanglement brick-wall
    cnt = N
    # Coppie pari: (0,1), (2,3), (4,5), ...
    for i in range(0, N - 1, 2):
        qml.CNOT(wires=[wires[i], wires[i+1]])
        qml.RZ(theta[cnt], wires=wires[i+1])
        cnt += 1
    # Coppie dispari: (1,2), (3,4), (5,6), ...
    for i in range(1, N - 1, 2):
        qml.CNOT(wires=[wires[i], wires[i+1]])
        qml.RZ(theta[cnt], wires=wires[i+1])
        cnt += 1
    # Chiusura ring: (N-1) → 0
    qml.CNOT(wires=[wires[-1], wires[0]])
    qml.RZ(theta[cnt], wires=wires[0])


def poolN(phi, wires):
    """
    Layer pooling per N qubit (N deve essere pari).

    Struttura (invariata rispetto a pool4, generalizzata):
      - N//2 × CRZ(ctrl=odd,  tgt=even)
      - N//2 × CRX(ctrl=next, tgt=odd)
    Totale parametri: N

    Args:
        phi:   tensor di dimensione N
        wires: lista di N indici di wire
    """
    N = len(wires)
    half = N // 2

    for i in range(half):
        qml.CRZ(phi[i], wires=[wires[2*i+1], wires[2*i]])
    for i in range(half):
        ctrl = (2*i + 2) % N
        tgt  = (2*i + 1) % N
        qml.CRX(phi[half + i], wires=[wires[ctrl], wires[tgt]])
```

### Conteggio parametri

| N_QUBITS | `theta_conv` | `phi_pool` | Totale kernel |
|---|---|---|---|
| 4 (originale) | 8 | 4 | 12 |
| 8 | 17 | 8 | 25 |
| 16 | 33 | 16 | 49 |

---

## 6. Blocco 4 — Embeddings (E2, E3, E4, E1)

### 6.1 E2 — Fixed Angle Encoding

**Modifica:** banale. Aggiungere gate `RY` per ogni qubit aggiuntivo.

```python
def embed_E2_local(feats_N):
    """
    feats_N: tensor (N_QUBITS,) con valori in [0,1]
    Gate: RY(π * x_i) su wire i
    """
    for i in range(len(feats_N)):
        qml.RY(math.pi * feats_N[i], wires=i)
```

**Nessun nuovo parametro.** Difficoltà: minima.

---

### 6.2 E4 — Learned Angle Encoding

**Modifica:** stesso pattern di E2, ma i coefficienti `a` e `c` diventano `N_QUBITS`-dimensionali invece di 4-dimensionali.

```python
def embed_E4_local(feats_N, aN, cN):
    """
    feats_N: tensor (N_QUBITS,) feature in [0,1]
    aN, cN:  tensor (N_QUBITS,) parametri addestrabili
    Gate:    RY(a_i * π * x_i + c_i) su wire i
    """
    for i in range(len(feats_N)):
        qml.RY(aN[i] * (math.pi * feats_N[i]) + cN[i], wires=i)
```

**Nuovi parametri:** da 4 a `N_QUBITS` per `a` e per `c` (×4 quadranti → totale `4*N_QUBITS` ciascuno nel model). Difficoltà: minima.

---

### 6.3 E3 — Amplitude Encoding

Questo è il caso più delicato. `qml.AmplitudeEmbedding` con N qubit richiede un vettore di **2^N** ampiezze normalizzate.

```python
def embed_E3_amp(amp_2N):
    """
    amp_2N: tensor (2^N_QUBITS,) valori grezzi in [0,1]
    Normalizza e codifica come stato quantistico.
    """
    amp = torch.clamp(amp_2N, 0.0, 1.0)
    nrm = torch.linalg.norm(amp)
    if nrm.item() < 1e-12:
        amp = torch.zeros_like(amp)
        amp[0] = 1.0  # fallback: |0⟩
    else:
        amp = amp / nrm
    qml.AmplitudeEmbedding(amp, wires=LOCAL_WIRES, normalize=False)
```

**Vincolo critico:** il gate di state preparation ha una profondità circuitale O(2^N) — non parallelizzabile in modo efficiente. Con N=8 il circuito di preparation ha già centinaia di gate. Vedere §10 per le strategie.

---

### 6.4 E1 — Hybrid Global Injection

E1 usa un qubit globale aggiuntivo (wire `N_QUBITS`) che riceve feature globali dell'immagine intera (mean, variance, gradient, asymmetry) e le "inietta" nei qubit locali tramite CNOT.

**Modifiche:**

La funzione `inject_global_on_wire` rimane **identica** (opera solo sul wire `GLOBAL_WIRE = N_QUBITS`).

La funzione `fuse_global_to_locals` va estesa da 4 a N qubit:

```python
def fuse_global_to_locals(lam=LAMBDA_FUSION):
    """
    Iniezione del qubit globale in tutti N qubit locali.
    Era: for i in range(4)
    Ora: for i in LOCAL_WIRES
    """
    for i in LOCAL_WIRES:
        qml.CNOT(wires=[GLOBAL_WIRE, i])
        qml.RZ(lam, wires=i)
        qml.CNOT(wires=[GLOBAL_WIRE, i])
```

**Costo aggiuntivo:** da 3×4=12 gate a 3×N gate per l'iniezione. Difficoltà: media (la logica è invariata, si aggiunge solo il loop).

---

## 7. Blocco 5 — Devices e QNodes

### Dove si trova nel file

Sezione `# Devices` e `# QNodes`.

### Devices

```python
# Sostituisce dev4 e dev5
LOCAL_WIRES = list(range(N_QUBITS))
GLOBAL_WIRE = N_QUBITS               # usato solo da E1

devN,  devN_name,  diffN  = make_device(N_QUBITS)
devN1, devN1_name, diffN1 = make_device(N_QUBITS + 1)   # E1 only

QNODEN_KW  = dict(interface="torch", diff_method=diffN)
QNODEN1_KW = dict(interface="torch", diff_method=diffN1)
```

> **Nota su `diff_method`:** con N_QUBITS ≥ 8 il backend `lightning.qubit` con `adjoint` differenziation è **fortemente raccomandato**. È da 10× a 100× più veloce di `parameter-shift` per la simulazione classica. Il codice di Vittori tenta già `lightning.qubit` con fallback su `default.qubit`.

### QNodes

Tutti i QNode cambiano solo nella lista dei wire e nelle chiamate alle funzioni:

```python
@qml.qnode(devN, **QNODEN_KW)
def qnode_quadrant_E2(feats_N, theta_conv, phi_pool):
    embed_E2_local(feats_N)                   # N gate RY
    convN(theta_conv, wires=LOCAL_WIRES)      # brick-wall
    poolN(phi_pool,   wires=LOCAL_WIRES)      # CRZ + CRX
    return [qml.expval(qml.PauliZ(i)) for i in LOCAL_WIRES]


@qml.qnode(devN, **QNODEN_KW)
def qnode_quadrant_E3(amp_2N, theta_conv, phi_pool):
    embed_E3_amp(amp_2N)                      # AmplitudeEmbedding
    convN(theta_conv, wires=LOCAL_WIRES)
    poolN(phi_pool,   wires=LOCAL_WIRES)
    return [qml.expval(qml.PauliZ(i)) for i in LOCAL_WIRES]


@qml.qnode(devN, **QNODEN_KW)
def qnode_quadrant_E4(feats_N, aN, cN, theta_conv, phi_pool):
    embed_E4_local(feats_N, aN, cN)
    convN(theta_conv, wires=LOCAL_WIRES)
    poolN(phi_pool,   wires=LOCAL_WIRES)
    return [qml.expval(qml.PauliZ(i)) for i in LOCAL_WIRES]


@qml.qnode(devN1, **QNODEN1_KW)
def qnode_quadrant_E1(feats_N, gA4_vec, aN, cN,
                      theta_conv, phi_pool,
                      include_global_readout: bool,
                      omega=None, use_reupload=True):
    embed_E4_local(feats_N, aN, cN)           # local: N qubit
    inject_global_on_wire(gA4_vec, omega=omega,
                          use_reupload=use_reupload)   # wire GLOBAL_WIRE
    fuse_global_to_locals()                   # N CNOT-RZ-CNOT
    convN(theta_conv, wires=LOCAL_WIRES)
    poolN(phi_pool,   wires=LOCAL_WIRES)
    outs = [qml.expval(qml.PauliZ(i)) for i in LOCAL_WIRES]
    if include_global_readout:
        outs.append(qml.expval(qml.PauliZ(GLOBAL_WIRE)))
    return outs
```

---

## 8. Blocco 6 — `QuanvEmbedModel`

### `__init__`: aggiornamento delle dimensioni

```python
class QuanvEmbedModel(torch.nn.Module):
    def __init__(self, variant: str, fair_dim_match: bool = True):
        super().__init__()
        self.variant = variant
        self.fair_dim_match = fair_dim_match

        N = N_QUBITS
        N_TOTAL = N * 4   # N qubit × 4 quadranti

        # --- E1 local params (N_TOTAL invece di 16)
        self.e1_aN = torch.nn.Parameter(
            torch.full((N_TOTAL,), float(CONFIG["E1_INIT_A"]), dtype=torch.float32))
        self.e1_cN = torch.nn.Parameter(
            torch.zeros(N_TOTAL, dtype=torch.float32))

        # --- E1 global omega (invariato)
        if CONFIG["E1NP_OMEGA_TRAINABLE"]:
            self.e1np_omega = torch.nn.Parameter(
                torch.tensor(float(CONFIG["E1NP_OMEGA_FIXED"]), dtype=torch.float32))
        else:
            self.register_buffer("e1np_omega",
                torch.tensor(float(CONFIG["E1NP_OMEGA_FIXED"]), dtype=torch.float32))

        # --- E4 params (N_TOTAL invece di 16)
        self.e4_aN = torch.nn.Parameter(torch.ones (N_TOTAL, dtype=torch.float32))
        self.e4_cN = torch.nn.Parameter(torch.zeros(N_TOTAL, dtype=torch.float32))

        # --- Kernel quantistico (2*N+1 e N invece di 8 e 4)
        self.theta_conv = torch.nn.Parameter(
            0.01 * torch.randn(2*N + 1, dtype=torch.float32))
        self.phi_pool   = torch.nn.Parameter(
            torch.zeros(N, dtype=torch.float32))

        # --- Head MLP (input = 4*N invece di 16)
        out_dim = 4 * N
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(out_dim),
            torch.nn.Linear(out_dim, max(32, out_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.05),
            torch.nn.Linear(max(32, out_dim), N_CLASSES),
        )
```

### `features_from_sample`: aggiornamento del forward

Il meccanismo di slicing `QUAD_PATCH_IDXS` va sostituito con uno slicing diretto su `N_QUBITS`:

```python
def features_from_sample(self, sample: dict):
    feats = []
    N = N_QUBITS

    if self.variant == "E2":
        quad_means = sample["quad_means"]   # (4, N)
        for q in range(4):
            out = qnode_quadrant_E2(
                quad_means[q],
                self.theta_conv, self.phi_pool)
            feats.append(torch.stack(out))

    elif self.variant == "E3":
        quads_amp = sample["quads_amp"]     # (4, 2^N)
        for q in range(4):
            out = qnode_quadrant_E3(
                quads_amp[q],
                self.theta_conv, self.phi_pool)
            feats.append(torch.stack(out))

    elif self.variant == "E4":
        quad_means = sample["quad_means"]   # (4, N)
        for q in range(4):
            sl = slice(q*N, (q+1)*N)       # parametri del quadrante q
            out = qnode_quadrant_E4(
                quad_means[q],
                self.e4_aN[sl], self.e4_cN[sl],
                self.theta_conv, self.phi_pool)
            feats.append(torch.stack(out))

    elif self.variant == "E1":
        quad_means = sample["quad_means"]   # (4, N)
        gA4        = sample["gA4"]          # (4,) — invariato
        include_global = not self.fair_dim_match
        for q in range(4):
            sl = slice(q*N, (q+1)*N)
            out = qnode_quadrant_E1(
                quad_means[q], gA4,
                self.e1_aN[sl], self.e1_cN[sl],
                self.theta_conv, self.phi_pool,
                include_global_readout=include_global,
                omega=self.e1np_omega,
                use_reupload=CONFIG["E1NP_USE_REUPLOAD"])
            feats.append(torch.stack(out))

    feat_vec = torch.cat(feats, dim=0)   # (4*N,)
    return feat_vec.to(dtype=torch.float32)
```

---

## 9. Blocco 7 — Training loop e diagnostica

### Cosa rimane invariato

Il training loop (`run_one`, `run_all`), `eval_subset`, `stratified_split_indices`, `group_grad_norms`, `confusion_matrix` e tutto il codice di plotting **non richiedono modifiche** se le modifiche ai blocchi precedenti sono state fatte correttamente. Il feat_vec ha dimensione diversa ma il forward del model gestisce già tutto.

### Cosa aggiornare

**`group_grad_norms`:** cambiare i riferimenti ai parametri rinominati:

```python
def group_grad_norms(model: QuanvEmbedModel):
    return {
        "grad_e1_embed": grad_norm_of_params(
            [model.e1_aN, model.e1_cN,
             getattr(model, "e1np_omega", None)]
        ) if model.variant == "E1" else 0.0,
        "grad_e4_embed": grad_norm_of_params(
            [model.e4_aN, model.e4_cN]
        ) if model.variant == "E4" else 0.0,
        "grad_qkernel":  grad_norm_of_params(
            [model.theta_conv, model.phi_pool]),
        "grad_head":     grad_norm_of_params(
            list(model.head.parameters())),
    }
```

**Optimizer param groups** in `run_one`:

```python
embed_params = []
if variant == "E1":
    embed_params += [model.e1_aN, model.e1_cN,
                     getattr(model, "e1np_omega", None)]
    embed_params = [p for p in embed_params if p is not None]
elif variant == "E4":
    embed_params += [model.e4_aN, model.e4_cN]
# Resto invariato
```

---

## 10. Gestione speciale di E3 (Amplitude Encoding)

### Il problema

Con N_QUBITS qubit, `AmplitudeEmbedding` richiede un vettore di 2^N ampiezze. La tabella seguente mostra le combinazioni valide:

| N_QUBITS | Ampiezze richieste | IMAGE_SIZE necessario | Pixel/quadrante | Fattibile? |
|---|---|---|---|---|
| 4 | 16 | 8 | 16 | ✓ (baseline) |
| 8 | 256 | 32 | 256 | ✓ ma lento |
| 16 | 65.536 | — | — | ✗ non pratico |

### Strategia consigliata: disabilitazione condizionale

Il guard nel CONFIG (§3) disabilita automaticamente E3 se i pixel disponibili sono insufficienti. Questo permette di eseguire E2/E4 a 8 qubit senza modificare altro.

### Strategia alternativa: run separato per E3

Se si vuole comunque testare E3 a 8 qubit, creare un secondo CONFIG con `IMAGE_SIZE=32` e `VARIANTS=["E3"]`:

```python
CONFIG_E3_8Q = {
    **CONFIG,
    "N_QUBITS":   8,
    "IMAGE_SIZE": 32,
    "VARIANTS":   ["E3"],
    "BASE_DIR":   "./e3_8qubit_img32",
}
```

Rieseguire tutto il pipeline con questo config in un notebook separato e poi confrontare i risultati in analisi.

---

## 11. Nuovo dataset — Fashion-MNIST

Fashion-MNIST è la scelta ideale per il nuovo dataset: stessa struttura di MNIST (28×28 grayscale, 10 classi, ~60k train), ma classificazione più difficile. Il codice richiede modifiche minime.

### Modifica al caricamento dati

```python
# Sostituire:
train_dataset = torchvision.datasets.MNIST(...)
test_dataset  = torchvision.datasets.MNIST(...)

# Con:
train_dataset = torchvision.datasets.FashionMNIST(
    "./data", train=True, download=True, transform=transform)
test_dataset  = torchvision.datasets.FashionMNIST(
    "./data", train=False, download=True, transform=transform)
```

### Classi di Fashion-MNIST

| Indice | Classe |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### Scelta delle classi per il sottoinsieme

Per mantenere comparabilità con il lavoro di Vittori (5 classi), usare classi visivamente distinte:

```python
# Sottoinsieme consigliato (5 classi distinte)
TARGET_CLASSES = [0, 1, 5, 7, 8]  # T-shirt, Trouser, Sandal, Sneaker, Bag
LABEL_MAP = {c: i for i, c in enumerate(TARGET_CLASSES)}
```

### Aggiornamento del BASE_DIR

```python
CONFIG["BASE_DIR"] = f"./fmnist_{N_QUBITS}q_{IMAGE_SIZE}px"
```

---

## 12. Checklist delle modifiche

### Modifiche obbligatorie (core)

- [ ] Aggiungere `N_QUBITS` e `IMAGE_SIZE` in `CONFIG`
- [ ] Aggiungere le costanti derivate e i sanity check dopo `CONFIG`
- [ ] Aggiornare `transform` per usare `IMAGE_SIZE`
- [ ] Sostituire `extract_patches_2x2` + `quadrants_4x4_flat` + `QUAD_PATCH_IDXS` con `extract_quad_features`
- [ ] Aggiornare `get_features` (nuove chiavi `quad_means`, `quads_amp`)
- [ ] Sostituire `conv4` con `convN`
- [ ] Sostituire `pool4` con `poolN`
- [ ] Aggiornare `embed_E2_local` (loop su `len(feats_N)`)
- [ ] Aggiornare `embed_E4_local` (loop su `len(feats_N)`)
- [ ] Aggiornare `fuse_global_to_locals` (loop su `LOCAL_WIRES`)
- [ ] Sostituire `dev4`/`dev5` con `devN`/`devN1`
- [ ] Aggiornare tutti e 4 i QNode (wire e chiamate)
- [ ] Aggiornare `QuanvEmbedModel.__init__` (dimensioni parametri, head)
- [ ] Aggiornare `features_from_sample` (slicing, chiavi sample)
- [ ] Aggiornare `group_grad_norms` (nomi parametri)
- [ ] Aggiornare param groups in `run_one`

### Modifiche opzionali (nuovo dataset)

- [ ] Sostituire `MNIST` con `FashionMNIST`
- [ ] Aggiornare `TARGET_DIGITS` → `TARGET_CLASSES`
- [ ] Aggiornare `BASE_DIR` per riflettere dataset + qubit + image size

### Verifiche post-modifica

- [ ] Eseguire `draw_circuits()` per verificare visivamente i nuovi circuiti
- [ ] Verificare che `count_trainable_params` ritorni un valore coerente con le tabelle §13
- [ ] Eseguire un singolo sample in forward per verificare che le dimensioni siano corrette
- [ ] Verificare che gradient norm non sia zero alla prima epoca (barren plateau check)

---

## 13. Tabella riepilogativa dei parametri

### Parametri totali per configurazione

| Config | `theta_conv` | `phi_pool` | `e4_aN`/`e4_cN` | Head (→ 5 classi) | **Totale E4** |
|---|---|---|---|---|---|
| N=4 (originale) | 8 | 4 | 16+16 | 16→32→5 ≈ 709 | ~773 |
| N=8 | 17 | 8 | 32+32 | 32→32→5 ≈ 1189 | ~1278 |
| N=16 | 33 | 16 | 64+64 | 64→64→5 ≈ 4549 | ~4726 |

### Vettore di feature per quadrante e output totale

| N_QUBITS | Feature/quadrante (E2/E4) | Ampiezze E3 | Output totale (4 quad) |
|---|---|---|---|
| 4 | 4 | 16 | 16 |
| 8 | 8 | 256 | 32 |
| 16 | 16 | 65.536 | 64 |

---

## 14. Rischi e mitigazioni

### Barren Plateau

**Rischio:** con N_QUBITS ≥ 8 e circuiti profondi, la norma del gradiente scala come O(2^{-N}). I parametri del kernel quantistico potrebbero non ricevere segnale utile.

**Mitigazione:**
1. Inizializzare `theta_conv` con valori piccoli (già presente: `0.01 * randn`) — non usare init random uniforme
2. Usare `CLIP_NORM` aggressivo nei primi epoch (già presente: `0.5`)
3. Learning rate separato per `qkernel` più basso rispetto alla head (già presente: `LR_QKERNEL=1e-3`)
4. Monitorare `grad_qkernel` dal primo epoch: se < 1e-5 il plateau è attivo
5. Considerare di aggiungere un terzo layer con re-uploading dei dati per aumentare il segnale

### Costo computazionale

**Rischio:** a 8 qubit ogni chiamata al QNode richiede la simulazione di uno stato da 256 ampiezze complesse. Con parameter-shift, ogni parametro richiede 2 esecuzioni aggiuntive.

**Mitigazione:**
1. Usare `lightning.qubit` con `adjoint` differentiazione (già tentato automaticamente)
2. Ridurre `TRAIN_SAMPLES` a 500 per i run esplorativi iniziali
3. Aumentare `ACC_STEPS` per ridurre le chiamate all'optimizer
4. Se disponibile, usare `lightning.gpu` con una GPU (CUDA)

### E3 con immagini più grandi

**Rischio:** passare a `IMAGE_SIZE=32` cambia significativamente la distribuzione dell'input rispetto a Vittori (8×8 → 32×32), rendendo il confronto meno diretto.

**Mitigazione:** eseguire E3 in un run separato documentando esplicitamente la differenza di preprocessing. Non mischiare risultati E3-32px con E2/E4-8px nello stesso plot comparativo.

### Comparabilità con Vittori

**Rischio:** cambiare simultaneamente qubit e dataset rende difficile attribuire le differenze di performance alla causa corretta.

**Mitigazione:** piano di esperimenti a variabile singola:
1. Run A: N=4, MNIST (replica Vittori → baseline)
2. Run B: N=8, MNIST (effetto qubit isolato)
3. Run C: N=4, Fashion-MNIST (effetto dataset isolato)
4. Run D: N=8, Fashion-MNIST (combinazione)

---

*Documento generato a partire dall'analisi del codice `Codice.ipynb` di Lorenzo Vittori e della sua tesi magistrale.*
