# Modifiche: Test.py → Test_8_qubit.py

Documento di tracciamento di tutte le modifiche apportate per estendere l'architettura da 4 qubit (MNIST 8×8) a 8 qubit (Fashion-MNIST 16×16).

---

## 1. Dataset e immagini

| | Test.py | Test_8_qubit.py |
|---|---|---|
| Dataset | MNIST | Fashion-MNIST (`FashionMNIST`) |
| Classi | 5 (0,1,2,3,8 filtrate) | 10 (tutte, nessun filtraggio) |
| Risoluzione | 8×8 | 16×16 |
| `transforms.Resize` | `(8, 8)` | `(16, 16)` |
| Caricamento | `filter_and_remap()` (filtra e rimappa le label) | list comprehension diretta `[(x, int(y)) for x, y in dataset]` |

`TARGET_DIGITS`, `LABEL_MAP`, `filter_and_remap` rimossi perché non necessari.

---

## 2. Feature extraction

### `extract_patches_2x2`
- **Prima**: loop `for r in range(4): for c in range(4)` → 16 patch, output shape `(16, 4)`
- **Dopo**: loop `for r in range(8): for c in range(8)` → 64 patch, output shape `(64, 4)`

### `global_set_A_features`
- **Prima**: `dx_o = dx[0:7, 0:7]`, `dy_o = dy[0:7, 0:7]` (solo angolo top-left di un'immagine 8×8)
- **Dopo**: `dx_o = dx[0:15, 0:15]`, `dy_o = dy[0:15, 0:15]` (regione di overlap completa per immagine 16×16)
- Output invariato: 4 statistiche globali `(g1, g2, g3, g4)`.

### `quadrants_4x4_flat` → `quadrants_8x8_flat`
- **Prima**: 4 quadranti di 4×4 = 16 pixel ciascuno, shape `(4, 16)`
- **Dopo**: 4 quadranti di 8×8 = 64 pixel ciascuno, shape `(4, 64)`
- Usata solo da E3 per l'amplitude embedding.

### `QUAD_PATCH_IDXS` → `QUAD_META_PATCH_IDXS`

Questa è la modifica strutturalmente più rilevante per E2/E4/E1.

**Problema**: con 16×16, ogni quadrante contiene 16 patch (griglia 4×4), ma servono esattamente 8 feature per 8 qubit.

**Soluzione**: si raggruppano le 16 patch in 8 **meta-patch** coppie di patch orizzontalmente adiacenti. La feature di ogni meta-patch è la media aritmetica dei mean delle due patch.

Struttura della griglia di patch (8×8, indice = `r*8+c`):
```
 0  1  2  3  4  5  6  7
 8  9 10 11 12 13 14 15
16 17 18 19 20 21 22 23
24 25 26 27 28 29 30 31
32 33 34 35 36 37 38 39
40 41 42 43 44 45 46 47
48 49 50 51 52 53 54 55
56 57 58 59 60 61 62 63
```

| Quadrante | Patch della griglia | Meta-patch (8 coppie) |
|---|---|---|
| Q0 (top-left, pixel [0:8,0:8]) | righe 0-3, col 0-3 | [0,1],[2,3],[8,9],[10,11],[16,17],[18,19],[24,25],[26,27] |
| Q1 (top-right, pixel [0:8,8:16]) | righe 0-3, col 4-7 | [4,5],[6,7],[12,13],[14,15],[20,21],[22,23],[28,29],[30,31] |
| Q2 (bot-left, pixel [8:16,0:8]) | righe 4-7, col 0-3 | [32,33],[34,35],[40,41],[42,43],[48,49],[50,51],[56,57],[58,59] |
| Q3 (bot-right, pixel [8:16,8:16]) | righe 4-7, col 4-7 | [36,37],[38,39],[44,45],[46,47],[52,53],[54,55],[60,61],[62,63] |

Il calcolo in `get_features`:
```python
quad_means[q][i] = (means64[pair[0]] + means64[pair[1]]) * 0.5
```
Output shape: `(4, 8)` (era `(4, 4)`).

### `get_features` — dizionario sample

| Chiave | Test.py | Test_8_qubit.py |
|---|---|---|
| `patches` | `(16, 4)` | `(64, 4)` |
| `means16` → `means64` | `(16,)` | `(64,)` |
| `quad_means` | `(4, 4)` | `(4, 8)` |
| `quads16` → `quads64` | `(4, 16)` | `(4, 64)` |
| `gA4` | `(4,)` | `(4,)` invariato |

---

## 3. Dispositivi quantistici

| | Test.py | Test_8_qubit.py |
|---|---|---|
| Dispositivo principale | `dev4` (4 wire) | `dev8` (8 wire) |
| Dispositivo E1 | `dev5` (5 wire: 4 locali + 1 globale) | `dev9` (9 wire: 8 locali + 1 globale) |
| Wire globale (E1) | wire `4` | wire `8` |

---

## 4. Circuiti quantistici

### `conv4` → `conv8`

`conv4` usa 8 parametri: 4 RY + 4 RZ in topologia ad anello.

`conv8` usa **16 parametri**: 8 RY (uno per qubit) + 8 RZ sugli archi di un anello a 8 nodi con CNOT nearest-neighbor:

```
Primo strato CNOT (coppie pari):  q0→q1, q2→q3, q4→q5, q6→q7  (theta[8..11])
Secondo strato CNOT (coppie dispari): q1→q2, q3→q4, q5→q6, q7→q0  (theta[12..15])
```

### `pool4` → `pool8`

`pool4` usa 4 parametri: 2 CRZ + 2 CRX.

`pool8` usa **8 parametri**: 4 CRZ + 4 CRX sulle stesse coppie scalate a 8 qubit:
```
CRZ: (q1→q0), (q3→q2), (q5→q4), (q7→q6)   (phi[0..3])
CRX: (q2→q1), (q4→q3), (q6→q5), (q0→q7)   (phi[4..7])
```

---

## 5. Funzioni di embedding

| Funzione | Test.py | Test_8_qubit.py |
|---|---|---|
| `embed_E2_local` | loop su 4 qubit | `embed_E2_local_8`: loop su 8 qubit |
| `embed_E4_local` | loop su 4 qubit | `embed_E4_local_8`: loop su 8 qubit |
| `inject_global_on_wire` | wire fisso `4` | `inject_global_on_wire_8`: wire fisso `8` |
| `fuse_global_to_locals` | CNOT da wire 4 a wires 0-3 | `fuse_global_to_locals_8`: CNOT da wire 8 a wires 0-7 |

---

## 6. QNode

| QNode | Test.py | Test_8_qubit.py |
|---|---|---|
| E2 | `qnode_quadrant_E2(quad_means_4, ...)` su `dev4` | `qnode_quadrant_E2_8(quad_means_8, ...)` su `dev8` |
| E4 | `qnode_quadrant_E4(quad_means_4, a4, c4, ...)` su `dev4` | `qnode_quadrant_E4_8(quad_means_8, a8, c8, ...)` su `dev8` |
| E1 | `qnode_quadrant_E1_optionB(...)` su `dev5` | `qnode_quadrant_E1_8(...)` su `dev9` |
| E3 | `qnode_quadrant_E3(quad_amp_16, ...)` su `dev4`, input 16 elem = 2^4 → fit esatto | `qnode_quadrant_E3_8(quad_amp_64, ...)` su `dev8`, input 64 elem → zero-pad a 256 = 2^8 |

### Gestione E3: zero-padding

In `qnode_quadrant_E3_8`, dopo la normalizzazione del vettore a 64 elementi, si applica:
```python
amp = amp / nrm                         # normalizza a norma 1 in R^64
amp = torch.nn.functional.pad(amp, (0, 192))  # estende a 256 con zeri
```
Il vettore risultante ha già norma 1 (aggiungere zeri non cambia la norma), quindi `AmplitudeEmbedding` riceve un vettore valido di 256 = 2^8 componenti.

Si usa il parametro nativo `pad_with=0.0, normalize=True` di `AmplitudeEmbedding` (documentazione PennyLane stabile): il padding da 64 a 256 e la normalizzazione sono gestiti internamente. L'unico caso gestito manualmente è il fallback su vettori quasi-zero (norma < 1e-12), che causerebbe una divisione per zero con `normalize=True`.

---

## 7. Modello `QuanvEmbedModel`

### Parametri trainabili

| Parametro | Test.py | Test_8_qubit.py | Motivazione |
|---|---|---|---|
| `e1_a16` | shape `(16,)` | `e1_a32` shape `(32,)` | 4 quadranti × 8 meta-patch |
| `e1_c16` | shape `(16,)` | `e1_c32` shape `(32,)` | idem |
| `e4_a16` | shape `(16,)` | `e4_a32` shape `(32,)` | idem |
| `e4_c16` | shape `(16,)` | `e4_c32` shape `(32,)` | idem |
| `theta_conv` | shape `(8,)` | shape `(16,)` | parametri di `conv8` |
| `phi_pool` | shape `(4,)` | shape `(8,)` | parametri di `pool8` |

### `_quadrant_params_from_a16c16` → `_quadrant_params`

Prima: indicizzava con `QUAD_PATCH_IDXS` (indici irregolari).
Dopo: semplice `reshape(4, 8)` — il vettore da 32 è già organizzato in 4 blocchi da 8.

### Dimensione feature vector e head

| | Test.py | Test_8_qubit.py |
|---|---|---|
| Feature vec (default) | 16 (4 qubit × 4 quadranti) | 32 (8 qubit × 4 quadranti) |
| Feature vec E1 no fair_dim_match | 20 (5 output × 4 quadranti) | 36 (9 output × 4 quadranti) |
| Head hidden | `Linear(16→32)` | `Linear(32→64)` |
| Head output | `Linear(32→5)` | `Linear(64→10)` |

---

## 8. Training loop (`run_one`)

Solo i riferimenti ai nomi dei parametri aggiornati:
- `model.e1_a16`, `model.e1_c16` → `model.e1_a32`, `model.e1_c32`
- `model.e4_a16`, `model.e4_c16` → `model.e4_a32`, `model.e4_c32`

Il resto della logica (gradient accumulation, gradient clipping, optimizer con LR separati, diagnostic backward, eval loop) è **invariato**.

---

## 9. Main

- `get_datasets()` usa `FashionMNIST` invece di `MNIST`
- Rimossa la chiamata a `filter_and_remap`; i dati vengono caricati come lista diretta
- `BASE_DIR` = `./fashion_8qubit_16x16`
- Stampa di diagnostica aggiornata: `dev8`/`dev9` invece di `dev4`/`dev5`

---

## Riepilogo numerico

| Grandezza | 4 qubit (Test.py) | 8 qubit (Test_8_qubit.py) |
|---|---|---|
| Immagine | 8×8 = 64 pixel | 16×16 = 256 pixel |
| Patch totali | 16 (griglia 4×4) | 64 (griglia 8×8) |
| Feature per quadrante (E2/E4/E1) | 4 patch means | 8 meta-patch means |
| Pixel per quadrante (E3) | 16 = 2^4 (fit esatto) | 64 → zero-pad a 256 = 2^8 |
| Parametri `theta_conv` | 8 | 16 |
| Parametri `phi_pool` | 4 | 8 |
| Parametri embed (E1/E4) | 16+16 | 32+32 |
| Feature vector finale | 16 | 32 |
| Classi | 5 | 10 |
| Head | Linear(16→32)→Linear(32→5) | Linear(32→64)→Linear(64→10) |
