# Changelog: v1 -> v2

**File v1:** `fashion_patch_amplitude_qcnn_32_frozen_diagnostics.py`  
**File v2:** `fashion_patch_amplitude_qcnn_32_v2.py`

Obiettivo: massimizzare cio che impara la parte quantistica, mantenendo la testa classica semplice (`LayerNorm + Linear`).

---

## C1 - Fix bug nel gate di pooling

### Problema
In v1 il pooling usava `PauliX * CRX * PauliX`, che produce un gate **anti-controlled** (condizionato su |0>). Il pooling QCNN standard (Cong et al. 2019) richiede entrambi i gate condizionati su |1>.

### Modifica
```python
# v1 (bug: anti-controlled CRX)
for j, (control, target) in enumerate(pairs):
    qml.PauliX(wires=control)
    qml.CRX(phi_pool[2 * j], wires=[control, target])
    qml.PauliX(wires=control)
    qml.CRZ(phi_pool[2 * j + 1], wires=[control, target])

# v2 (corretto: standard CRX+CRZ entrambi condizionati su |1>)
for j, (control, target) in enumerate(pairs):
    qml.CRX(phi_pool[2 * j],     wires=[control, target])
    qml.CRZ(phi_pool[2 * j + 1], wires=[control, target])
```

### Impatto
Il pooling ora opera come descritto nella letteratura QCNN. I gradienti di `phi_pool` sono calcolati rispetto al gate corretto.

---

## C2 - Secondo strato convoluzionale (`conv4_retained`)

### Motivazione
V1 aveva un solo strato conv+pool. Dopo il pooling i 4 qubit retained `[0,2,4,6]` non subivano ulteriore elaborazione quantistica. Aggiungere un secondo conv approfondisce il circuito senza toccare la testa classica.

### Modifica
Nuova funzione `conv4_retained(theta_conv2)`: 4 RY sui retained wires + 4 CNOT-RZ a formare un anello (0->2->4->6->0). **8 nuovi parametri quantistici.**

```python
def conv4_retained(theta_conv2):
    for k, w in enumerate(KEEP_WIRES):           # 4 RY
        qml.RY(theta_conv2[k], wires=w)
    for k in range(len(KEEP_WIRES) - 1):          # 3 CNOT-RZ
        a, b = KEEP_WIRES[k], KEEP_WIRES[k + 1]
        qml.CNOT(wires=[a, b])
        qml.RZ(theta_conv2[4 + k], wires=b)
        qml.CNOT(wires=[a, b])
    # chiusura anello: 6 -> 0
    qml.CNOT(wires=[KEEP_WIRES[-1], KEEP_WIRES[0]])
    qml.RZ(theta_conv2[7], wires=KEEP_WIRES[0])
    qml.CNOT(wires=[KEEP_WIRES[-1], KEEP_WIRES[0]])
```

Il QNode diventa:
```
AmplitudeEmbedding -> RY(norm_angle) -> conv8 -> pool_8_to_4 -> conv4_retained -> readout
```

`theta_conv2` viene aggiunto al gruppo `qkernel` nell'ottimizzatore e congelato nella run `frozen_qcnn`. Il drift di `theta_conv2` e tracciato come colonna separata in `metrics.csv`.

---

## C3 - Readout esteso: Z + correlazioni ZZ

### Motivazione
V1 leggeva solo 4 expectation values `<Z>` sui retained wires. Le correlazioni a 2 qubit `<ZZ>` catturano entanglement che i singoli Z non possono vedere. Raddoppiare le feature per patch arricchisce il segnale senza aumentare la testa.

### Modifica
```python
# v1: 4 feature/patch
QFEATURES_PER_PATCH = 4
HEAD_IN_DIM = 16

def readout_features():
    return [qml.expval(qml.PauliZ(i)) for i in KEEP_WIRES]

# v2: 8 feature/patch
ZZ_PAIRS            = [(0, 2), (2, 4), (4, 6), (6, 0)]  # anello tra KEEP_WIRES
QFEATURES_PER_PATCH = 8
HEAD_IN_DIM         = 32

def readout_features():
    z_vals  = [qml.expval(qml.PauliZ(i)) for i in KEEP_WIRES]
    zz_vals = [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for i, j in ZZ_PAIRS]
    return z_vals + zz_vals
```

La testa diventa `LayerNorm(32) + Linear(32->10)` — ancora semplice, solo l'input dim cambia.

---

## C4 - Inizializzazione near-zero dei parametri quantistici

### Motivazione
V1 usava `1e-3 * torch.randn(...)`. L'init a zero corrisponde a gate identita: i parametri partono da un punto noto e ben condizionato, riducendo il rischio di barren plateau nei primi epoch.

### Modifica
```python
# v1
self.theta_conv = torch.nn.Parameter(1e-3 * torch.randn(16, dtype=torch.float32))
self.phi_pool   = torch.nn.Parameter(1e-3 * torch.randn(8,  dtype=torch.float32))

# v2
self.theta_conv  = torch.nn.Parameter(torch.zeros(16, dtype=torch.float32))
self.phi_pool    = torch.nn.Parameter(torch.zeros(8,  dtype=torch.float32))
self.theta_conv2 = torch.nn.Parameter(torch.zeros(8,  dtype=torch.float32))
```

---

## C5 - Ribilanciamento dei learning rate

### Motivazione
V1: `LR_HEAD=1e-3`, `LR_QKERNEL=3e-4` -> la testa imparava 3.3x piu velocemente del quantum. Dato che l'obiettivo e far imparare il quantum, i LR vanno bilanciati (o invertiti).

### Modifica
```python
# v1
"LR_HEAD":    1e-3,
"LR_QKERNEL": 3e-4,

# v2
"LR_HEAD":    5e-4,   # ridotto
"LR_QKERNEL": 1e-3,   # aumentato
```

Il rapporto LR_QKERNEL/LR_HEAD passa da 0.3x a 2x.

---

## C6 - Scheduler: CosineAnnealingLR invece di ReduceLROnPlateau

### Motivazione
V1 usava `ReduceLROnPlateau(patience=2, factor=0.5)`: dopo 2 epoch di plateau sul validation loss, il LR veniva dimezzato. I parametri quantistici hanno gradienti piu lenti e piu rumorosi -> venivano penalizzati sproporzionatamente. `CosineAnnealingLR` decade smoothly e in modo deterministico, mantenendo un segnale di gradiente stabile per tutti i parametri per tutta la durata del training.

### Modifica
```python
# v1
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
)
scheduler.step(val_metrics["loss"])   # nel loop

# v2
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG["EPOCHS"], eta_min=1e-6
)
scheduler.step()   # nel loop, nessun argomento
```

---

## C7 - Training piu lungo

### Motivazione
Nei run v1 a 30 epoch la `test_acc` saliva ancora linearmente all'epoch 30. Il modello non aveva convergito.

### Modifica
```python
# v1
"EPOCHS":               15,
"EARLY_STOP_PATIENCE":   5,

# v2
"EPOCHS":               60,
"EARLY_STOP_PATIENCE":  10,
```

---

## C8 - Fix transform + augmentation separata

### Problema 1: Resize vs Pad
V1 usava `transforms.Resize((32, 32))` che applica interpolazione bilineare, distorcendo i valori dei pixel originali. Il commento nel codice diceva "black padding" ma il codice usava resize. `transforms.Pad(2)` aggiunge 2 pixel neri su ogni lato (28x28 -> 32x32) senza nessuna interpolazione.

### Problema 2: stesso transform per train e test
V1 aveva un unico `transform`. L'augmentation non puo essere applicata al test set.

### Modifica
```python
# v1: un solo transform con Resize
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# v2: transform separati
transform_eval = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),   # solo su train
    transforms.Pad(2),
    transforms.ToTensor(),
])
```

`get_datasets()` usa `transform_train` per il training set e `transform_eval` per il test set.

**Nota sull'augmentation e la cache:** `train_data = [(x, y) for x, y in train_dataset]` itera il dataset una volta applicando il transform (incluso il flip random). La cache memorizza i vettori di ampiezza calcolati da queste immagini. L'augmentation e "one-shot" (applicata una volta al caricamento, non ad ogni epoch), ma diversifica comunque il dataset effettivo senza overhead per gli epoch successivi.

---

## C9 - Iniezione della norma per patch nel circuito

### Problema
La normalizzazione per-patch in `patch_amplitudes_4x256` proietta ogni patch su un vettore unitario indipendentemente. Questo cancella il contrasto relativo tra patch: una patch scura e una chiara producono lo stesso stato quantistico dopo l'embedding. L'informazione sulla luminosita relativa e persa.

### Soluzione
Le norme L2 raw di ogni patch vengono preservate e convertite in angoli relativi in [0, pi]:

```python
norm_angles[p] = pi * norm_p / (norm_0 + norm_1 + norm_2 + norm_3)
```

L'angolo viene iniettato nel circuito come rotazione `RY` sul wire 0 dopo `AmplitudeEmbedding`:

```python
@qml.qnode(dev8, **QNODE_KW)
def patch_amplitude_qnode(amp256, norm_angle, theta_conv, phi_pool, theta_conv2):
    qml.AmplitudeEmbedding(amp256, wires=range(N_QUBITS), normalize=True)
    qml.RY(norm_angle, wires=0)   # iniezione contrasto relativo
    conv8(theta_conv)
    ...
```

`norm_angle` e un dato (non un parametro trainable). La `sample` dict include `"norm_angles": tensor(4,)`, collazionato dal DataLoader in `(B, 4)` per batch, passato al QNode come `norm_angles[:, p]` di shape `(B,)`.

---

## Riepilogo parametri

| Componente | v1 | v2 |
|---|---|---|
| `theta_conv` | 16 | 16 |
| `phi_pool` | 8 | 8 |
| `theta_conv2` | - | **8** (C2) |
| **Totale quantum** | **24** | **32** |
| Head LayerNorm | 32 | 64 |
| Head Linear | 170 | 330 |
| **Totale head** | **202** | **394** |
| **Totale trainable** | **226** | **426** |
| **Totale frozen** | **202** | **394** |

---

## Riepilogo CONFIG

| Parametro | v1 | v2 | Motivazione |
|---|---|---|---|
| `EPOCHS` | 15 (eseguito 30) | 60 | modello non convergeva (C7) |
| `EARLY_STOP_PATIENCE` | 5 | 10 | quantum converge piu lentamente (C7) |
| `LR_HEAD` | 1e-3 | 5e-4 | bilanciamento con quantum (C5) |
| `LR_QKERNEL` | 3e-4 | 1e-3 | quantum impara piu veloce (C5) |
| Scheduler | ReduceLROnPlateau(patience=2) | CosineAnnealingLR(T_max=60) | gradiente quantum stabile (C6) |
| Transform | Resize(32,32) bilineare | Pad(2) no interpolazione | pixel esatti (C8) |
| Augmentation | nessuna | HorizontalFlip(p=0.5) solo train | dataset effettivo piu grande (C8) |

---

## Nuove colonne nei file di output

`metrics.csv` e `final_eval.json` includono ora:

- `theta_conv2_drift` — norma L2 dello spostamento di `theta_conv2` dall'inizializzazione
- `final_theta_conv2_drift` — drift finale di `theta_conv2`

Il print per epoch include `th2=...` accanto a `th1=...` e `phi=...`.

---

## Post-run analysis: bug trovati e fix applicati (v2.1)

Dopo il primo run completo di v2 (60 epoch, seed=0), sono emersi due bug critici
rilevabili dall'output di training.

### Bug B1 — `phi_pool` mai aggiornato (bloccante)

**Sintomo:** `phi=0.00e+00` e `phi_pool_drift=0.0` per tutti e 60 gli epoch,
anche nella run trainable.

**Causa:** `qml.CRX` e `qml.CRZ` non producono gradienti validi con
`lightning.qubit` + `diff_method="adjoint"`. La funzione `grad_norm` non
segnala l'errore perche salta silenziosamente i parametri con `.grad is None`.

**Impatto:** Il pooling rimane identita per tutto il training. I qubit discarded
(1,3,5,7) non trasferiscono mai informazione ai retained (0,2,4,6). 8
parametri quantum completamente sprecati.

**Fix:** Sostituzione di `CRX + CRZ` con una struttura CNOT-sandwich che usa
solo gate differenziabili via adjoint (`RY`, `CNOT`, `RZ`). La semantica del
pooling e preservata: il CNOT crea entanglement tra control e target, le
rotazioni apprese su target filtrano l'informazione dal control.

```python
# v2 (bug: CRX+CRZ non differenziabili via adjoint)
for j, (control, target) in enumerate(pairs):
    qml.CRX(phi_pool[2 * j],     wires=[control, target])
    qml.CRZ(phi_pool[2 * j + 1], wires=[control, target])

# v2.1 (fix: CNOT-sandwich, tutti i gate differenziabili)
for j, (control, target) in enumerate(pairs):
    qml.RY(phi_pool[2 * j],     wires=target)
    qml.CNOT(wires=[control, target])
    qml.RZ(phi_pool[2 * j + 1], wires=target)
    qml.CNOT(wires=[control, target])
```

### Bug B2 — `theta_conv_drift == theta_conv2_drift` esattamente

**Sintomo:** `th1` e `th2` identici a molte cifre decimali per tutti e 60 gli
epoch. Confermato nel JSON finale:

```
"final_theta_conv_drift":  0.7523466348648071
"final_theta_conv2_drift": 0.7523466348648071
```

`theta_conv` ha 16 elementi, `theta_conv2` ne ha 8: L2 norme identiche su
60 epoch sono impossibili per coincidenza.

**Causa ipotizzata:** Con `phi_pool.grad = None` (Bug B1), PennyLane adjoint
shiftava gli indici dei gradienti restituiti, assegnando a `theta_conv2` lo
stesso tensore di gradiente di `theta_conv`. Di conseguenza i due parametri
ricevevano update identici per elemento, portando a drift identico.

**Fix:** Correggendo B1, gli indici sono allineati e `theta_conv2` riceve il
suo gradiente indipendente. Verificato dallo smoke test con il check
`theta_conv.grad != theta_conv2.grad`.

### Diagnostico aggiunto (v2.1)

`model_grad_norms` ora restituisce — e il loop salva in `metrics.csv` —
le norme di gradiente separate:

- `grad_theta_conv_mean` — gradiente medio di conv8
- `grad_phi_pool_mean`   — gradiente medio del pooling (era sempre 0 in v2)
- `grad_theta_conv2_mean` — gradiente medio di conv4_retained

Il print per epoch mostra `gconv=... gpool=... gconv2=...` prima di `grad_q`.

### Bug B3 — Simmetria di circuito a zero initialization (bloccante)

**Sintomo:** Anche dopo il fix di B1 e B2, `theta_q[SL_CONV][:8]` e `theta_q[SL_POOL]`
ricevono gradienti identici al passo 0 e rimangono uguali dopo ogni step, rendendo
il pooling layer ridondante rispetto ai primi 8 parametri di conv8.

**Causa:** A zero initialization, tutti i gate del circuito sono identita. Quindi
`conv8[RY su wire w]` e `pool[RY su wire w]` operano sullo stesso stato quantistico
con lo stesso contesto circuitale a destra (tutto il resto e identita). Per il
parameter-shift rule, i loro gradienti sono matematicamente identici. Con update
identici partendo dallo stesso valore, rimangono identici indefinitamente.

**Fix (C4 rev):** Sostituire `torch.zeros(32)` con `1e-3 * torch.randn(32)`.

- Angoli scala 1e-3 rad -> gate quasi-identita -> rischio barren plateau irrilevante
- Valori diversi per segmento -> gradienti diversi gia dal passo 1
- Riprodducibile: `set_global_seed(seed)` e chiamato prima di `PatchAmplitudeQCNN()`

```python
# v2 (zero init — simmetria circuitale a init)
self.theta_q = torch.nn.Parameter(torch.zeros(N_THETA_Q, dtype=torch.float32))

# v2.1 (fix: near-zero random init, rompe la simmetria)
self.theta_q = torch.nn.Parameter(1e-3 * torch.randn(N_THETA_Q, dtype=torch.float32))
```

Verifica: dopo 5 step di training con B=4 campioni sintetici, `max|theta_conv - theta_pool| = 2.98e-3`
(era 2.6e-11 con zero init).

### Tensore quantistico unificato (fix architetturale B2-prereq)

Passare tre tensori separati (`theta_conv`, `phi_pool`, `theta_conv2`) al QNode
causava con alcune versioni di PennyLane adjoint un disallineamento degli indici
di gradiente. Il fix consolidato usa un singolo `theta_q` di 32 elementi con slice
definite da costanti modulo (`SL_CONV`, `SL_POOL`, `SL_CONV2`). Il QNode riceve
un solo tensore trainable e restituisce un gradiente di 32 elementi correttamente
indicizzato.

### Smoke test aggiunto

`smoke_test_v2.py`: 20 check in ~3 minuti, verifica forward/backward, gradienti
per segmento, divergenza parametri dopo 5 step, comportamento frozen, forma QNode.
Eseguire prima di ogni run completo.

### Risultati run v2 originale (pre-fix, per riferimento)

| Metrica | trainable | frozen |
|---|---|---|
| `final_test_acc` | 64.95% | 61.15% |
| `phi_pool_drift` | 0.0 (bug B1) | 0.0 (frozen) |
| `th1 == th2` | si (bug B2) | 0.0 (frozen) |
| `sep_ratio` val | 0.622 | 0.480 |
| `center_dist` val | 1.087 | 1.070 |

Con il fix di B1+B2 ci si aspetta un incremento di test_acc di 2-4 punti
percentuali (8 parametri di pooling ora attivi + update indipendenti di conv4).

---

---

## v2.2 — Tuning (2026-05-23)

**Motivazione:** run v2.1 (seed=0, 60 epoch, 10k train) ha convergito a `test_acc=66.95%` con
`train_loss ≈ val_loss ≈ 0.86` — segnale di underfitting, non overfitting. Il modello ha saturato
il segnale disponibile dal dataset prima di usare tutta la capacità del circuito.

### Risultati run v2.1 (pre-tuning, per riferimento)

| Metrica | trainable | frozen |
|---|---|---|
| `final_test_acc` | **66.95%** | 61.5% |
| `final_val_acc` | 69.35% | 62.9% |
| `phi_pool_drift` | 1.253 ✅ | 0 (frozen) |
| `theta_conv_drift` | 3.403 | 0 |
| `theta_conv2_drift` | 0.966 | 0 |
| `between/within ratio` val | 0.971 | 0.480 |
| `train_loss ≈ val_loss` | 0.856 ≈ 0.860 | — |
| Plateau inizio | epoch ~35 | epoch ~50 |

Osservazioni:
- `train_loss ≈ val_loss` → underfitting puro, non overfitting
- LR cosine decaduto a ~10% entro epoch 40 → plateau hard senza convergenza
- `theta_conv2_drift=0.97 << theta_conv_drift=3.40` → secondo conv layer sottoutilizzato
- Solo 10k campioni su 60k disponibili → dati insufficienti

---

### T1 — Nuovo output directory

```python
"BASE_DIR": "./fashion_patch_amplitude_qcnn_32_v2_2",  # era v2
```

Preserva i risultati di v2.1 intatti per confronto.

---

### T2 — TRAIN_SAMPLES: 10k → 20k

```python
"TRAIN_SAMPLES": 20000,  # era 10000
```

`train_loss ≈ val_loss ≈ 0.86` indica bottleneck nei dati, non nella capacità del modello.
Con 10k campioni (1000/classe) il modello raggiunge il suo ottimo in ~35 epoch. Raddoppiare i dati
allunga l'orizzonte di apprendimento utile e riduce la varianza delle feature per classe.
`VAL_SAMPLES` e `TEST_SAMPLES` rimangono 2000 ciascuno — disgiunti da `TRAIN_SAMPLES` per costruzione.

**Impatto stimato:** +3-5% test_acc.

---

### T3 — WEIGHT_DECAY_HEAD: 1e-3 → 1e-4

```python
"WEIGHT_DECAY_HEAD": 1e-4,  # era 1e-3
```

La testa ha 330 parametri. Con `train_loss ≈ val_loss` (nessun overfitting), `weight_decay=1e-3`
comprimeva capacità già scarsa. Con 20k campioni, ridurre di 10x è sicuro.

---

### T4 — EPOCHS: 60 → 80, EARLY_STOP_PATIENCE: 10 → 15

```python
"EPOCHS":              80,  # era 60
"EARLY_STOP_PATIENCE": 15,  # era 10
```

Con `T_0=20` (T5) otteniamo 4 cicli warm restart in 80 epoch. `patience=15` evita l'early
stopping subito dopo un restart: il LR sale bruscamente e val_loss può peggiorare temporaneamente
per 1-3 epoch prima di migliorare di nuovo.

---

### T5 — Scheduler: CosineAnnealingLR → CosineAnnealingWarmRestarts

```python
# v2.1
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG["EPOCHS"], eta_min=1e-6
)

# v2.2
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=1, eta_min=1e-6
)
# scheduler.step() invariato — chiamata senza argomenti alla fine di ogni epoch
```

Con `CosineAnnealingLR(T_max=60)` il LR quantistico era al 15% entro epoch 40 → plateau duro.
Con `CosineAnnealingWarmRestarts(T_0=20, T_mult=1)` il LR si resetta ogni 20 epoch:

| Epoch | LR_QKERNEL |
|---|---|
| 1 | 1e-3 (top ciclo 1) |
| 20 | 1e-6 → restart → 1e-3 (top ciclo 2) |
| 40 | 1e-6 → restart → 1e-3 (top ciclo 3) |
| 60 | 1e-6 → restart → 1e-3 (top ciclo 4) |
| 80 | ~1e-6 (fine ciclo 4) |

4 sprint indipendenti permettono al modello di uscire da minimi locali in cicli successivi.

---

### Riepilogo modifiche v2.2

| Parametro | v2.1 | v2.2 | Ragione |
|---|---|---|---|
| `BASE_DIR` | `v2` | `v2_2` | output separato (T1) |
| `TRAIN_SAMPLES` | 10000 | **20000** | underfitting da dati (T2) |
| `WEIGHT_DECAY_HEAD` | 1e-3 | **1e-4** | meno penalità su head (T3) |
| `EPOCHS` | 60 | **80** | 4 cicli WarmRestarts (T4) |
| `EARLY_STOP_PATIENCE` | 10 | **15** | sopravvive restart (T4) |
| Scheduler | `CosineAnnealingLR(T_max=60)` | **`CosineAnnealingWarmRestarts(T_0=20)`** | no plateau (T5) |

Architettura circuito e parametri quantistici invariati.

**Stima run:** ~7-8 ore (1 seed, 2 run, 80 epoch, 20k train).

---

## Architettura circuito: v1 vs v2

```
v1 per patch:
  AmplitudeEmbedding(256 px, 8 qubit)
  -> conv8 [16 param]
  -> pool_8_to_4 [8 param, bug anti-controlled]
  -> readout: 4 x <Z>  =  4 feature

v2 per patch:
  AmplitudeEmbedding(256 px, 8 qubit)
  -> RY(norm_angle, wire=0)        [dati, no param] (C9)
  -> conv8 [16 param]
  -> pool_8_to_4 [8 param, fixed]  (C1)
  -> conv4_retained [8 param, NEW] (C2)
  -> readout: 4x<Z> + 4x<ZZ>  =  8 feature (C3)

Totale feature: 4 patch x 4 = 16  -->  4 patch x 8 = 32
```
