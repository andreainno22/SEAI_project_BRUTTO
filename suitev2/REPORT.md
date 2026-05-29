# Fashion-MNIST Test Suite - Struttura

Test suite parametrica per il confronto **3 ansatz × 3 encoding (+ 1 placeholder ansatz)** su Fashion-MNIST 4-classi, basata sul paper Hur et al. 2022 e sul baseline [hur8_two_pool_experiment.ipynb](../hur8_two_pool_experiment.ipynb).

## 1. Matrice esperimenti

|ansatz \ encoding   | e1 (9q ancilla)  | e3 (8q amp.)       | custom (8q re-up.)  |
|--------------------|------------------|--------------------|---------------------|
| **hur6** (6 par)   | ✅               | ✅                  | ✅                  |
| **hur8** (10 par)  | ✅               | ✅ (= hur8_two_pool_experiment)     | ✅                  |
| **hur9** (15 par)  | ✅               | ✅                  | ✅                  |
| **custom4q**       | ⏭️ SKIP         | ⏭️ SKIP            | ⏭️ SKIP            |

- **9 combo eseguibili × 3 seed = 27 run** (seeds = `[42, 43, 44]`).
- Le 3 combo `custom4q × *` vengono saltate con log `SKIPPED: placeholder custom4q` (ansatz lasciato per implementazione futura).

## 2. Configurazione dati (allineata al paper)

|              | per classe | totale |
|--------------|-----------:|-------:|
| Train        | 3000       | 12000  |
| Val          |  500       |  2000  |
| Test (base)  |  500       |  2000  |

- 4 classi Fashion-MNIST: `{0: T-shirt, 1: Trouser, 7: Sneaker, 8: Bag}` remappate a `{0,1,2,3}`.
- Immagini riscalate a 16×16 (= 256 pixel = 2⁸ → AmplitudeEmbedding diretto).

## 3. Architettura QCNN

Identica a hur8_two_pool_experiment (nessuna testa classica, readout su 2 qubit):

```
Encoding → Conv₁ (wires 0..7) → Pool 8→4 → Conv₂ (wires [0,2,4,6]) → Pool 4→2 → probs(wires=[0,4])
```

Output: 4 probabilità `[P(00), P(01), P(10), P(11)]` mappate sulle 4 classi.

**Loss**: probability cross-entropy `-log(P(y_true))` (come paper Hur), con `label_smoothing=0.1` applicato **solo in training** (la formulazione segue Szegedy 2016: `(1-ε)·NLL + ε·H_uniform`). La eval/test usa la CE plain così le loss restano confrontabili con la letteratura.

**Ottimizzatore**: `AdamW` con **due gruppi di parametri** (per-group LR + WD):

| gruppo | parametri | LR | WD |
|---|---|---|---|
| `qkernel` | `theta_conv1`, `theta_pool1`, `theta_conv2`, `theta_pool2` | `1e-3` | `1e-5` |
| `embed`   | `a_embed`/`c_embed` (E1) o `theta_enc` (custom)            | `1e-4` | `1e-5` |

`LR_EMBED` deliberatamente più basso: evita che gli affini E1 (o le rotazioni di re-uploading del custom) divergano prima che il kernel quantistico si adatti.

**Scheduler**: `CosineAnnealingWarmRestarts(T_0=epochs//3, T_mult=1, eta_min=1e-5)`. Sul resume di `extend_training` viene ricreato un nuovo schedule con `T_0=extra_epochs//3`.

**Batch size**: 24. **Default epochs**: 15. **Early stopping**: `patience=10`. **Grad clip**: `clip_grad_norm_=1.0`.

**Data augmentation**: disattivata. Train/val/test usano solo resize a 16x16 + `ToTensor()`.

## 4. Struttura file

```
fashion_suite/
├── __init__.py
├── config.py            # BASELINE, ANSATZ_REGISTRY, ENCODING_REGISTRY, SUITE_SEEDS
├── ansatz.py            # hur_convolution_circuit{6,8,9} + hur_pool_pair + custom_4q stub
├── encodings.py         # encoding_e1 (9q ancilla), encoding_e3 (8q amp), encoding_custom (8q re-up.)
├── data.py              # Fashion-MNIST loader + feature extraction (E1) + test extension loader
├── model.py             # FashionQCNN(ansatz_name, encoding_name) parametrico
├── train.py             # loss, run_epoch, train_one_run, extend_training, evaluate_test_chunk
├── run_suite.py         # entry point: cicla (ansatz, encoding, seed), scrive summary.csv
├── aggregate.py         # post-processing: mean ± std + CM per (combo, n_epochs, chunk)
├── extend_run.py        # modalità extend training / extend test
├── REPORT.md            # questo file
└── results/             # output (popolato dopo la run)
```

## 5. Componenti chiave

### 5.1 Ansatz ([ansatz.py](ansatz.py))

| nome    | param | struttura                                          | corrispondenza paper/repo                  |
|---------|------:|----------------------------------------------------|--------------------------------------------|
| `hur6`  | 6     | RY-RY-CNOT-RY-RY-CNOT-RY-RY                        | `U_SO4` nel repo ufficiale = Fig. 2(f)     |
| `hur8`  | 10    | RX-RX-RZ-RZ-RX-RX-CNOT-RX-RX-RZ-RZ                 | hur8_two_pool_experiment baseline = Fig. 2(h)              |
| `hur9`  | 15    | U3-U3-CNOT-RY-RZ-CNOT-RY-CNOT-U3-U3 (KAK SU(4))    | `U_SU4` nel repo ufficiale = Fig. 2(i), variante 9b |

Nota: per `hur9` usiamo la variante 9b del paper: pooling solo trace-out, senza gate parametrizzate.

Pooling parametrico per `hur6`/`hur8`: `hur_pool_pair` = CRZ(theta0) + CRX(theta1) open-controlled, 2 param condivisi per layer.

### 5.2 Encoding ([encodings.py](encodings.py))

| nome    | qubit | param trainabili        | input feature                |
|---------|------:|-------------------------|------------------------------|
| `e3`    | 8     | 0                       | 256 pixel (amplitude)        |
| `e1`    | 9     | 16 (`a_embed`+`c_embed`)| 8 quad_means + 4 gA4 globali |
| `custom`| 8     | 32 (`theta_enc[2,4,4]`) | 4 patch 8×8 → 16 medie 2×2 ciascuna |

**E3 dettagli** - `qml.AmplitudeEmbedding(x_flat, wires=range(8), normalize=True)`, seguito da una singola `RY(norm_angle, wires=0)` parameter-free. `norm_angle = π · ||x||₂ / √n_pixels` è calcolato in `model.forward` (fuori dal QNode per non finire sul tape PennyLane).

**E1 dettagli** - replica esatta di [versione_1_test/qcnn_builder.py:165-220](../versione_1_test/qcnn_builder.py#L165-L220):
- RY trainabile affine su wires 0–7: `RY(a_i · π·quad_means[i] + c_i)`
- Ancilla wire 8: re-uploading di 4 gammas (= π·tanh(β·gA4)) con sandwich `RY-RZ-RX-RZ-RY(ω_fixed)-RZ-RX-RY-RZ`
- Fusion: 8× `CNOT(8,i) - RZ(λ=π/4) - CNOT(8,i)`
- Dopo l'encoding, wire 8 viene ignorato (trace-out implicito) e la QCNN procede su wires 0–7.

**Custom dettagli - Pairwise Fragment Encoding** ([encodings.py:128-176](encodings.py#L128-L176)):

Schema non-standard di **data re-uploading** che alterna feature dei pixel e rotazioni trainable. È strutturato per coppie disgiunte di qubit (4 coppie, 4 patch).

*Pipeline di pre-processing* ([model.py:43-62](model.py#L43-L62)):
1. `images_to_four_patches`: l'immagine 16×16 viene splittata in 4 patch 8×8 (TL, TR, BL, BR).
2. `patch_8x8_to_2x2_means`: ogni patch 8×8 viene compresso in **16 medie di blocchi 2×2** (griglia 4×4). Risultato: `(B, 4, 16)`, scalato per π.

*Mapping patch ↔ coppia di qubit*:

| patch idx | regione  | coppia wire | gruppo theta_enc |
|-----------|----------|-------------|------------------|
| 0         | TL       | (0, 1)      | 0 (top-half)     |
| 1         | TR       | (2, 3)      | 0 (top-half)     |
| 2         | BL       | (4, 5)      | 1 (bottom-half)  |
| 3         | BR       | (6, 7)      | 1 (bottom-half)  |

Le coppie di patch nella stessa metà dell'immagine **condividono lo stesso blocco di parametri trainable** `theta_enc[group]` di shape `(4, 4)`. Totale: `theta_enc` ha shape `(2, 4, 4) = 32` parametri.

*Loop di encoding per ogni coppia di qubit `(a, b)`* - 4 step sequenziali, ognuno usa **una riga del grid 4×4** del patch (4 feature):

```
per s in 0..3:                       # 4 step di re-uploading
  base = 4*s
  x0, x1, x2, x3 = patch[base:base+4]   # 4 medie 2x2 della riga s

  RY(x0, a); RZ(x1, a)                # 2 feature → qubit a
  RY(x2, b); RZ(x3, b)                # 2 feature → qubit b
  CNOT(a, b)
  RY(theta_enc_group[s, 0], a)        # 2 rot trainable
  RY(theta_enc_group[s, 1], b)
  CNOT(b, a)
  RZ(theta_enc_group[s, 2], a)        # 2 rot trainable
  RZ(theta_enc_group[s, 3], b)
```

In 4 step ogni qubit della coppia riceve `4 × 2 = 8` feature dei pixel intervallate da `4 × 2 = 8` rotazioni trainable (ring CNOT-RY × CNOT-RZ).

*Mapping spaziale*: lo step `s` corrisponde alla riga `s` del grid 4×4 di medie 2×2 del patch. Le feature `x0, x1` sono le 2 colonne sinistre della riga; `x2, x3` le 2 colonne destre. Quindi `qubit a` "vede" sempre la **metà sinistra** del patch (su tutte e 4 le righe), `qubit b` la **metà destra**. È una scelta deliberata per dare alle due qubit una decomposizione spaziale coerente.

**Caveat e differenze rispetto al paper Hur**:
- Il paper Hur 2022 confronta encoding **fissi** (amplitude + angle) - l'incremento di accuracy dipende solo dal scelta di ansatz/pool. Il `custom` qui introduce **parametri trainable nell'encoding** (32 in più), un design tipico dei *quantum re-uploading classifier* (Pérez-Salinas 2020) ma non valutato nel paper.
- La condivisione di `theta_enc` **per coppie di patch** (top-half ↔ bottom-half) è arbitraria: ho scelto top/bottom perché Fashion-MNIST ha forte differenza semantica fra parte alta (collo/cuciture) e bassa (suola/cintura) di molti capi. Alternative ragionevoli (full sharing, no sharing, sharing per colonna L/R) non sono testate.
- Il numero di "step" `N_ENCODING_STEPS=4` deriva da `16 feature / 4 feature_per_step = 4 step`. Modificare la compressione del patch (es. 8×8 → 4 medie 4×4 invece di 16 medie 2×2) richiederebbe ri-bilanciare `N_ENCODING_STEPS` e la shape di `theta_enc`.
- Non c'è una *baseline* contro cui confrontare: il `custom` è coperto dalla suite ma **non c'è motivo a priori per aspettarsi che batta `e3`** (che ha 0 param di encoding e usa AmplitudeEmbedding "perfetto"). È un test di ipotesi: aggiungere capacità nell'encoding aiuta o solo aggiunge gradienti rumorosi?

### 5.3 Feature extraction E1 ([data.py](data.py))

- `quad_means_8(img)`: 16×16 → griglia 2×4 → 8 medie di blocchi 8×4 pixel.
- `gA4(img)`: 4 statistiche globali `[mean-0.5, var, grad_energy, H_var-V_var]`.

### 5.4 Model ([model.py](model.py))

`FashionQCNN(ansatz_name, encoding_name)` istanzia dinamicamente:
- `theta_conv1, theta_conv2`: shape `(n_conv,)` (6/10/15 a seconda di ansatz)
- `theta_pool1, theta_pool2`: shape `(2,)` ciascuno per `hur6`/`hur8`; assenti per `hur9` trace-pool
- `a_embed, c_embed`: shape `(8,)` solo per E1
- `theta_enc`: shape `(2, 4, 4)` solo per `custom`
- QNode su `default.qubit` con `diff_method="backprop"`, `wires = 8 (E3/custom) | 9 (E1)`
- Init: `init_scale * randn` (default `0.01`, come hur8_two_pool_experiment)

Numero parametri trainabili (per combo):

| encoding | quantum (conv+pool) | encoding param | totale per ansatz n_conv |
|----------|---------------------|----------------|--------------------------|
| `e3`     | `2·n_conv + pool`   | 0              | hur6:16 · hur8:24 · hur9:30 |
| `e1`     | `2·n_conv + pool`   | 16             | hur6:32 · hur8:40 · hur9:46 |
| `custom` | `2·n_conv + pool`   | 32             | hur6:48 · hur8:56 · hur9:62 |

`pool = 4` per `hur6`/`hur8`; `pool = 0` per `hur9` trace-pool.

### 5.5 Data preprocessing ([data.py:149-210](data.py#L149-L210))

Data augmentation disattivata dopo confronto empirico: non migliorava le metriche sul subset Fashion-MNIST 4-classi. Train/val/test usano lo stesso preprocessing:
```python
Compose([
    Resize((16, 16)),
    ToTensor(),
])
```

`RemapFashionMNIST` restituisce comunque `(x_flat, quad_means_8, gA4, y)` per supportare tutti gli encoding con lo stesso loader.

## 6. Workflow

### 6.1 Suite base

```bash
conda activate seai_env
python -m fashion_suite.run_suite
```

Loop su tutte le 12 combo × 3 seed. Le combo placeholder (custom4q, dense) sono saltate con log esplicito. Le run già completate (`test.json` presente) sono saltate (resume).

**Override per smoke test**:
```bash
python -m fashion_suite.run_suite --combo hur8_e3 --epochs 1 --train-per-class 50 --seeds 42
python -m fashion_suite.run_suite --ansatz hur8 --encoding e3 --seeds 42 43
```

### 6.2 Estendere il training

Se una run sembra avere margine di miglioramento (val_loss ancora in calo all'epoca 15), si possono aggiungere N epoche:

```bash
python -m fashion_suite.extend_run --mode train --epochs 10
# oppure su singole run:
python -m fashion_suite.extend_run --mode train --epochs 10 --runs hur8_e3_seed42,hur9_e3_seed42
```

Cosa fa: carica `last_state.pt` (model + optimizer + epoch count + best_val_loss), continua il training appendendo a `metrics.csv`, aggiorna `best_state.pt` se trova un nuovo best, riesegue il test sul chunk **base** (stessi 500 sample/classe), **aggiunge una nuova riga** a `summary.csv` con lo stesso `chunk_id="base"` ma `n_epochs_trained` più alto.

### 6.3 Estendere il test set

Fashion-MNIST ha 1000 sample/classe nel test set; la suite ne usa 500 per la baseline. I rimanenti 500 (offset 500–1000) possono essere usati come chunk disgiunti.

```bash
python -m fashion_suite.extend_run --mode test --extra-per-class 200
```

Questo:
- Carica `best_state.pt` per ogni run
- Valuta sul nuovo chunk (offset_per_class = 500 alla prima estensione, poi 500+200=700 alla seconda, ecc.)
- Aggiunge una riga a `summary.csv` con `chunk_id="ext_1"` (poi `ext_2`, ...) e `test_offset`, `test_size` aggiornati

**Vincoli**: la modalità test forza l'esecuzione su **tutte le 18 run** (no `--runs`), per garantire confronto equo. Caveat metodologici (vedi sezione 8).

### 6.4 Aggregazione

```bash
python -m fashion_suite.aggregate
```

Legge `summary.csv` e raggruppa per `(ansatz, encoding, n_epochs_trained, chunk_id)`. Produce:
- `results/summary_aggregated.csv`: media ± std del `test_acc/loss` sui 3 seed, più acc per classe.
- `results/confusion_matrices/<ansatz>_<encoding>__<chunk_id>__ep<N>.csv`: matrice 4×4 sommata sui seed.

## 7. Schema output

### 7.1 Per-run (`results/<ansatz>_<encoding>_seed<N>/`)

| file              | contenuto                                                              |
|-------------------|------------------------------------------------------------------------|
| `config.json`     | iperparametri della run (scritto all'inizio)                           |
| `metrics.csv`     | per-epoca: `epoch, train_loss, train_acc, val_loss, val_acc, grad_norm, time_sec` (append + flush) |
| `best_state.pt`   | `model.state_dict()` al best val_loss                                  |
| `last_state.pt`   | `model.state_dict() + optimizer.state_dict() + last_epoch + best_val_loss` |
| `test.json`       | test eval più recente (sovrascritto a ogni training extension)         |

### 7.2 `results/summary.csv` (una riga = un eval event)

```
ansatz, encoding, seed, n_params, timestamp,
n_epochs_trained, chunk_id, test_offset, test_size,
train_acc_final, val_acc_best, val_acc_final,
best_epoch, best_val_loss,
test_loss, test_acc,
cm_00, cm_01, cm_02, cm_03,
cm_10, cm_11, cm_12, cm_13,
cm_20, cm_21, cm_22, cm_23,
cm_30, cm_31, cm_32, cm_33
```

**Tipi di riga**:
| chunk_id | n_epochs_trained | scenario                                            |
|----------|------------------|-----------------------------------------------------|
| `base`   | 15 (default)     | run iniziale                                        |
| `base`   | 25, 40, ...      | dopo `extend_run --mode train --epochs N`           |
| `ext_1`  | depende          | dopo prima `extend_run --mode test`                 |
| `ext_2`  | depende          | dopo seconda estensione test (offset cumulativo)    |

I 16 `cm_ij` permettono di ricostruire la confusion matrix completa a posteriori (`cm_ij` = numero di sample con `true=i, predicted=j`).

### 7.3 `results/summary_aggregated.csv`

Una riga per ogni gruppo `(ansatz, encoding, n_epochs_trained, chunk_id)`, con `test_acc_mean ± test_acc_std` sui seed e accuracy per classe.

## 8. Decisioni di design

- **Resume by default**: `run_suite.py` salta automaticamente le run già completate (presenza di `test.json`). Per rifare una run, cancellare la sua cartella.
- **Scritture incrementali**: ogni riga di `summary.csv` viene scritta con `flush()+fsync()` subito dopo il completamento della run. In caso di crash, tutto ciò che era completato è preservato.
- **Append-only**: nessuna riga di `summary.csv` viene mai sovrascritta. L'audit trail completo di una run è la sua catena di righe ordinate per `timestamp`.
- **Best checkpoint per eval**: tutti i `test_acc` riportati derivano dal modello al best val_loss (non dall'ultima epoca). Garantisce coerenza tra runs di lunghezza diversa.
- **Placeholder espliciti**: `custom4q` e `dense` esistono nei registry come `(None, None)` per essere visibili nella matrice combo, ma sollevano `NotImplementedError` se istanziati. Saltati con log esplicito in `run_suite.py`.

## 9. Caveat metodologici

### 9.1 Estensione training

Decidere di allenare più a lungo solo le config "promettenti" introduce bias. Per mantenere confronti equi, estendere train **per tutte le config** (default) o documentare esplicitamente il criterio di selezione.

### 9.2 Estensione test

Tre regole:

1. **Sempre su tutte le config insieme** (lo script lo forza in modalità `test`). Estendere il test solo su alcune config è p-hacking.
2. **Metriche non confrontabili tra chunk diversi**: `test_acc` su `chunk=base` (500/classe) ≠ `test_acc` su `chunk=ext_1` (es. 200/classe). Sono valutazioni indipendenti su sample diversi. Aggregare con cautela (es. `pooled_acc = (correct_base + correct_ext) / (size_base + size_ext)`).
3. **Decisione di estendere indipendente dai risultati osservati**: se decido "estendo il test" dopo aver visto che ext_1 è più favorevole, sto facendo soft p-hacking. La regola "sempre su tutte le config" mitiga ma non elimina il problema; idealmente il piano di estensione è pre-registrato.

## 10. Stato attuale

- ✅ Pipeline E2E completa e testata (base run → extend train → extend test → aggregate).
- ✅ Tutti gli ansatz (hur6/8/9) girano correttamente con E3.
- ✅ Encoding `custom` (Pairwise Fragment Encoding) implementato (vedi sez. 5.2).
- ✅ Per-group AdamW (LR_EMBED=1e-4 separato da LR_QKERNEL=1e-3) applicato come mitigazione del problema di convergenza E1.
- ✅ Label smoothing `0.1` + scheduler cosine warm restarts attivi.
- ⚠️ **E1 non ancora validato in produzione**: i risultati attuali in `results/summary.csv` sono solo smoke test (1 epoch, 50 sample/classe). Con 1 epoch E1 collassa a predire sempre la classe 0 (confusion matrix `[[500,0,0,0],[500,0,0,0],...]`); va rifatto un run completo per confermare se la per-group LR risolve il problema o se servono ulteriori interventi (init più piccolo di `theta_conv/pool`, warmup degli embed).
- ⏭️ `custom4q` ansatz: placeholder, da implementare in seguito.

## 11. Divergenze rispetto a `suitev1`

Per chi viene da `suitev1` (ablation 8-qubit 10-class amplitude+head), alcuni punti dove le due suite divergono **per design** ma con la stessa nomenclatura - segnalo per evitare confusione:

| componente | suitev1 | suitev2 |
|---|---|---|
| **Target** | Fashion-MNIST 10-class | Fashion-MNIST 4-class `{0,1,7,8}` |
| **Risoluzione** | `Pad(2)` → 32×32 → 4 patch 16×16 | `Resize((16,16))` → 1 immagine |
| **QNode calls / img** | 4 (una per patch) | 1 |
| **Output** | 32 feature (4 patch × 8) → `LayerNorm + Linear(32→10)` | `qml.probs(wires=[0,4])` → 4 prob, no head |
| **Pool gate** | `RY-CNOT-RZ-CNOT` (CNOT-sandwich, fix B1 per adjoint) | `CRZ + (X·CRX·X)` (open-controlled, paper Hur) |
| **Pool param** | 8 (2/pair, no sharing) | 2 (condivisi su tutte le pair del layer) |
| **Readout** | `Z + ZZ ring` (8 feat/patch) | `qml.probs(wires=[0,4])` |
| **diff_method** | `lightning.qubit` + adjoint | `default.qubit` + backprop |
| **gA4[0]** | `X.mean()` (raw, in `[0,1]`) | `mean − 0.5` (centrato, in `[-0.5, 0.5]`) |
| **gA4[2]** | `(dx_o² + dy_o²).mean()` su overlap 31×31 | `(gx².mean() + gy².mean()) / 2` (medie separate) |
| **gA4[3]** | `(V − H)/(V + H)` su `sum(dy²)`/`sum(dx²)` (gradienti) | `(h_var − v_var)/(h_var + v_var)` su varianze riga/colonna |
| **E3 norm injection** | `π · norm_p / Σ norm_q` (relativo fra patch) | `π · ‖x‖₂ / √n_pixels` (assoluto per-immagine) |
| **E1 quad_means** | 8 strisce 2×16 **per patch** (4 patch × 8 = 32) | 8 blocchi 8×4 **per intera immagine** (1 × 8 = 8) |
| **β_global** | `(1, 10, 10, 1)` su gA4 raw (mean in `[0,1]`) | `(1, 10, 10, 1)` su gA4 centrato (mean in `[-0.5, 0.5]`) - gli angoli risultanti coprono range diversi |

Non sono bug, ma se si confrontano metriche o si copia/incolla codice fra le due suite, queste differenze non sono catturate dai nomi delle variabili.
