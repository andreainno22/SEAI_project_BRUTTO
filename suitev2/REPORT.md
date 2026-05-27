# Fashion-MNIST Test Suite — Struttura

Test suite parametrica per il confronto **3 ansatz × 2 encoding (+ 2 placeholder)** su Fashion-MNIST 4-classi, basata sul paper Hur et al. 2022 e sul baseline [pulito86.ipynb](../pulito86.ipynb).

## 1. Matrice esperimenti

|ansatz \ encoding   | e1 (9q ancilla)  | e3 (8q amp.)       | dense (placeholder) |
|--------------------|------------------|--------------------|---------------------|
| **hur6** (6 par)   | ✅               | ✅                  | ⏭️ SKIP            |
| **hur8** (10 par)  | ✅               | ✅ (= pulito86)     | ⏭️ SKIP            |
| **hur9** (15 par)  | ✅               | ✅                  | ⏭️ SKIP            |
| **custom4q**       | ⏭️ SKIP         | ⏭️ SKIP            | ⏭️ SKIP            |

- **6 combo eseguibili × 3 seed = 18 run** (seeds = `[42, 43, 44]`).
- **6 combo** vengono saltate con log `SKIPPED: placeholder X` (lasciate per implementazione futura).

## 2. Configurazione dati (allineata al paper)

|              | per classe | totale |
|--------------|-----------:|-------:|
| Train        | 3000       | 12000  |
| Val          |  500       |  2000  |
| Test (base)  |  500       |  2000  |

- 4 classi Fashion-MNIST: `{0: T-shirt, 1: Trouser, 7: Sneaker, 8: Bag}` remappate a `{0,1,2,3}`.
- Immagini riscalate a 16×16 (= 256 pixel = 2⁸ → AmplitudeEmbedding diretto).

## 3. Architettura QCNN

Identica a pulito86 (nessuna testa classica, readout su 2 qubit):

```
Encoding → Conv₁ (wires 0..7) → Pool 8→4 → Conv₂ (wires [0,2,4,6]) → Pool 4→2 → probs(wires=[0,4])
```

Output: 4 probabilità `[P(00), P(01), P(10), P(11)]` mappate sulle 4 classi.

**Loss**: probability cross-entropy `-log(P(y_true))` (come paper Hur).

**Ottimizzatore**: Adam, LR=1e-3, batch_size=24, default 15 epoche.

## 4. Struttura file

```
fashion_suite/
├── __init__.py
├── config.py            # BASELINE, ANSATZ_REGISTRY, ENCODING_REGISTRY, SUITE_SEEDS
├── ansatz.py            # hur_convolution_circuit{6,8,9} + hur_pool_pair + custom_4q stub
├── encodings.py         # encoding_e1 (9q ancilla), encoding_e3 (8q amp), encoding_dense stub
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
| `hur8`  | 10    | RX-RX-RZ-RZ-RX-RX-CNOT-RX-RX-RZ-RZ                 | pulito86 baseline = Fig. 2(h)              |
| `hur9`  | 15    | U3-U3-CNOT-RY-RZ-CNOT-RY-CNOT-U3-U3 (KAK SU(4))    | `U_SU4` nel repo ufficiale = Fig. 2(i)     |

Pooling identico per tutti: `hur_pool_pair` = CRZ(θ₀) + CRX(θ₁) open-controlled, 2 param condivisi per layer.

### 5.2 Encoding ([encodings.py](encodings.py))

| nome    | qubit | param trainabili | input feature                |
|---------|------:|------------------|------------------------------|
| `e3`    | 8     | 0                | 256 pixel (amplitude)        |
| `e1`    | 9     | 16 (`a_embed`+`c_embed`) | 8 quad_means + 4 gA4 globali |

**E1 dettagli** — replica esatta di [versione_1_test/qcnn_builder.py:165-220](../versione_1_test/qcnn_builder.py#L165-L220):
- RY trainabile affine su wires 0–7: `RY(a_i · π·quad_means[i] + c_i)`
- Ancilla wire 8: re-uploading di 4 gammas (= π·tanh(β·gA4)) con sandwich `RY-RZ-RX-RZ-RY(ω_fixed)-RZ-RX-RY-RZ`
- Fusion: 8× `CNOT(8,i) - RZ(λ=π/4) - CNOT(8,i)`
- Dopo l'encoding, wire 8 viene ignorato (trace-out implicito) e la QCNN procede su wires 0–7.

### 5.3 Feature extraction E1 ([data.py](data.py))

- `quad_means_8(img)`: 16×16 → griglia 2×4 → 8 medie di blocchi 8×4 pixel.
- `gA4(img)`: 4 statistiche globali `[mean-0.5, var, grad_energy, H_var-V_var]`.

### 5.4 Model ([model.py](model.py))

`FashionQCNN(ansatz_name, encoding_name)` istanzia dinamicamente:
- `theta_conv1, theta_conv2`: shape `(n_conv,)` (6/10/15 a seconda di ansatz)
- `theta_pool1, theta_pool2`: shape `(2,)` ciascuno
- `a_embed, c_embed`: shape `(8,)` solo per E1
- QNode su `default.qubit` con `wires = 8 (E3) | 9 (E1)`

Numero parametri trainabili: `2·n_conv + 4 + (16 per E1, 0 per E3)`.

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
- ⚠️ **E1 non converge** con `LR=1e-3` uniforme: loss iniziale 4–7 (vs 1.2 di E3), gradient norm 30–100 (vs 0.02 di E3). Cause probabili:
    - Probability cross-entropy unbounded (vs Pauli-Z expvals dell'implementazione originale)
    - Encoding con molti gate sull'ancilla → distribuzioni di probabilità molto piccate all'inizializzazione
  Possibili interventi (non ancora applicati):
    - LR separato più basso per `a_embed/c_embed` (es. 1e-4)
    - Init scale più piccolo per `theta_conv/pool` (es. 1e-3)
    - Warmup degli embed param prima di allenare la quantum kernel
- ⏭️ `custom4q` ansatz e `dense` encoding: placeholder, da implementare in seguito.
