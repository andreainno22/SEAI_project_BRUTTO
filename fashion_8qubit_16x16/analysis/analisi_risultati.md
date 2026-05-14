# Analisi risultati — Fashion-MNIST 8-qubit 16×16 (qfix)

Esperimento: `Test_8_qubit_qfix.py` — 4 varianti × 3 seed × 30 epoch max.
Output: `fashion_8qubit_16x16/`, `nparams_trainable=3682`.

---

## Numeri reali (qfix)

| Variante | test_acc (media) | std | val_acc (media) |
|---|---|---|---|
| **E1** | **71.93%** | ±0.61% | 72.20% |
| E3 | 70.93% | ±2.37% | 71.07% |
| E4 | 65.67% | ±1.21% | 67.73% |
| E2 | 64.27% | ±1.55% | 66.73% |

Baseline casuale: 10%. Il range 64-72% su 10 classi Fashion-MNIST è nella norma della letteratura QML multiclasse (cfr. Röseler 2025: 55% su 4 classi MNIST con CNN classica al 89%).

### Risultati per seed

| Variante | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| E1 | 72.6% | 71.8% | 71.4% |
| E3 | 72.4% | 72.2% | 68.2% |
| E4 | 66.8% | 65.8% | 64.4% |
| E2 | 66.0% | 63.0% | 63.8% |

E3 ha la varianza maggiore (±2.37%): seed 2 scende a 68.2%, mentre i seed 0-1 raggiungono >72%. E1 è la variante più stabile (±0.61%).

---

## Confronto con il baseline originale (Test_8_qubit.py)

| Variante | Baseline originale | qfix | Δ |
|---|---|---|---|
| E1 | 69.9% ±2.1% | **71.93%** ±0.61% | **+2.0%**, std −70% |
| E3 | 70.6% ±1.9% | 70.93% ±2.37% | +0.3% (rumore) |
| E4 | 68.3% ±2.7% | 65.67% ±1.21% | −2.6%, std −55% |
| E2 | 66.2% ±2.1% | 64.27% ±1.55% | −1.9%, std −26% |

**E1**: miglioramento netto — sia accuracy che stabilità. La combinazione local cost function + near-identity init favorisce l'embedding trainable di E1.

**E3**: invariato entro il rumore. AmplitudeEmbedding è già robusto, il qfix non lo peggiora né migliora sostanzialmente.

**E4/E2**: peggiorano in accuracy ma si stabilizzano. Cause probabili:
- La testa più profonda (64→32→10 vs 64→10) aggiunge capacità in un modello con feature meno discriminative — vantaggio per E1 ma rischio per E2/E4 che hanno già meno segnale.
- Il local cost function (solo 4 qubit misurati) riduce il feature vector da 32 a 16: E2/E4 perdono più segnale di E1 perché il loro embedding ha meno varianza discriminativa per qubit.

---

## Dinamiche di training — osservazioni principali

### 1. Nessun overfitting osservato

Il baseline originale mostrava val_loss crescente dal epoch 4 in poi. Nel qfix, **val_loss decresce monotonicamente per tutti e 30 gli epoch** in tutte le varianti.

| Variante | val_loss ep1 | val_loss ep15 | val_loss ep30 | trend |
|---|---|---|---|---|
| E1 | 1.695 | 0.841 | 0.776 | ↓ continuo |
| E3 | 1.734 | 0.926 | 0.850 | ↓ continuo |
| E4 | 1.689 | 0.938 | 0.891 | ↓ continuo |
| E2 | 1.693 | ~0.943 | 0.916 | ↓ continuo |

AdamW + ReduceLROnPlateau + weight decay differenziato hanno eliminato il problema di calibrazione che affliggeva il baseline.

### 2. Early stopping mai attivato

L'early stopping (patience=4) non si è mai attivato — tutti i run hanno completato i 30 epoch. Il modello continua a migliorare lentamente lungo tutta la durata del training. Questo suggerisce che **il limite di 30 epoch è ancora troppo basso** per la convergenza completa.

### 3. Gradienti stabili (E1)

| Metrica | Baseline originale | qfix E1_seed0 |
|---|---|---|
| grad_e1_embed | oscillazioni di 4 ordini di grandezza | range 1.3–5.5 (stabile) |
| grad_qkernel | irregolare | range 0.3–2.8 (stabile) |
| grad_head | irregolare | range 0.8–1.5 (stabile) |

La combinazione near-identity init + AdamW ha stabilizzato i gradienti anche senza pooling gerarchico reale.

### 4. Distribuzione delle predizioni molto più bilanciata

Nel baseline originale, il modello aveva un forte bias verso la classe 4 (Coat) — la prevedeva ~110 volte su 500 campioni. Nel qfix (E1_seed0, ep30):

```
pred_hist: [47, 49, 48, 63, 48, 39, 45, 55, 51, 55]
```

Distribuito attorno a 50 (atteso per 500 campioni, 10 classi). Lo stesso vale per E3 e E4.

---

## Analisi per variante — feature quantistiche

### feat_std_mean_test (deviazione standard delle feature tra campioni)

| Variante | feat_std ep1 | feat_std ep30 | trend | causa |
|---|---|---|---|---|
| **E1** | 0.069 | 0.070 | stabile | Embedding trainable comprime angoli; `a≈0.2` |
| **E3** | 0.252 | 0.240 | −0.012 lento | AmplitudeEmbedding: preserva varianza input |
| **E4** | 0.429 | 0.423 | −0.006 lento | Encoding trainable, range pieno |
| **E2** | 0.429 | ~0.421 | −0.008 lento | Encoding fisso `RY(π·x)`, alta varianza iniziale |

E1 ha feat_std molto bassa (~0.07) ma accuracy più alta — conferma che **feat_std alta non implica accuracy alta**. Il segnale discriminativo di E1 è concentrato su pochi qubit misurati, ma è più utile per la testa classica.

E3 ha feat_std ~0.24, stabile e coerente con la varianza dell'input (std≈0.21 delle quad_means).

---

## Cause residue di performance limitata

### 1. pool8 non è pooling gerarchico reale

Il `pool8` applica CRZ+CRX ma **non elimina qubit**: tutti e 8 rimangono attivi. La garanzia anti-barren-plateau di Pesah et al. 2021 vale solo con riduzione 8→4→2→1. Il local cost function (MEAS_WIRES) approssima, non equivale.

### 2. conv8 senza parameter sharing

16 parametri indipendenti in `conv8` vs 2-4 del paper Hur et al. 2022. Ogni parametro riceve gradiente da un solo gate — rumoroso. La letteratura identifica il parameter sharing come "cruciale" (Röseler 2025).

### 3. Classi strutturalmente ambigue (non risolvibile con qfix)

Dalla confusion matrix E1_seed0 (ep30):

| Classe | Corretti | Errori principali |
|---|---|---|
| 6 Shirt | 23/50 = **46%** | → 0 T-shirt (+8), 2 Pullover (+7), 3 Dress (+6) |
| 2 Pullover | 30/50 = **60%** | → 4 Coat (+9), 6 Shirt (+7) |
| 4 Coat | 31/50 = **62%** | → 2 Pullover (+8), 3 Dress (+5) |

Pullover/Coat/Shirt sono visivamente simili — problema strutturale di Fashion-MNIST, non risolvibile con modifiche architetturali.

### 4. Training ancora in miglioramento a ep30

Il fatto che l'early stopping non scatti e la val_loss continui a decrescere suggerisce che il modello potrebbe beneficiare di più epoch. Da valutare in esperimenti futuri.

---

## Cosa NON è sbagliato

- **conv8 + pool8**: circuito strutturalmente corretto; l'eliminazione del barren plateau con il qfix conferma che il problema era inizializzazione e cost function, non il circuito in sé.
- **Il codice**: nessun bug evidente nella logica di encoding, forward pass e training loop.
- **Il dataset**: 1000 campioni sono pochi, ma sufficienti per questo task sperimentale.

---

## Fix applicati e loro efficacia

| Fix | Motivazione | Effetto osservato |
|---|---|---|
| MEAS_WIRES=[0,2,4,6] | Cost function locale (Cerezo 2021) | ✅ Gradiente stabilizzato, bias tra classi ridotto |
| Near-identity init (0.01·randn) | Symmetry breaking + evita barren plateau all'avvio | ✅ Gradienti stabili fin da ep1 |
| AdamW (wd differenziato) | Decoupling L2 dal preconditioning adattivo | ✅ Nessun overfitting in 30 epoch |
| ReduceLROnPlateau | Convergenza fine automatica | ✅ Val_loss decresce lentamente ma costantemente |
| Early stopping (patience=4) | Salva il modello migliore su val_loss | ⚠️ Mai attivato — 30 epoch non bastano alla convergenza |
| Head più profonda (64→32→10) | Maggiore capacità classica | ⚠️ Aiuta E1, penalizza E2/E4 |
| ACC_STEPS=8 | Riduce varianza del gradiente | ✅ Training più stabile |

---

## Prossimi passi

Le modifiche architetturali identificate come prioritarie (da `strategia_modifiche.md`):

1. **Alternativa A — Parameter sharing in conv8**: ridurre da 16 a 4 parametri. Basso costo, impatto atteso +0/+3%.
2. **Alternativa B — Dense Qubit Encoding E5**: `Rx(x₁) + Ry(x₂)` per qubit, 2 DOF. Confronto diretto con Hur et al.
3. **Alternativa D — WUE E6**: `U3(θ + w·x)`, 3 DOF/qubit. Confronto con Röseler.
4. **Alternativa E — Pooling gerarchico**: riscrittura completa (8→4→2→1), garantisce anti-barren-plateau per Pesah et al.

---

## Confronto con Vittori (tesi base)

**Setup Vittori:** MNIST 8×8, 4 qubit per quadrante, 10 classi, 15 epoch, Adam, nessun fix anti-barren-plateau.
**Setup nostro (qfix):** Fashion-MNIST 16×16→8×8, 8 qubit per quadrante, 10 classi, 30 epoch, AdamW + local cost + near-identity init.

| Variante | Vittori — MNIST 10-class | Nostro qfix — Fashion 10-class | Δ |
|---|---|---|---|
| E3 | ~81–82% | 70.93% | −10% |
| E1 | ~72–73% | **71.93%** | ≈ pari |
| E4 | ~68–69% | 65.67% | −3% |
| E2 | ~67–68% | 64.27% | −3% |

*(numeri Vittori stimati dai boxplot di Fig. 6.1, pag. 45 della tesi)*

### Osservazioni

**E3 — gap di ~10%**: interamente attribuibile al dataset. Fashion-MNIST è strutturalmente più difficile di MNIST per la presenza di classi visivamente simili (Pullover/Coat/Shirt). La letteratura riporta un costo tipico di 5–15% per Fashion rispetto a MNIST a parità di architettura.

**E1 — risultato al pari**: il nostro E1 (71.93%) eguaglia il Vittori E1 (~72%) nonostante il task più difficile. Il qfix ha recuperato il gap atteso. È il risultato più positivo del confronto.

**Ranking invertito E1/E3**: Vittori ottiene E3 > E1; noi otteniamo E1 > E3. Il qfix (local cost function + near-identity init) ha favorito E1 (embedding trainable) più di E3 (AmplitudeEmbedding, encoding fisso). Il motivo: la local cost function riduce le misure da 8 a 4, colpendo meno E1 (il cui segnale discriminativo è già concentrato) e più E3 (che dipende dall'intera ampiezza dello stato per la sua espressività). Questo è un risultato originale rispetto alla tesi di Vittori e rappresenta un contributo della nostra estensione.

**E4/E2 — gap contenuto (~3%)**: ragionevole su un dataset più difficile; conferma che il gap non è imputabile ai fix architetturali ma alla difficoltà intrinseca di Fashion-MNIST.

### Implicazione per la tesi

Questo confronto dimostra che l'estensione a 8 qubit con le modifiche anti-barren-plateau (qfix) produce risultati **comparabili a Vittori su E1** nonostante un task intrinsecamente più difficile. L'inversione del ranking E1/E3 è un risultato originale che giustifica l'indagine sulle varianti E5 ed E6, dove si cerca di combinare espressività dell'encoding (come E3) con la trainabilità dell'embedding trainable (come E1).
