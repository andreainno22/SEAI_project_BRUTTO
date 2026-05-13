# Analisi risultati — Fashion-MNIST 8-qubit 16×16

## Numeri reali

| Variante | Test acc (media) | Std |
|---|---|---|
| E1 | 69.9% | ±2.1% |
| E2 | 66.2% | ±2.1% |
| E3 | **70.6%** | ±1.9% |
| E4 | 68.3% | ±2.7% |

Il baseline casuale è 10%. Per QML su Fashion-MNIST 10 classi con 1000 campioni è nella norma della letteratura (~65-75%). Il confronto con CNN classiche (~93%) è però quello che rende questi numeri "bassi".

---

## Causa 1 — Classi strutturalmente ambigue (problema principale)

Dalla confusion matrix E1 seed 0:

| Classe vera | Corretti | Confusione dominante |
|---|---|---|
| 2 Pullover | 21/50 = **42%** | → 4 Coat (26 campioni!) |
| 3 Dress | 22/50 = **44%** | → 4 Coat (13 campioni) |
| 6 Shirt | 13/50 = **26%** | → 0 T-shirt (11) + 4 Coat (18) |

Il modello predice la classe 4 (Coat) **110 volte** invece di ~50 (`pred_hist: [55,52,35,32,110,...]`). Il circuito ha sviluppato un **bias sistematico verso Coat** che assorbe tutti i capi simili (top). Questo non è un bug — Pullover/Coat/Shirt sono visivamente molto simili anche per CNN classiche (Fashion-MNIST accuracy per-class: Shirt classicamente ~70%).

---

## Causa 2 — Barren plateau: `feat_std_mean_test` piatto

### Cos'è `feat_std_mean_test`

È costruito in tre passaggi nel codice:

**Passo 1** — per ogni campione del test set, il `forward()` restituisce `feat_vec`: il vettore di 32 valori di aspettazione ⟨Z⟩ prodotti dai 4 QNode (4 quadranti × 8 qubit ciascuno), **prima** della testa classica. Ogni componente ∈ [-1, +1].

**Passo 2** — `eval_subset` accumula tutti i 500 `feat_vec` del test set e li impila in una matrice `F` di forma **(500, 32)**. Poi calcola `feat_std = F.std(axis=0)`, che è un vettore di 32 numeri: la **deviazione standard di ciascuna delle 32 feature attraverso i 500 campioni**.

**Passo 3** — `feat_std_mean_test = mean(feat_std)` riduce quel vettore a un singolo scalare: la **deviazione standard media delle feature quantistiche sul test set**.

### Come interpretarlo

Ragionando sul range possibile:

- Se il circuito produce output completamente casuali/uniformi in [-1, +1] → std ≈ **0.577**
- Se il circuito produce lo stesso output per ogni campione → std ≈ **0.0**
- Se il circuito discrimina bene le classi → std dovrebbe essere **alta** (campioni diversi → output diversi) e **crescere** durante il training

Il valore osservato è **0.062** — e resta praticamente costante per tutti i 15 epoch:

```
Epoch  1: feat_std = 0.06112
Epoch  5: feat_std = 0.06131
Epoch 10: feat_std = 0.06262
Epoch 15: feat_std = 0.06205
```

Questo vuol dire che i 32 ⟨Z⟩ del circuito variano pochissimo tra un campione e l'altro. Se si immagina la matrice F (500, 32), ogni colonna è quasi piatta — i 500 campioni producono valori di aspettazione quasi identici, indipendentemente dalla loro classe.

### Cosa implica per il modello

La testa classica riceve 500 vettori di 32 numeri che sono quasi uguali tra loro — ha pochissimo segnale su cui appoggiarsi per distinguere le 10 classi. Qualsiasi classificazione avviene **nonostante** le feature quantistiche, non grazie ad esse.

Il fatto che il valore **non cresca durante il training** è il segnale diagnostico chiave: il circuito quantistico non sta imparando a separare i campioni nello spazio delle feature. Questo è il comportamento tipico di un **barren plateau**: i parametri del circuito si muovono in una zona del landscape dove i gradienti sono quasi zero e l'output è quasi insensibile ai parametri — quindi la distribuzione delle feature non cambia anche se i parametri cambiano.

In sintesi: `feat_std_mean_test ≈ 0.062 costante` = il layer quantistico è essenzialmente un encoder fisso e non-informativo, e l'accuracy del 70% è dovuta quasi interamente alla testa classica lineare che riesce a estrarre qualcosa anche da feature debolissime.

### Verifica sperimentale: barren plateau confermato, non dati

Per escludere che la `feat_std` bassa fosse semplicemente dovuta a input poco variegati, è stata analizzata la distribuzione dei valori di ingresso al circuito su 500 campioni del test set (`check_input_variance.py`):

```
std input quad_means (pre-circuito):  0.2073
feat_std_mean_test   (post-circuito): 0.0620
```

Il circuito comprime la varianza di un fattore **~3.3x**. Gli angoli RY(π·x) effettivi hanno std di 49.5° e coprono quasi l'intero range [0°, 178°] — i qubit vengono inizializzati in stati molto diversi tra campione e campione. Distribuzione dei valori di input (decili):

```
min: 0.000 | 25%: 0.004 | 50%: 0.227 | 75%: 0.502 | 90%: 0.709 | max: 0.989
```

I dati di input sono ricchi di varianza — è il circuito a distruggerla. `conv8 + pool8` applicano un circuito sufficientemente espressivo da avvicinarsi a un 2-design globale, che mappa qualsiasi distribuzione di input verso una distribuzione di output quasi uniforme. **Il barren plateau è la causa, non i dati.**

---

## Causa 3 — Calibrazione degradata (non overfitting di accuracy)

Andamento E1 seed 0:

| Epoch | train_acc | val_acc | val_loss |
|---|---|---|---|
| 1 | 63% | 61.6% | 1.006 |
| 4 | 77% | 74.0% | **0.810** ← minimo |
| 10 | 75% | 70.4% | 0.894 |
| 15 | 79% | 72.6% | **1.307** |

La val_loss aumenta dal epoch 4 in poi anche se val_acc cresce lentamente. Il modello diventa progressivamente **overconfident ma meno calibrato** — senza early stopping si salva il modello peggiore in termini di loss. Il potenziale val_acc ottimale (~74%) viene perso.

---

## Causa 4 — Gradienti instabili

```
grad_e1_embed per epoch: 56.5 → 0.007 → 23.5 → 14.9 → 25.6 → 2.0 → 0.57 → 33.5 → 118.8 → 6.5 → ...
```

Oscillazione di **4 ordini di grandezza**. Nonostante il gradient clipping (0.5), il landscape di loss è molto irregolare. Questo è coerente con il barren plateau: gradienti molto piccoli nella zona piatta, poi grandi quando per caso si esce.

---

## Causa 5 — Pochi campioni, task difficile

- 1000 campioni totali = **100 per classe** per 10 classi
- Il QNode valuta un campione alla volta → il modello vede poca varietà per epoch
- Fashion-MNIST è intrinsecamente più duro di MNIST (texture simili tra capi)

---

## Cosa NON è sbagliato

- **Il circuito `conv8 + pool8`** è costruito correttamente
- **Il codice** non ha bug evidenti — la logica di encoding, forward pass e training loop è consistente con la versione 4-qubit
- **Il dataset** è adeguato, è il problema (10 classi con sovrapposizioni) ad essere più difficile

---

## Interventi più efficaci per migliorare

1. **Early stopping** su val_loss — si guadagnerebbero ~2-3% di test acc salvando il modello all'epoch ottimale (~4-7)
2. **Più campioni di training** — almeno 2000-3000 se il tempo lo consente
3. **Ridurre a 5-7 classi** escludendo le coppie ambigue (es. rimuovere classe 6 Shirt o 2 Pullover) per avere un confronto più pulito con il setup 4-qubit originale
4. **Cost function locale** — misurare solo 1-2 qubit di output invece di tutti e 8 (`[qml.expval(qml.PauliZ(i)) for i in range(8)]`). I gradienti di cost locali scalano polinomialmente, quelli globali esponenzialmente: è l'intervento più efficace contro il barren plateau.
5. **Inizializzazione near-identity** — inizializzare `theta_conv` e `phi_pool` a zero o quasi-zero invece che con rumore casuale. Il circuito parte vicino all'identità dove i gradienti sono non-nulli (*identity block initialization*).
6. **Ridurre la profondità di `conv8`** — rimuovere uno dei due layer di entanglement (even o odd) riduce l'espressività globale, che è la causa del plateau, non la soluzione.

---

## Risoluzione Barren Plateau

Applicati i fix anti-barren-plateau nel file `Test_8_qubit_qfix.py` (confrontabile con `Test_8_qubit.py` senza sovrascrivere i risultati originali in `fashion_8qubit_16x16/`). I risultati del nuovo esperimento vengono salvati in `fashion_8qubit_qfix/`.

### Fix 1 — Cost function locale

**Modifica:** i QNode restituiscono `expval` solo sui qubit `MEAS_WIRES = [0, 2, 4, 6]` invece di tutti e 8.

```python
# Prima (globale)
return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Dopo (locale)
MEAS_WIRES = list(range(0, N_QUBITS, 2))  # [0, 2, 4, 6]
return [qml.expval(qml.PauliZ(i)) for i in MEAS_WIRES]
```

I qubit [0,2,4,6] sono i target dei `CRZ` in `pool8` — raccolgono informazione dai qubit adiacenti prima della misura. Misurare solo loro riduce la dipendenza della cost function da tutti i qubit simultaneamente, avvicinandosi a una cost locale nel senso di Cerezo et al. 2021.

**Effetto sul modello:** il feature vector si riduce da 4×8=**32** a 4×4=**16**, e la testa classica diventa `Linear(16, 64)` invece di `Linear(32, 64)`. Il numero di parametri totali scende da ~2978 a ~1922.

### Fix 2 — Near-identity initialization

**Modifica:** `theta_conv` inizializzato a zero invece che con rumore casuale.

```python
# Prima
self.theta_conv = torch.nn.Parameter(0.01 * torch.randn(16, dtype=torch.float32))

# Dopo
self.theta_conv = torch.nn.Parameter(torch.zeros(16, dtype=torch.float32))
```

Con `theta_conv = 0` e `phi_pool = 0` (già zero nell'originale), il circuito parte vicino all'identità: tutti i gate `RY`, `RZ`, `CRZ`, `CRX` applicano rotazioni nulle, quindi lo stato iniziale è determinato quasi interamente dall'embedding. I gradienti in questo punto del landscape sono non-nulli per costruzione, evitando la zona piatta del barren plateau all'avvio del training.

### Cosa rimane invariato

- Struttura del circuito (`conv8`, `pool8`, tutti gli encoding)
- Iperparametri di training (lr, clip, acc_steps, epoche, dataset)
- Varianti E1/E2/E3/E4 e seeds

Questo garantisce che le differenze nei risultati siano attribuibili esclusivamente ai due fix.
