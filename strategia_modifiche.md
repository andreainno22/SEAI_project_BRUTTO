# Strategia di modifiche — Sintesi Röseler 2025 + Hur 2022
*Documento di sintesi per il progetto Fashion-MNIST 8-qubit QCNN*

---

## Convergenza tra i due paper

I due paper, pur differendo nel contesto (Röseler: hardware reale IBM, multiclasse; Hur: simulazione PennyLane, binario), indicano le **stesse tre cause strutturali** di scarsa performance nei QCNN su dati classici:

| Problema strutturale | Röseler 2025 | Hur 2022 | Nostro codice |
|---|---|---|---|
| Pooling non gerarchico | "Gerarchico garantisce anti-plateau (Pesah [7])" | "Trace-out riduce 8→4→2→1" | pool8 non elimina qubit |
| Assenza di parameter sharing | "Cruciale per il design del PQC" | "Translational invariance by construction" | 16 θ indipendenti |
| Encoding insufficientemente espressivo | WUE (U3, 3 DOF/qubit) | Dense (2 DOF/qubit: Rx+Ry) | E2/E4: 1 DOF/qubit; E3: globale |

In tutti e tre i casi il nostro modello è all'estremo peggiore. Il gap con la letteratura non è di iperparametri — è di architettura.

---

## Stato attuale del modello

### Cosa è già stato fatto
- **Cost function locale** (qfix): MEAS_WIRES=[0,2,4,6] — riduce le misure da 8 a 4 per QNode
- **Near-identity init**: `theta_conv = 0.01 * randn` — symmetry breaking + evita barren plateau all'avvio
- **AdamW + ReduceLROnPlateau + Early stopping** — training più stabile e regolarizzato
- **Weight decay differenziato** — protegge i parametri quantistici da L2 eccessivo

### Risultati qfix (3 seed × 30 epoch max, fashion_8qubit_16x16)

| Variante | test_acc | Δ vs baseline originale | Nota principale |
|---|---|---|---|
| E1 | **71.93%** ±0.61% | +2.0%, std −70% | Migliore e più stabile; embedding trainable beneficia del qfix |
| E3 | 70.93% ±2.37% | +0.3% (rumore) | AmplitudeEmbedding robusto, ma alta varianza tra seed |
| E4 | 65.67% ±1.21% | **−2.6%** | Encoding trainable regredisce — testa profonda penalizza |
| E2 | 64.27% ±1.55% | **−1.9%** | Encoding fisso regredisce — meno segnale con 4 qubit misurati |

**Nota critica su E4/E2**: entrambe le varianti peggiorano rispetto al baseline originale. La causa più probabile è la combinazione di (a) riduzione da 32 a 16 feature (local cost function) che colpisce più duramente gli encoding con meno DOF per qubit (Degree of Freedom, quante feature codificate per qubit), e (b) testa più profonda (64→32→10) che aggiunge capacità in assenza di segnale sufficientemente discriminativo. Questo è un avvertimento rilevante per le varianti B (E5, simile a E2) e D (E6, simile a E4).

**Nota su convergenza**: l'early stopping (patience=4) non si è mai attivato — tutti i 30 epoch sono stati completati e il training continuava a migliorare. Il limite di 30 epoch è probabilmente insufficiente per la convergenza completa.

### Bottleneck rimasti

1. **pool8 non riduce i qubit** → nessuna garanzia anti-barren-plateau (Pesah et al.)
2. **conv8 ha 16 parametri indipendenti** → meno regolarizzato, gradiente frammentato
3. **Training sample-by-sample** → gradient variance alta (acc_steps=8 mitiga, non risolve)
4. **Shirt/Pullover/Coat** confuse sistematicamente — problema del task, non del codice
5. **Testa classica più profonda (64→32→10)** → aiuta E1 (segnale ricco), penalizza E2/E4 (segnale sparso); da riconsiderare per le nuove varianti E5/E6
6. **30 epoch non sufficienti** → il modello converge ancora a ep30; testare 50-60 epoch per le varianti successive

---

## Alternative di modifica

Le alternative sono ordinate per **costo di implementazione crescente**, e sono pensate come interventi cumulativi (B include A, C include B, ecc.) o indipendenti a seconda delle risorse disponibili.

---

### Alternativa A — Parameter sharing in conv8 *(basso costo, implementabile in 1 ora)*

**Modifica:** invece di 16 θ indipendenti, conv8 usa parametri condivisi per tipo di gate.

```python
# Attuale: 16 θ indipendenti
for i in range(8): qml.RY(theta[i], wires=q[i])
qml.CNOT([q[0],q[1]]); qml.RZ(theta[8],  wires=q[1])
...

# Proposta: 4 θ condivisi
# θ[0]: tutti gli RY
# θ[1]: tutti i RZ nel layer even (CNOT 0-1, 2-3, 4-5, 6-7)
# θ[2]: tutti i RZ nel layer odd  (CNOT 1-2, 3-4, 5-6, 7-0)
# θ[3]: phi_pool condiviso (CRZ + CRX)
for i in range(8): qml.RY(theta[0], wires=q[i])
qml.CNOT([q[0],q[1]]); qml.RZ(theta[1], wires=q[1])
qml.CNOT([q[2],q[3]]); qml.RZ(theta[1], wires=q[3])
...
```

**Parametri:** da 16+8=24 (conv+pool) a 2+2=4 (2 per conv, 2 per pool) → fattore 6x di riduzione.

**Pro:**
- Implementazione banale: basta indicizzare con `theta[0]` invece di `theta[i]`
- Corrisponde esattamente a ciò che fanno sia Röseler che Hur per construction
- Gradienti più forti (ogni θ riceve contributo da 4-8 gate invece di 1)
- Compatibile con la struttura QNode esistente: nessuna modifica all'architettura

**Contro:**
- Riduce drasticamente l'espressività del circuito — potrebbe sottofit
- L'effetto su Fashion-MNIST 10-class è sconosciuto: in letteratura i risultati sono su classificazione binaria

**Stima di impatto:** +0/+3% di test_acc su E1 (la variante più adatta al parameter sharing); gradiente più stabile (osservabile da `grad_qkernel`). Effetto incerto su E2/E4 dato che già regrediscono nel qfix.

---

### Alternativa B — Dense Qubit Encoding come variante E5 *(basso costo, 2-3 ore)*

**Motivazione:** E2 usa 1 grado di libertà per qubit (RY su 1 asse). Hur et al. mostra che codificare su 2 assi (Rx+Ry) porta a 94.3% su Fashion MNIST binario (best result nel paper). È la versione fissa (non trainable) della WUE di Röseler.

**Struttura dell'encoding E5:**
```python
def embed_E5_dense_8(quad_means_8_A, quad_means_8_B):
    """Dense Qubit Encoding: 2 feature per qubit.
    Usa le 8 meta-patch means (A) sull'asse X e le 8 successivi (B) sull'asse Y.
    Con 4 quadranti e 8 qubit si usano 2*4*8 = 64 feature totali."""
    for i in range(N_QUBITS):
        qml.RX(math.pi * quad_means_8_A[i], wires=i)
        qml.RY(math.pi * quad_means_8_B[i], wires=i)
```

**Dove prendere le feature B:** le meta-patch del quadrante successivo (es. Q1 per il blocco che elabora Q0), oppure una seconda finestra di patch dalla stessa immagine. Alternativa: usare `gA4` (4 statistiche globali) come feature aggiuntive per i primi 4 qubit, e le meta-patch sui restanti 4.

**Pro:**
- Nessun parametro trainable aggiuntivo (encoding fisso)
- Circuit depth invariata
- Risolve il problema di E2 (troppo pochi gradi di libertà) senza la complessità di E3 (AmplitudeEmbedding)
- Confrontabile direttamente con i risultati di Hur et al.

**Contro:**
- Serve scegliere una strategia coerente per le feature B (le seconde 8 per qubit)
- Non testato nel contesto multiclasse
- **Rischio regressione**: E2 (encoding fisso, 1 DOF/qubit) ha perso 1.9% con il qfix. E5 ha 2 DOF/qubit e dovrebbe essere più discriminativo, ma il bottleneck potrebbe essere la testa profonda, non i DOF. Valutare se testare E5 con la testa leggera (Linear(16,64)→GELU→Dropout→Linear(64,10)) per isolare l'effetto dell'encoding.

**Stima di impatto (rivista):** +0/+4% rispetto al qfix di E2 (64.27%). Il gain atteso viene dall'aumento di DOF 1→2; se il bottleneck è altrove, il gain potrebbe essere nullo.

**Come implementare nel codice esistente:**
1. Modificare `get_features()` per restituire anche `quad_means_B` (es. meta-patch dal quadrante adiacente o una finestra shiftata)
2. Aggiungere `qnode_quadrant_E5_8` con la funzione `embed_E5_dense_8`
3. Aggiungere "E5" a `CONFIG["VARIANTS"]`

---

### Alternativa C — Alternativa A + B insieme *(medio, 3-5 ore)*

Combinare parameter sharing (Alt. A) con Dense Qubit Encoding (Alt. B).

Il vantaggio combinato: meno parametri + encoding più espressivo → meno overfitting + rappresentazioni più discriminative. Questa combinazione è la più vicina all'architettura di Hur et al. senza toccare il pooling.

**Stima di impatto (rivista):** +1/+4% rispetto al qfix, con minor varianza tra seed. Il beneficio combinato di meno parametri nel kernel e più DOF nell'encoding dovrebbe compensare il rischio di regressione di E5 da sola.

---

### Alternativa D — Parameter sharing + WUE encoding come E6 *(medio, 5-8 ore)*

**Motivazione:** WUE di Röseler è l'encoding trainable più espressivo nella letteratura recente:
```
U3(θ + w₁·x₁, θ + w₂·x₂, θ + w₃·x₃)
```
Usa tutti e 3 i gradi di libertà del qubit (rotazione arbitraria sulla Bloch sphere). È E4 generalizzato a 3 parametri invece di 1 per qubit.

**Come implementare E6:**
```python
def embed_E6_WUE_8(quad_means_8, theta_base, w1, w2, w3):
    """Weighted Universal Encoding: U3(θ + w1*x, θ + w2*x, θ + w3*x).
    3 parametri per qubit: theta_base (8,), w1 (8,), w2 (8,), w3 (8,) = 32 params."""
    for i in range(N_QUBITS):
        phi   = theta_base[i] + w1[i] * quad_means_8[i]
        theta = theta_base[i] + w2[i] * quad_means_8[i]
        omega = theta_base[i] + w3[i] * quad_means_8[i]
        qml.U3(theta, phi, omega, wires=i)
```
Parametri trainable: `theta_base`(8) + `w1`(8) + `w2`(8) + `w3`(8) = 32 per quadrante.

**Combinato con Alt. A (parameter sharing):** il circuito ha un encoding molto espressivo (32 params per quadrante) e un kernel poco espressivo ma ben regolarizzato (4 params). Bilancia espressività del dato con semplicità del kernel — esattamente la filosofia di Röseler.

**Contro:**
- 32 parametri embedding × 4 quadranti = 128 parametri in più vs E4 (32 param)
- U3 in PennyLane richiede verifica della compatibilità con `lightning.qubit` + adjoint diff
- **Rischio regressione**: E4 (encoding trainable, 1 DOF/qubit) ha perso 2.6% con il qfix. E6 ha 3 DOF/qubit ma lo stesso tipo di encoding trainable — la regressione potrebbe permanere se il bottleneck è la testa profonda, non i DOF. Considerare anche qui la testa leggera nei test iniziali.

**Stima di impatto (rivista):** +1/+5% rispetto al qfix di E4 (65.67%), condizionato a che il bottleneck di E4 fosse davvero i DOF insufficienti. Con parameter sharing (Alt. A) il rischio di overfitting del kernel si riduce, il che potrebbe sbloccare l'encoding.

---

### Alternativa E — Pooling gerarchico reale *(alto costo, 1-2 giorni)*

**Motivazione:** È l'unico intervento che garantisce teoricamente l'assenza di barren plateau (Pesah et al. 2021). Entrambi i paper la indicano come architettura ideale.

**Struttura del nuovo QNode (da Hur et al. Fig. 3):**

```python
def pool_layer_reduce(phi, ctrl_wire, tgt_wire):
    """Riduce ctrl_wire → tgt_wire e traccia fuori ctrl_wire.
    Due rotazioni controllate: CRZ (ctrl=|1>) + CRX (ctrl=|0> = X·CRX·X)."""
    qml.CRZ(phi[0], wires=[ctrl_wire, tgt_wire])
    qml.PauliX(wires=ctrl_wire)
    qml.CRX(phi[1], wires=[ctrl_wire, tgt_wire])
    qml.PauliX(wires=ctrl_wire)
    # dopo questa funzione: misurare/tracciare ctrl_wire fuori

@qml.qnode(dev8, ...)
def qnode_hierarchical_E2(quad_means_8, theta_conv1, theta_conv2, theta_conv3, phi1, phi2, phi3):
    # Layer 1: conv su 8 qubit, poi pool 8→4
    embed_E2_local_8(quad_means_8)
    conv_layer(theta_conv1, wires=range(8))
    # pool: (1→0), (3→2), (5→4), (7→6) — ctrl traced out
    for i in range(4):
        pool_layer_reduce(phi1[2*i:2*i+2], ctrl=2*i+1, tgt=2*i)
    # Layer 2: conv su 4 qubit rimasti [0,2,4,6], poi pool 4→2
    conv_layer(theta_conv2, wires=[0,2,4,6])
    # pool: (2→0), (6→4) — ctrl traced out
    pool_layer_reduce(phi2[0:2], ctrl=2, tgt=0)
    pool_layer_reduce(phi2[2:4], ctrl=6, tgt=4)
    # Layer 3: conv su 2 qubit [0,4], poi pool 2→1
    conv_layer(theta_conv3, wires=[0,4])
    pool_layer_reduce(phi3[0:2], ctrl=4, tgt=0)
    # Misura finale: solo qubit 0
    return qml.expval(qml.PauliZ(0))
```

**Problema tecnico:** PennyLane non supporta il "mid-circuit qubit reuse" direttamente in `lightning.qubit` con adjoint diff — il trace-out fisico richiede `default.qubit` o gestione esplicita della misura intermedia. In pratica, si può **simulare** il trace-out misurando il qubit di controllo e smettendo di usarlo, oppure usando `qml.measure()` con `postselect` su `default.qubit`.

**Alternativa pratica:** mantenere tutti e 8 i qubit ma applicare i gate solo ai qubit "sopravvissuti" a ogni layer, e misurare solo il qubit finale. Questo non è un vero trace-out ma si comporta in modo simile per scopi di training.

**Parametri:** con parameter sharing per ogni layer e 3 conv layers: (4+2+2) pool params + 3 conv layers × 2-4 params = ~20 parametri totali. Molto vicino a Hur et al. (12-51 params).

**Pro:**
- Garanzia teorica anti-barren-plateau (Pesah et al. 2021)
- Profondità O(log n) = 3 layer per 8 qubit → NISQ-compatible
- Architettura pubblicabile come confronto diretto con Hur et al.

**Contro:**
- Riscrittura completa dei QNode
- Incompatibilità con `lightning.qubit` adjoint diff per il trace-out reale (fallback a `default.qubit` = più lento)
- Richiede gestione separata per ciascuna delle 4 varianti di encoding

---

## Raccomandazione strategica

| | A (param sharing) | B (Dense E5) | C (A+B) | D (A+WUE) | E (pool gerarc.) |
|---|---|---|---|---|---|
| Costo implementazione | Basso | Basso | Medio | Medio | Alto |
| Impatto atteso (rivisto) | +0/+3% su E1 | +0/+4% su E2 | +1/+4% | +1/+5% su E4 | +3/+10% |
| Rischio | Basso | **Medio** (E2 regredisce) | Medio | **Medio-alto** (E4 regredisce) | Alto |
| Citabilità nella tesi | Buona | Ottima | Ottima | Eccellente | Eccellente |
| Compatibile con setup corrente | Sì | Sì | Sì | Sì | No (riscrittura) |

Le stime di impatto sono state riviste al ribasso per B e D dopo aver osservato che E2 e E4 peggiorano con il qfix. L'impatto di E (pooling gerarchico) rimane invariato perché è una riscrittura completa indipendente dai risultati attuali.

**Percorso suggerito (aggiornato):**

```
qfix eseguito ✅ (E1=71.93%, E3=70.93%, E4=65.67%, E2=64.27%)
    ↓
Alternativa A (parameter sharing) — verifica gradienti e impatto su E1
    ↓
Alternativa B (Dense E5) — testare prima con testa leggera per isolare l'encoding
    ↓
Alternativa C (A+B) — combinazione se B dà risultati positivi
    ↓
[opzionale] Alternativa D (WUE+A) — confronto con Röseler, testa leggera
    ↓
[solo se tempo disponibile] Alternativa E (pool gerarchico) — teoricamente più forte
```

**Nota sul percorso**: testare B prima con la testa leggera (64→10 invece di 64→32→10) permette di capire se il bottleneck di E2 era la testa o l'encoding. Se E5 con testa leggera supera E2 qfix (64.27%), il gain è attribuibile ai DOF aggiuntivi.

Il percorso A→B→C→D massimizza la **completezza del confronto con la letteratura** senza riscrivere tutto. La tesi può citare:
- Hur et al. come giustificazione per B (Dense Encoding) e A (parameter sharing)
- Röseler et al. come giustificazione per D (WUE) e la cost function locale già implementata
- Pesah et al. come motivazione teorica per E (pooling gerarchico) anche se non implementato

---

## Mappa di dipendenze

```
Qfix ✅ ESEGUITO (E1=71.93%, E3=70.93%, E4=65.67%, E2=64.27%)
├── A: param sharing in conv8+pool8         → modifica conv8(), pool8()
├── B: Dense E5 (testa leggera prima)       → nuova embed_E5_dense_8() + qnode_E5
├── C: A + B                                → combinazione dei due
├── D: A + WUE (E6)                         → nuova embed_E6_WUE_8() + qnode_E6
└── E: pool gerarchico reale                → riscrittura completa QNode
    └── [compatibile con A, B, D come encoding]

Nota: per B e D testare prima con testa leggera (Linear(16,64)→GELU→Dropout→Linear(64,10))
per isolare l'effetto dell'encoding dalla capacità della testa classica.
```

---

*Riferimenti: Röseler et al. arXiv:2505.05957v2 (2025), Hur et al. arXiv:2108.00661v2 (2022), Pesah et al. Phys. Rev. X 11:041011 (2021), Cerezo et al. Nat. Rev. Phys. 3:625 (2021)*
