# Strategie per massimizzare le prestazioni di Fashion Patch Amplitude QCNN

Questo documento raccoglie tutte le modifiche strutturali, architetturali e di addestramento analizzate per massimizzare le performance dello script `fashion_patch_amplitude_qcnn_32_frozen_diagnostics.py`. 
Le modifiche sono divise tra quelle **Implementate** (integrate nel codice) e quelle **Rimandate/Documentate** (opzioni valide ma non attivate per specifiche scelte progettuali, come mantenere la rete classica "stupida" per testare il vero potenziale del circuito quantistico).

---

## 🟢 Modifiche Implementate

Le seguenti modifiche sono state testate e **attivamente integrate** nello script Python per stabilizzare l'apprendimento e fornire dati sensati all'Amplitude Embedding.

### 1. Pre-elaborazione: Resize al posto di Zero-Padding
> [!IMPORTANT]
> L'Amplitude Embedding è estremamente sensibile agli input sparsi (molti zeri), in quanto la codifica di uno zero non porta alcuna attivazione utile nel circuito.
- **Problema**: L'iniziale padding da 28x28 a 32x32 con bordi neri causava la presenza di patch (in particolare quelle esterne) piene di zeri. Questo sprecava inefficacemente lo spazio di Hilbert dell'Amplitude Embedding.
- **Implementazione**: Il `transforms.Pad` è stato sostituito con un `transforms.Resize((32, 32))`. L'immagine viene interpolata e occupa tutto lo spazio, riempiendo le 4 patch da 16x16 (256 pixel esatti) con informazioni strutturali significative.

### 2. Inizializzazione "Near-Identity"
> [!TIP]
> L'inizializzazione casuale dei parametri in un circuito variazionale è la causa primaria del fenomeno dei *Barren Plateaus* (Cerezo et al., 2021).
- **Problema**: L'uso di `0.01 * randn` generava angoli casuali. All'aumentare dei layer, questo deviava le attivazioni in pattern caotici e appiattiva i gradienti.
- **Implementazione**: I pesi `theta_conv` e `phi_pool` sono stati inizializzati esattamente a **zero** (`torch.zeros()`). Essendo rotazioni $RX, RY, RZ$, avere parametri a $0$ fa partire il circuito come una pura identità temporale, assicurando un flusso perfetto dei gradienti nelle primissime epoche di training.

### 3. Ottimizzazione: Incremento del Batch Size
- **Problema**: L'aggiornamento dei pesi con un batch size di 32 generava gradienti quantistici estremamente rumorosi (tipico nel QML).
- **Implementazione**: È stato preferito e applicato un innalzamento del `BATCH_SIZE` da 32 a 128 (al posto della Gradient Accumulation) per calcolare gradienti quantistici statisticamente più stabili pur mantenendo il codice e il training loop estremamente pulito e leggibile.

### 4. Regime di Training e Multi-LR
- **Implementazione pre-esistente e mantenuta**:
  - **Multi-LR**: Lo script usava già ottimizzatori separati (`LR_HEAD` a 1e-3, `LR_QKERNEL` a 3e-4) per evitare che gli aggiornamenti massicci della rete classica distruggessero le delicate rotazioni quantistiche. Questa logica è stata mantenuta intatta.
  - **Early Stopping basato su Validation**: È stata confermata la logica di Early Stopping (già presente, `patience=5`) accoppiata al corretto ripristino dei pesi migliori (salvataggio del `best_state`), un must-have per evitare overfitting quantistico.
  - **Fallback per tensori nulli**: Il codice possedeva già una gestione anti-NaN per patch vuote (`if norm < EPS: amp = np.zeros ... amp[0] = 1.0`), che abbiamo analizzato e validato come robusta e vitale.

---

## 🟡 Modifiche Documentate ma NON Implementate (Scelte Progettuali)

Le seguenti modifiche avrebbero senso per la massimizzazione estrema della "Classification Accuracy", ma sono state volutamente scartate per mantenere intatto il nucleo dell'esperimento (ossia dimostrare le capacità estrattive del circuito quantistico e non del post-processing classico o distorcendo il set originale).

### 1. Sostituzione Pooling (pool8) e Misurazioni Locali
- **Idea**: Abbandonare il pooling gerarchico in favore di un pooling alternato (CRZ/CRX) e assicurarsi che le misurazioni finali (le expectation value su PauliZ) avvengano **esclusivamente sui qubit bersaglio** (Local Cost Function).
- **Perché non implementata**: Modificare il circuito e le misurazioni avrebbe alterato pesantemente la struttura interna dell'Ansatz di `fashion_patch`, snaturando la sua architettura originale gerarchica. La variante *qfix* esplora già abbondantemente queste differenze circuitali.

### 2. Aumento Capacità della "Classical Head"
- **Idea**: Sostituire l'attuale linear layer `LayerNorm -> Linear(16, 10)` con un profondo Multi-Layer Perceptron (MLP) come `LayerNorm(16) -> Linear(64) -> GELU -> Dropout -> Linear(32) -> GELU -> Dropout -> Linear(10)`.
- **Perché non implementata**: Come esplicitamente richiesto per motivazioni didattiche/accademiche, la parte non-quantistica (classica) deve rimanere il più "stupida" e limitata possibile. Lo scopo dell'esperimento è provare che *è il circuito quantistico ad apprendere le relazioni* interne all'immagine. Inserire un MLP profondo alla rete classica avrebbe permesso al classificatore di trovare correlazioni non-lineari da solo, mascherando i meriti effettivi del *Quantum Feature Extractor*.

### 3. Data Augmentation Quantistica
- **Idea**: Aggiungere trasformazioni classiche stocastiche prima dell'encoding quantistico (come `RandomHorizontalFlip()` e `RandomRotation(10)`).
- **Perché non implementata**: Rimandata in favore di un testing pulito sull'efficacia dell'architettura stessa con i dati nativi in questa prima fase. Le distorsioni non-lineari esterne all'immagine (come le rotazioni spaziali asimmetriche) su un Amplitude Embedding genererebbero differenze enormi nei vettori di ampiezza, che potrebbero inficiare le rigide misurazioni diagnostiche di separabilità dei cluster di classi scritte nello script originale.
