# Giustificazione dell'Architettura (QCNN Ablation Test Suite)

Questo documento spiega le ragioni dietro la struttura della nuova Test Suite e perché abbiamo sostituito i vecchi script esplorativi con un framework unificato.

## Perché una Test Suite Unificata?
Nelle versioni precedenti (come `v1`, `v2`, e la variante `qfix`), avevamo file separati per ogni esperimento. Questo approccio generava ridondanza e rendeva difficile confrontare oggettivamente l'efficacia di due varianti (ad esempio: misurare il contributo reale di `ZZ` senza che altri hyper-parameter sbilanciassero il training).

Con questa architettura:
1. **Oggettività**: Isoliamo le variabili. Possiamo testare il **Custom Encoding** (etichetta `"E1"`) contro l'**Amplitude Encoding** (etichetta `"amplitude"`) a parità assoluta di preprocessing e inizializzazione dei pesi quantistici.
2. **Scalabilità**: Aggiungere un nuovo ansatz (es. topologia `All-to-All`) ora richiede solo l'aggiunta di una funzione in `qcnn_builder.py` e due righe in `test_suite_config.py`, senza dover fare copia-incolla di uno script da 900 righe.
3. **Efficienza**: Lo script `ablation_suite.py` pre-estrae dal dataset tutte le features (sia le *strip means dei patch* per il Custom Encoding, sia *le ampiezze raw* per l'Amplitude Encoding) in un'unica passata tramite cache, permettendo di swappare configurazioni a tempo zero.

> **Nota sulla nomenclatura.** Le vecchie versioni del codice (`Test_8_qubit*.py`) usavano quattro etichette `E1`/`E2`/`E3`/`E4` per quattro strategie distinte di encoding. Nella test-suite ne sopravvivono solo due: il *custom global-local encoding* (qui rietichettato `"E1"`) e l'*amplitude encoding* (qui `"amplitude"`, che corrispondeva al vecchio `E3`). Le strategie `E2` (angle fisso) e `E4` (angle affine) sono state rimosse perché dominati da E1 nei run di `qfix`.

## Giustificazione dei Mattoncini Quantistici (Ansatz)

Abbiamo consolidato il codice quantistico mantenendo solo le versioni *differenziabili* e matematicamente corrette dei layer:

- **Pooling CNOT-Sandwich (`hierarchical_pool_8_to_4`)**: Abbiamo abbandonato la versione basata su `CRX`/`CRZ` (presente nei primi script) perché, come riscontrato, PennyLane falliva nel calcolarne il gradiente (rimanevano a zero). Il CNOT-sandwich è completamente differenziabile e permette al livello di pooling di "apprendere".
- **Ansatz `v2_deep` vs `qfix_shallow` vs `flat_no_pool`**: Questi tre design riflettono tre filosofie di network:
  1. **v2_deep**: Aumenta i parametri dopo il pooling (Deep Learning approach), assumendo che i qubit residui abbiano bisogno di rotazioni per condensare il feature space.
  2. **qfix_shallow**: Si fida del pooling quantistico e legge immediatamente i risultati.
  3. **flat_no_pool**: Rimuove del tutto il collo di bottiglia spaziale del pooling. Legge tutti 8 i qubit dopo conv8 e confronta direttamente con la baseline (conv8 → pool → conv4_retained → 4 qubit): se la differenza di accuracy è nulla o negativa, il pooling + secondo layer convoluzionale non aggiunge valore.


## Giustificazione del Readout (`Z` vs `Z+ZZ`)
Misurare solo Pauli-$Z$ estrarrà sempre e solo informazioni su singoli qubit (le probabilità marginali). Aggiungere il termine di cross-correlazione $ZZ$ offre alla rete classica informazioni esplicite su quanto due qubit siano interconnessi, sfruttando a pieno l'entanglement generato dai layer convoluzionali. L'idea è testare empiricamente se questa informazione aggiuntiva accelera o migliora la classificazione a fronte dello stesso numero di dati.

## Test eseguiti (strategia OFAT)

Ogni test varia un solo asse rispetto a `baseline_amplitude`. Le righe combinano ENCODING × ANSATZ, le colonne il tipo di readout; `(F)` indica i parametri quantistici congelati.

| ENCODING + ANSATZ | Z | Z\_ZZ |
|---|:---:|:---:|
| **amplitude + v2\_deep** | ✓ `test_readout_Z_only` | ✓ `baseline_amplitude` · ✓`(F)` `test_frozen_qcnn` |
| **amplitude + qfix\_shallow** | — | ✓ `test_ansatz_qfix_shallow` |
| **amplitude + flat\_no\_pool** | — | ✓ `test_ansatz_flat_no_pool` |
| **E1 + v2\_deep** | — | ✓ `test_E1_custom` |
| **E1 + qfix\_shallow** | — | — |
| **E1 + flat\_no\_pool** | — | — |

**6 test totali.** Le celle `—` sono combinazioni valide ma non testate (il prodotto cartesiano completo sarebbe 12 celle × 2 valori di TRAINABLE = 24 run).

## La pulizia dei file legacy
I vecchi script (`Test.py`, `Test_8_qubit.py`, `Test_8_qubit_qfix.py`, `fashion_patch_amplitude_qcnn_32_frozen_diagnostics.py`, `Latest/fashion_patch_amplitude_qcnn_32_v2.py`) contenevano logica hardcoded e versioni buggate o obsolete del caricamento dati (es. `Resize` bilineare anziché `Pad`, pool con `CRX`/`CRZ` non differenziabili, ecc.). Il loro scopo esplorativo si è esaurito una volta consolidate le scelte architetturali validate in `Latest/CHANGES_v2.md` (v2.2): la test-suite distilla quelle scelte in un unico framework parametrico e abbandona le ramificazioni minoritarie (encoding E2/E4, head profonda con dropout, training sample-by-sample). Mantenere gli script vecchi in parallelo creerebbe confusione sul "quale script eseguire per avere il risultato corretto". Ora la *single source of truth* per l'intero progetto è `test_suite_config.py`, costruita sopra la baseline di v2.2.
