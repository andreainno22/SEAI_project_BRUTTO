# Giustificazione dell'Architettura (QCNN Ablation Test Suite)

Questo documento spiega le ragioni dietro la struttura della nuova Test Suite e perché abbiamo sostituito i vecchi script esplorativi con un framework unificato.

## Perché una Test Suite Unificata?
Nelle versioni precedenti (come `v1`, `v2`, e la variante `qfix`), avevamo file separati per ogni esperimento. Questo approccio generava ridondanza e rendeva difficile confrontare oggettivamente l'efficacia di due varianti (ad esempio: misurare il contributo reale di `ZZ` senza che altri hyper-parameter sbilanciassero il training). 

Con questa architettura:
1. **Oggettività**: Isoliamo le variabili. Possiamo testare il Custom Encoding (E1) contro l'Amplitude Encoding (E3) a parità assoluta di preprocessing e inizializzazione dei pesi quantistici.
2. **Scalabilità**: Aggiungere un nuovo ansatz (es. topologia `All-to-All`) ora richiede solo l'aggiunta di una funzione in `qcnn_builder.py` e due righe in `test_suite_config.py`, senza dover fare copia-incolla di uno script da 900 righe.
3. **Efficienza**: Lo script `ablation_suite.py` pre-estrae dal dataset tutte le features (sia le *medie dei patch* per il Custom Encoding E1, sia *le ampiezze raw* per l'Amplitude Encoding E3) in un'unica passata tramite cache, permettendo di swappare configurazioni a tempo zero.

## Giustificazione dei Mattoncini Quantistici (Ansatz)

Abbiamo consolidato il codice quantistico mantenendo solo le versioni *differenziabili* e matematicamente corrette dei layer:

- **Pooling CNOT-Sandwich (`hierarchical_pool_8_to_4`)**: Abbiamo abbandonato la versione basata su `CRX`/`CRZ` (presente nei primi script) perché, come riscontrato, PennyLane falliva nel calcolarne il gradiente (rimanevano a zero). Il CNOT-sandwich è completamente differenziabile e permette al livello di pooling di "apprendere".
- **Ansatz `v2_deep` vs `qfix_shallow` vs `flat_no_pool`**: Questi tre design riflettono tre filosofie di network:
  1. **v2_deep**: Aumenta i parametri dopo il pooling (Deep Learning approach), assumendo che i qubit residui abbiano bisogno di rotazioni per condensare il feature space.
  2. **qfix_shallow**: Si fida del pooling quantistico e legge immediatamente i risultati.
  3. **flat_no_pool**: Rimuove del tutto il collo di bottiglia spaziale del pooling, scambiando una potenziale perdita di informazioni con una testa classica più grande (in questo caso legge 8 qubit direttamente).

## Giustificazione del Readout (`Z` vs `Z+ZZ`)
Misurare solo Pauli-$Z$ estrarrà sempre e solo informazioni su singoli qubit (le probabilità marginali). Aggiungere il termine di cross-correlazione $ZZ$ offre alla rete classica informazioni esplicite su quanto due qubit siano interconnessi, sfruttando a pieno l'entanglement generato dai layer convoluzionali. L'idea è testare empiricamente se questa informazione aggiuntiva accelera o migliora la classificazione a fronte dello stesso numero di dati.

## La pulizia dei file legacy
I vecchi file (`Test_8_qubit_qfix.py`, `fashion_qcnn_final.py`, ecc.) contenevano logica harcoded e versioni buggate o obsolete del caricamento dati (es. Resize anziché Padding). Il loro scopo esplorativo si è esaurito: il 100% della loro logica funzionante è stata distillata in questa test suite. Mantenerli avrebbe creato confusione sul "quale script eseguire per avere il risultato corretto". Ora la "Single Source of Truth" per l'intero progetto è `test_suite_config.py`.
