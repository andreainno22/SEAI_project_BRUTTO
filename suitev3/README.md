# suitev3 — Does the quantum part matter? (hybrid QCNN ablation)

A single focused test on **Fashion-MNIST, 4 classes** (T-shirt, Trouser, Sneaker, Bag),
inheriting **all training hyperparameters from `suitev2`** (AdamW lr=1e-3, weight
decay=0, 20 epochs, batch 24, label smoothing 0.1, cosine warm-restarts, early
stop patience 10, best-by-val-loss). The only goal: measure how much the
**quantum circuit** contributes in a hybrid (circuit + classical head) design.

## Backbone (shared by the quantum arms)

amplitude encoding (**no norm injection**) → 1 `flat_no_pool` conv layer
(`hur8`, applied once on all 8 wires, **no pooling**) → 8 single-qubit ⟨Z⟩
readout → classical head `LayerNorm(8) + Linear(8, 4)`.

## Arms (5) × seeds (3: 42, 43, 44), paired on the data split

A **2×2 of the 256→8 feature extractor** (quantum vs classical, trainable vs
fixed), all feeding the **same** `LayerNorm(8) + Linear(8, 4)` head, plus a
logistic reference:

|                  | quantum extractor            | classical extractor                     |
|------------------|------------------------------|-----------------------------------------|
| **trainable**    | `trained` (circuit, 10 params) | `mlp8` (`Linear(256,8)`+tanh, 2056 params) |
| **fixed random** | `frozen` (`U(0,2π)`)         | `randfeat` (rand proj + cos)            |

Plus `logistic` = `Linear(256,4)` on raw pixels (no bottleneck), a trivial reference.

Every arm shares the same 8-feature bottleneck and the same head, so the
**extractor is the only thing that varies**:
- **`trained` vs `mlp8`** → does a *quantum* trainable extractor beat a *classical*
  one at the same bottleneck? (the decisive control)
- **`frozen` vs `randfeat`** → same question for *fixed random* extractors.
- **`trained` vs `frozen`** → value of training the circuit at all.

The quantum extractor uses far fewer params (10 vs 2056); the comparison is
iso-**architecture** (same bottleneck + head), not iso-parameter.

## Run (on the VM)

Run **from the repo root** (the directory that contains both `suitev2/` and
`suitev3/`). Using `-m` is mandatory — `python suitev3/run_suite.py` breaks
the package imports.

### Quick single run (sequential, ~1.5–2 h)

```bash
conda activate seai_env
export CUDA_VISIBLE_DEVICES=""          # CPU only — ignore the GPU
python -m suitev3.run_suite > suitev3_run.log 2>&1 &
tail -f suitev3_run.log                 # watch progress (Ctrl-C only stops tail)
```

### Recommended: parallel by seed (~30 min wall-clock with many cores)

A single `trained` run is inherently sequential (epochs in series), so extra
cores help only if you parallelise **across seeds**. Three processes run the
three seeds simultaneously; wall-clock drops from ~90 min to ~30 min.

`OMP_NUM_THREADS` prevents the three processes from fighting over cores. Set
it to `floor(n_cores / 3)`; past ~8 threads per process the gain saturates
at 8-qubit circuit size.

```bash
conda activate seai_env
export CUDA_VISIBLE_DEVICES=""          # CPU only
export OMP_NUM_THREADS=4               # tune to floor(n_cores / 3)
export MKL_NUM_THREADS=4

for s in 42 43 44; do
  nohup python -m suitev3.run_suite \
    --seeds $s \
    --results-dir suitev3/results_s$s \
    > suitev3_s$s.log 2>&1 &
done

# watch one of them
tail -f suitev3_s42.log
```

When **all three** processes have finished, merge and aggregate:

```bash
mkdir -p suitev3/results
head -n1 suitev3/results_s42/summary.csv > suitev3/results/summary.csv
tail -q -n+2 suitev3/results_s*/summary.csv >> suitev3/results/summary.csv

python -m suitev3.aggregate
python -m suitev3.cost
```

- Fashion-MNIST auto-downloads to `./data` on first run (internet required).
- Each seed folder is **resumable**: if a process dies, re-run the same
  command for that seed and it skips completed runs (those with `test.json`).
- Optional smoke test (a few seconds, verifies the pipeline end-to-end):
  ```bash
  python -m suitev3.run_suite --epochs 1 --train-per-class 40 \
    --results-dir suitev3/_smoke && rm -rf suitev3/_smoke
  ```

## Expected wall-clock (single CPU)

| Arm        | per seed | note |
|------------|----------|------|
| `trained`  | ~30 min  | bottleneck (quantum fwd+bwd every step) |
| `frozen`   | ~1 min   | **cached features** — the circuit is fixed, so we run the quantum forward once and train only the head |
| `mlp8`     | <1 min   | pure classical |
| `randfeat` | <1 min   | pure classical |
| `logistic` | <1 min   | pure classical |

Total ≈ **1.5–2 h**, dominated by the three `trained` runs. A GPU helps only the
quantum arms marginally (8-qubit `default.qubit` is CPU-bound).

## Fair cost comparison (the encoding is part of the model cost)

Parameter count is not comparable across paradigms. In the **kernel-methods view**
(Schuld 2021, *"Supervised quantum ML models are kernel methods"*; Havlíček et al.,
Nature 2019) the **data-encoding feature map IS the model**, so its circuit must be
counted. `python -m suitev3.cost` decomposes the circuit to `{RX,RY,RZ,CNOT}` and
reports a Clifford+T T-count estimate (Ross & Selinger 2016; real numbers from the
actual circuit):

| model | trainable params | two-qubit gates | rotations | T-count (ε=1e-3) | classical FLOPs |
|-------|:---:|:---:|:---:|:---:|:---:|
| quantum `trained` | 62 | **261** | 325 | **~9.7k** | — |
| — of which **encoding** | 0 | **254 (97%)** | 255 | ~7.6k | — |
| — of which ansatz (conv) | 10 | 7 | 70 | — | — |
| `mlp8` | 2108 | — | — | — | **~4.2k** |
| `logistic` | 1028 | — | — | — | ~2.0k |

**Conclusion:** once the encoding is counted, the quantum model costs ~261
two-qubit gates and ~10⁴ T-gates per inference — vs ~10³ FLOPs for the classical
twin — and **97% of that two-qubit cost is the fixed amplitude-encoding state
prep** (`2ⁿ−2 = 254` CNOTs, the Möttönen construction). The "10 trainable params"
figure is **not** a fair cost proxy; counting the encoding, the quantum model is
*more* expensive, not less — orders of magnitude heavier on the axis that matters
for quantum hardware, for no accuracy gain.

### Two cost regimes: NISQ (today) vs fault-tolerant (future)

The T-count above is the **fault-tolerant** (error-corrected) cost — the regime of
*future* hardware, where Clifford gates (including `CNOT`) are cheap and every
arbitrary rotation must be synthesized from `T` gates via magic-state distillation,
so **rotations dominate** (255 RY → ~10⁴ `T`).

On **today's NISQ** hardware the cost inverts: `RZ` is a virtual (free) frame
change, so the bottleneck is **two-qubit gates + SWAP overhead** from limited qubit
connectivity. That regime is measured by `transpile_comparison.ipynb`, which
transpiles the circuits to real IBM backend models (`FakeSherbrooke`, `FakeTorino`)
at optimization level 3.

| | NISQ (today) | Fault-tolerant (future) |
|---|---|---|
| cheap | rotations (virtual `RZ`) | Clifford (incl. `CNOT`) |
| **expensive** | **two-qubit gates + SWAP** (254 CNOT + routing) | **`T` gates ← rotations** (255 RY → ~10⁴ `T`) |

The amplitude encoding is prohibitive in **both** eras, for opposite reasons — so
its cost is structural, not an artifact of one device generation. `cost.py` gives
the fault-tolerant figure; `transpile_comparison.ipynb` gives the NISQ figure.

## Reading the result

The 2×2 makes the extractor the only variable:
- **`trained` > `mlp8`** → a quantum trainable extractor beats a classical one at
  the same bottleneck → a genuine, defensible quantum contribution.
- **`trained` ≈ or < `mlp8`** → the quantum circuit adds nothing a tiny classical
  `Linear(256,8)` can't (the likely outcome — an early smoke had `mlp8` ≈ 94%).
- **`frozen` ≈ `randfeat`** → quantum random features ≈ classical random features
  (no structural quantum advantage).
- **`trained` ≫ `frozen`** → training the circuit does real work; **≈** → the head does it.
- **`logistic`** brackets what a trivial linear classifier already achieves.

The frozen confusion matrix in early epochs shows *structured* (not random)
misclassification: the quantum features separate the classes; the head just
needs to learn the alignment. A barely-trained head can therefore score *below*
chance — this is under-training, not a bug.

## References

References the claims above rest on:

- **Schuld (2021)** — *Supervised quantum machine learning models are kernel methods.*
  arXiv:2101.11020. — the encoding feature map *is* the model (justifies counting
  the encoding as part of the cost).
- **Havlíček et al. (2019)** — *Supervised learning with quantum-enhanced feature
  spaces.* Nature 567, 209–212. doi:10.1038/s41586-019-0980-2. — quantum
  feature-map / kernel view.
- **Möttönen, Vartiainen, Bergholm & Salomaa (2005)** — *Transformation of quantum
  states using uniformly controlled rotations.* Quantum Inf. Comput. 5(6), 467–473
  (arXiv:quant-ph/0407010). — the amplitude-encoding state preparation used by
  PennyLane's `MottonenStatePreparation` (the `2ⁿ−2` CNOT count).
- **Ross & Selinger (2016)** — *Optimal ancilla-free Clifford+T approximation of
  z-rotations.* Quantum Inf. Comput. 16(11–12), 901–953. — the T-count-per-rotation
  estimate used by `cost.py`.
- **Rahimi & Recht (2007)** — *Random features for large-scale kernel machines.*
  NeurIPS 20. — the `randfeat` classical random-feature baseline.
- **Hur, Kim & Park (2022)** — *Quantum convolutional neural network for classical
  data classification.* Quantum Mach. Intell. 4(1). doi:10.1007/s42484-021-00061-x.
  — the `hur8` ansatz and the head-free QCNN lineage.

Capacity metrics discussed as *alternatives* — they measure trainable-parameter
capacity and therefore **miss** the fixed-encoding cost, so they are not used as
the primary fairness metric here:

- **Abbas et al. (2021)** — *The power of quantum neural networks.* Nature Comput.
  Sci. 1, 403–409. doi:10.1038/s43588-021-00084-1. — effective dimension via the
  Fisher information.
- **Haug, Bharti & Kim (2021)** — *Capacity and quantum geometry of parametrized
  quantum circuits.* PRX Quantum 2, 040309. — effective dimension via the quantum
  Fisher metric.
- **Caro et al. (2022)** — *Generalization in quantum machine learning from few
  training data.* Nat. Commun. 13, 4919. doi:10.1038/s41467-022-32550-3. —
  generalization bounds scaling with the number of trainable gates.
- **Schuld, Sweke & Meyer (2021)** — *Effect of data encoding on the expressive
  power of variational quantum-machine-learning models.* Phys. Rev. A 103, 032430.
  — the QNN-as-truncated-Fourier-series expressivity view.
