"""
run_suite.py — Entry point: loops over (ansatz, encoding, seed).

Writes one row to results/summary.csv IMMEDIATELY at the end of each run
(not in batch at the end of the suite). If the suite crashes halfway,
summary.csv already contains all completed runs.

Each row of summary.csv has all 16 cells of the test confusion matrix
flattened, so the matrix can be reconstructed offline without re-reading
the per-run test.json files.

Usage examples:
  python -m fashion_suite.run_suite                          # full suite
  python -m fashion_suite.run_suite --combo hur8_e3          # one combo
  python -m fashion_suite.run_suite --ansatz hur8 --encoding e3 --seeds 42
  python -m fashion_suite.run_suite --epochs 1 --train-per-class 50   # smoke
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

from fashion_suite.config import (
    SUITE_SEEDS,
    all_combos,
    combo_is_runnable,
    populate_registries,
)
from fashion_suite.train import train_one_run


SUMMARY_HEADER = (
    ["ansatz", "encoding", "seed", "n_params", "timestamp",
     "n_epochs_trained", "chunk_id", "test_offset", "test_size",
     "train_acc_final", "val_acc_best", "val_acc_final",
     "best_epoch", "best_val_loss",
     "test_loss", "test_acc"]
    + [f"cm_{i}{j}" for i in range(4) for j in range(4)]
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=os.path.join(
        os.path.dirname(__file__), "results"))
    p.add_argument("--ansatz", default=None,
                   help="Run only this ansatz (e.g., hur8). Default: all.")
    p.add_argument("--encoding", default=None,
                   help="Run only this encoding (e.g., e3). Default: all.")
    p.add_argument("--combo", default=None,
                   help="Run only this combo, format '<ansatz>_<encoding>' "
                        "(e.g., hur8_e3). Overrides --ansatz/--encoding.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help=f"Seeds to run. Default: {SUITE_SEEDS}.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override BASELINE.EPOCHS (smoke test).")
    p.add_argument("--train-per-class", type=int, default=None,
                   help="Override BASELINE.TRAIN_PER_CLASS (smoke test).")
    return p.parse_args()


# ============================================================
# CSV helpers
# ============================================================

def ensure_summary_header(summary_path: str):
    if os.path.exists(summary_path):
        return
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        csv.writer(f).writerow(SUMMARY_HEADER)


def append_summary_row(summary_path: str, run_result: dict):
    """Append one row to summary.csv with full confusion-matrix flattened.

    Flush + fsync after writing so a crash in the next run can't lose this one.
    """
    cm = run_result["confusion_matrix"]   # 4x4 list of lists
    flat_cm = [cm[i][j] for i in range(4) for j in range(4)]

    row = [
        run_result["ansatz"], run_result["encoding"], run_result["seed"],
        run_result["n_params"], datetime.utcnow().isoformat(timespec="seconds"),
        run_result["n_epochs_trained"],
        run_result["chunk_id"],
        run_result["test_offset"],
        run_result["test_size"],
        f"{run_result['train_acc_final']:.6f}",
        f"{run_result['val_acc_best']:.6f}",
        f"{run_result['val_acc_final']:.6f}",
        run_result["best_epoch"],
        f"{run_result['best_val_loss']:.6f}",
        f"{run_result['test_loss']:.6f}",
        f"{run_result['test_acc']:.6f}",
    ] + flat_cm

    with open(summary_path, "a", newline="") as f:
        csv.writer(f).writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# Main loop
# ============================================================

def main():
    args = parse_args()
    populate_registries()

    # Pick combos
    combos = all_combos()
    if args.combo:
        a, _, e = args.combo.partition("_")
        if not e:
            raise SystemExit(f"--combo must be '<ansatz>_<encoding>', got {args.combo!r}")
        combos = [(a, e)]
    elif args.ansatz or args.encoding:
        combos = [(a, e) for (a, e) in combos
                  if (args.ansatz is None or a == args.ansatz)
                  and (args.encoding is None or e == args.encoding)]

    seeds = args.seeds if args.seeds is not None else SUITE_SEEDS

    summary_path = os.path.join(args.results_dir, "summary.csv")
    ensure_summary_header(summary_path)

    runnable = sum(1 for (a, e) in combos if combo_is_runnable(a, e)[0])
    total_runs = runnable * len(seeds)
    completed = 0
    skipped = 0
    t_suite_start = time.time()

    print(f"=" * 70)
    print(f"Fashion Suite: {len(combos)} combos, {len(seeds)} seeds each")
    print(f"  -> {runnable} runnable combos * {len(seeds)} seeds = {total_runs} runs")
    print(f"  results -> {args.results_dir}")
    print(f"=" * 70)

    for combo_idx, (ansatz_name, encoding_name) in enumerate(combos, start=1):
        ok, reason = combo_is_runnable(ansatz_name, encoding_name)
        if not ok:
            print(f"[{combo_idx}/{len(combos)}] SKIPPED {ansatz_name}_{encoding_name}: {reason}")
            skipped += 1
            continue

        for seed_idx, seed in enumerate(seeds, start=1):
            run_name = f"{ansatz_name}_{encoding_name}_seed{seed}"
            run_dir = os.path.join(args.results_dir, run_name)

            # Resume: if test.json already exists for this run, skip it
            if os.path.exists(os.path.join(run_dir, "test.json")):
                print(f"[combo {combo_idx}/{len(combos)} seed {seed_idx}/{len(seeds)}] "
                      f"RESUME-SKIP {run_name} (already complete)")
                completed += 1
                continue

            t_run_start = time.time()
            print(f"[combo {combo_idx}/{len(combos)} seed {seed_idx}/{len(seeds)}] "
                  f"START {run_name}")

            try:
                result = train_one_run(
                    ansatz_name, encoding_name, seed, run_dir,
                    epochs=args.epochs,
                    train_per_class=args.train_per_class,
                    log_fn=print,
                )
            except Exception as exc:
                print(f"  !! FAILED: {type(exc).__name__}: {exc}")
                raise

            # 1. Per-run test.json (full metrics for this single run)
            with open(os.path.join(run_dir, "test.json"), "w") as f:
                json.dump(result, f, indent=2)

            # 2. Summary row appended IMMEDIATELY (single source of truth for
            #    post-processing — contains the full 4x4 confusion matrix).
            append_summary_row(summary_path, result)

            completed += 1
            elapsed = time.time() - t_run_start
            print(f"  DONE in {elapsed:.1f}s | "
                  f"test_acc={result['test_acc']:.4f} test_loss={result['test_loss']:.4f}")

            # ETA after first 2 runs
            if completed >= 2 and total_runs > completed:
                avg = (time.time() - t_suite_start) / completed
                remaining = (total_runs - completed) * avg
                eta_min = remaining / 60
                print(f"  ETA: ~{eta_min:.1f} min remaining "
                      f"({completed}/{total_runs} runs done)")

    print(f"=" * 70)
    print(f"SUITE FINISHED: {completed} runs completed, {skipped} placeholder combos skipped")
    print(f"  total time: {(time.time() - t_suite_start) / 60:.1f} min")
    print(f"  summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
