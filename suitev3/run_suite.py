"""
run_suite.py - Entry point: loop over (arm, seed) and write results/summary.csv.

One row is written (flushed + fsynced) immediately after each run, so a crash
halfway leaves all completed runs on disk. Existing runs (test.json present)
are skipped, so the suite is resumable.

Usage:
  python -m suitev3.run_suite                       # full suite (4 arms x 3 seeds)
  python -m suitev3.run_suite --arms trained frozen # only the two quantum arms
  python -m suitev3.run_suite --seeds 42
  python -m suitev3.run_suite --epochs 1 --train-per-class 40   # smoke test
"""

import argparse
import csv
import os
import time
from datetime import datetime

from suitev3 import config as V3
from suitev3.train import train_one_run


SUMMARY_HEADER = (
    ["arm", "seed", "n_params", "timestamp", "n_epochs_trained",
     "best_epoch", "best_val_loss", "val_acc_best", "test_loss", "test_acc"]
    + [f"cm_{i}{j}" for i in range(4) for j in range(4)]
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    p.add_argument("--arms", nargs="+", default=None,
                   help=f"Arms to run. Default: {V3.ARMS}.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help=f"Seeds to run. Default: {V3.SUITE_SEEDS}.")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs (smoke).")
    p.add_argument("--train-per-class", type=int, default=None, help="Override TRAIN_PER_CLASS (smoke).")
    return p.parse_args()


def ensure_header(path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(SUMMARY_HEADER)


def append_row(path, r):
    cm = r["confusion_matrix"]
    flat = [cm[i][j] for i in range(4) for j in range(4)]
    row = [r["arm"], r["seed"], r["n_params"],
           datetime.utcnow().isoformat(timespec="seconds"),
           r["n_epochs_trained"], r["best_epoch"],
           f"{r['best_val_loss']:.6f}", f"{r['val_acc_best']:.6f}",
           f"{r['test_loss']:.6f}", f"{r['test_acc']:.6f}"] + flat
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)
        f.flush()
        os.fsync(f.fileno())


def main():
    args = parse_args()
    arms = args.arms if args.arms is not None else V3.ARMS
    seeds = args.seeds if args.seeds is not None else V3.SUITE_SEEDS

    summary_path = os.path.join(args.results_dir, "summary.csv")
    ensure_header(summary_path)

    total = len(arms) * len(seeds)
    done = 0
    t_start = time.time()
    print("=" * 70)
    print(f"suitev3: {len(arms)} arms x {len(seeds)} seeds = {total} runs")
    print(f"  arms={arms} seeds={seeds}")
    print(f"  results -> {args.results_dir}")
    print("=" * 70)

    for arm in arms:
        for seed in seeds:
            run_name = f"{arm}_seed{seed}"
            run_dir = os.path.join(args.results_dir, run_name)
            if os.path.exists(os.path.join(run_dir, "test.json")):
                print(f"[{done+1}/{total}] RESUME-SKIP {run_name}")
                done += 1
                continue

            print(f"[{done+1}/{total}] START {run_name}")
            t0 = time.time()
            result = train_one_run(arm, seed, run_dir,
                                   epochs=args.epochs,
                                   train_per_class=args.train_per_class)
            append_row(summary_path, result)
            done += 1
            print(f"  DONE in {time.time()-t0:.1f}s | test_acc={result['test_acc']:.4f}")
            if done >= 2 and total > done:
                avg = (time.time() - t_start) / done
                print(f"  ETA ~{(total-done)*avg/60:.1f} min ({done}/{total})")

    print("=" * 70)
    print(f"FINISHED: {done}/{total} runs | {(time.time()-t_start)/60:.1f} min")
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()
