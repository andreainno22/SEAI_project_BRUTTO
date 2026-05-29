"""
extend_run.py - Two extension modes for a finished suite:

  1) TRAIN EXTENSION (--mode train --epochs N)
     For each existing run dir (matching the filters), load last_state.pt,
     train N more epochs, save updated checkpoints, re-eval on the BASE test
     chunk, and append a new row to summary.csv with the same chunk_id='base'
     but a higher n_epochs_trained.

  2) TEST EXTENSION (--mode test --extra-per-class N)
     For ALL existing runs (we enforce "all configs together" to keep the
     comparison fair), load best_state.pt, evaluate on a NEW disjoint chunk
     of `N` per-class test samples picked from the previously-unused tail of
     Fashion-MNIST's test set, and append new rows to summary.csv with a fresh
     chunk_id like 'ext_1', 'ext_2', etc.

Usage:
  python -m suitev2.extend_run --mode train --epochs 10
  python -m suitev2.extend_run --mode test  --extra-per-class 200
  python -m suitev2.extend_run --mode train --epochs 5 --runs hur8_e3_seed42
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

from suitev2.config import BASELINE
from suitev2.train import extend_training, evaluate_test_chunk_at_offset
from suitev2.run_suite import (
    SUMMARY_HEADER, ensure_summary_header, append_summary_row,
)


RESULTS_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "results")


# ============================================================
# Discovery helpers
# ============================================================

def discover_run_dirs(results_dir):
    """List all subdirs that look like a finished run (have config.json + last_state.pt)."""
    runs = []
    for name in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, name)
        if (os.path.isdir(d)
                and os.path.exists(os.path.join(d, "config.json"))
                and os.path.exists(os.path.join(d, "last_state.pt"))):
            runs.append(d)
    return runs


def filter_runs(runs, names=None):
    if not names:
        return runs
    name_set = set(names)
    out = []
    for d in runs:
        if os.path.basename(d) in name_set:
            out.append(d)
    missing = name_set - {os.path.basename(d) for d in runs}
    if missing:
        sys.exit(f"runs not found: {sorted(missing)}")
    return out


# ============================================================
# Summary helpers
# ============================================================

def read_summary_rows(summary_path):
    if not os.path.exists(summary_path):
        return []
    with open(summary_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def next_chunk_id_and_offset(summary_rows, ansatz, encoding, seed):
    """Compute the next chunk_id ('ext_K') and the test_offset for it, given
    the rows already in summary.csv for this (ansatz, encoding, seed).

    Offset (per class) = base_test_per_class + sum of previous extension sizes
    (per class). All chunks for the same seed must use the same offset pool.
    """
    base_per_class = BASELINE.TEST_PER_CLASS
    matching = [r for r in summary_rows
                if r["ansatz"] == ansatz
                and r["encoding"] == encoding
                and int(r["seed"]) == seed]

    # Find existing extension chunks (chunk_id like 'ext_<n>')
    ext_chunks = []
    for r in matching:
        cid = r["chunk_id"]
        if cid.startswith("ext_"):
            try:
                n = int(cid.split("_")[1])
                size_per_class = int(r["test_size"]) // BASELINE.N_CLASSES
                ext_chunks.append((n, size_per_class))
            except (ValueError, IndexError):
                continue

    ext_chunks.sort()
    next_n = (max(n for n, _ in ext_chunks) + 1) if ext_chunks else 1
    used_extension_per_class = sum(s for _, s in ext_chunks)

    next_offset_per_class = base_per_class + used_extension_per_class
    next_chunk_id = f"ext_{next_n}"
    return next_chunk_id, next_offset_per_class


# ============================================================
# Modes
# ============================================================

def mode_train(args, runs, summary_path):
    if not runs:
        sys.exit("no runs to extend")

    print(f"=" * 70)
    print(f"TRAIN EXTENSION: {len(runs)} runs, +{args.epochs} epochs each")
    print(f"=" * 70)

    t_total = time.time()
    for i, run_dir in enumerate(runs, start=1):
        name = os.path.basename(run_dir)
        print(f"[{i}/{len(runs)}] EXTEND {name}")
        t0 = time.time()
        result = extend_training(run_dir, extra_epochs=args.epochs, log_fn=print)
        append_summary_row(summary_path, result)
        print(f"  -> n_epochs_trained={result['n_epochs_trained']} "
              f"test_acc={result['test_acc']:.4f} "
              f"({time.time() - t0:.1f}s)")

    print(f"=" * 70)
    print(f"DONE: {len(runs)} runs extended in {(time.time() - t_total) / 60:.1f} min")


def mode_test(args, runs, summary_path):
    if not runs:
        sys.exit("no runs to extend")

    # Group runs by (ansatz, encoding, seed) and read existing summary to find next chunk
    summary_rows = read_summary_rows(summary_path)

    # Verify all runs share the same set of (ansatz, encoding) - they should,
    # since "test extension always on all configs together".
    if args.runs:
        sys.exit("--runs is not allowed in test mode (must extend all configs together)")

    print(f"=" * 70)
    print(f"TEST EXTENSION: {len(runs)} runs, +{args.extra_per_class} test samples/class each")
    print(f"=" * 70)

    t_total = time.time()
    for i, run_dir in enumerate(runs, start=1):
        # Re-read config to get (ansatz, encoding, seed)
        import json as _json
        with open(os.path.join(run_dir, "config.json"), "r") as f:
            cfg = _json.load(f)
        ansatz, encoding, seed = cfg["ansatz"], cfg["encoding"], cfg["seed"]

        chunk_id, offset_per_class = next_chunk_id_and_offset(
            summary_rows, ansatz, encoding, seed)
        name = os.path.basename(run_dir)
        print(f"[{i}/{len(runs)}] EVAL {name} "
              f"chunk={chunk_id} offset_per_class={offset_per_class}")

        t0 = time.time()
        result = evaluate_test_chunk_at_offset(
            run_dir,
            offset_per_class=offset_per_class,
            extra_per_class=args.extra_per_class,
            chunk_id=chunk_id,
            log_fn=print,
        )
        append_summary_row(summary_path, result)

        # Update in-memory summary so the NEXT seed of the same combo computes
        # the right offset (in case there's seed variance in the loop order;
        # actually offset only depends on (ansatz, encoding, seed) so we'd
        # re-find it, but if the same seed appears twice we'd be safe).
        summary_rows = read_summary_rows(summary_path)

        print(f"  -> test_acc={result['test_acc']:.4f} "
              f"size={result['test_size']} "
              f"({time.time() - t0:.1f}s)")

    print(f"=" * 70)
    print(f"DONE: {len(runs)} runs eval'd in {(time.time() - t_total) / 60:.1f} min")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("train", "test"), required=True)
    p.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    p.add_argument("--epochs", type=int, default=None,
                   help="(train mode) extra epochs per run")
    p.add_argument("--extra-per-class", type=int, default=None,
                   help="(test mode) extra test samples per class to evaluate")
    p.add_argument("--runs", type=str, default=None,
                   help="(train mode only) comma-separated run dir names; "
                        "default: all. Test mode forces ALL.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "train" and args.epochs is None:
        sys.exit("--mode train requires --epochs N")
    if args.mode == "test" and args.extra_per_class is None:
        sys.exit("--mode test requires --extra-per-class N")

    all_runs = discover_run_dirs(args.results_dir)
    if not all_runs:
        sys.exit(f"no completed runs found in {args.results_dir!r}")

    names = args.runs.split(",") if args.runs else None
    runs = filter_runs(all_runs, names)

    summary_path = os.path.join(args.results_dir, "summary.csv")
    ensure_summary_header(summary_path)

    if args.mode == "train":
        mode_train(args, runs, summary_path)
    else:
        mode_test(args, runs, summary_path)


if __name__ == "__main__":
    main()

