"""
aggregate.py - Post-process results/summary.csv into per-arm mean/std and the
key PAIRED comparisons that answer "does the quantum part matter?".

Outputs:
  - results/summary_aggregated.csv : per-arm mean/std of test_acc/test_loss.
  - stdout                          : paired deltas (per seed) for
        A-B   trained  - frozen    (value of TRAINING the circuit)
        B-C2  frozen   - randfeat  (value of the quantum random STRUCTURE)
        A-C1  trained  - logistic  (quantum vs trivial classical)

Paired deltas use the SAME seed on both sides (identical data split), so each
delta is a within-split difference; we report mean +/- std over seeds.

Run AFTER run_suite.py.
"""

import csv
import os
import sys
from collections import defaultdict


SUMMARY = os.path.join(os.path.dirname(__file__), "results", "summary.csv")
AGG = os.path.join(os.path.dirname(__file__), "results", "summary_aggregated.csv")


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _read(summary_path):
    rows = []
    with open(summary_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def aggregate(summary_path=SUMMARY, agg_path=AGG):
    if not os.path.exists(summary_path):
        sys.exit(f"summary.csv not found at {summary_path!r}")
    rows = _read(summary_path)
    if not rows:
        sys.exit("summary.csv has no data rows")

    # ---- per-arm mean/std ----
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    agg_rows = []
    for arm in sorted(by_arm):
        accs = [float(r["test_acc"]) for r in by_arm[arm]]
        losses = [float(r["test_loss"]) for r in by_arm[arm]]
        agg_rows.append({
            "arm": arm,
            "n_seeds": len(accs),
            "n_params": int(by_arm[arm][0]["n_params"]),
            "test_acc_mean": _mean(accs),
            "test_acc_std": _std(accs),
            "test_loss_mean": _mean(losses),
            "test_loss_std": _std(losses),
        })

    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)

    # ---- paired deltas (per seed) ----
    acc = defaultdict(dict)   # acc[arm][seed] = test_acc
    for r in rows:
        acc[r["arm"]][int(r["seed"])] = float(r["test_acc"])

    def paired(a, b):
        seeds = sorted(set(acc.get(a, {})) & set(acc.get(b, {})))
        deltas = [acc[a][s] - acc[b][s] for s in seeds]
        return seeds, deltas

    acc_mean = {r["arm"]: r["test_acc_mean"] for r in agg_rows}

    print("=" * 70)
    print("Per-arm test accuracy (mean +/- std over seeds)")
    print("=" * 70)
    name = {"trained": "trained ", "frozen": "frozen  ", "mlp8": "mlp8    ",
            "randfeat": "randfeat", "logistic": "logistic"}
    for row in agg_rows:
        label = name.get(row["arm"], row["arm"])
        print(f"  {label:9s} | acc {row['test_acc_mean']*100:6.2f} "
              f"+/- {row['test_acc_std']*100:5.2f}  (n_params={row['n_params']})")

    def _cell(arm):
        return f"{acc_mean[arm]*100:5.1f}" if arm in acc_mean else "  -  "

    print("=" * 70)
    print("2x2 extractor design (test acc %, same 8-feat bottleneck + same head)")
    print("=" * 70)
    print("                  quantum     classical")
    print(f"  trainable        {_cell('trained')}        {_cell('mlp8')}")
    print(f"  fixed-random     {_cell('frozen')}        {_cell('randfeat')}")
    print(f"  (logistic on raw pixels: {_cell('logistic')} )")

    print("=" * 70)
    print("Paired deltas (same seed = same split); mean +/- std over seeds")
    print("=" * 70)
    comparisons = [
        ("trained", "mlp8",     "trained - mlp8     | Q vs classical TRAINABLE extractor"),
        ("frozen",  "randfeat", "frozen  - randfeat | Q vs classical FIXED-random extractor"),
        ("trained", "frozen",   "trained - frozen   | value of TRAINING the circuit"),
        ("trained", "logistic", "trained - logistic | hybrid Q vs linear-on-pixels"),
    ]
    for a, b, label in comparisons:
        seeds, deltas = paired(a, b)
        if not deltas:
            print(f"  {label}: (missing arm)")
            continue
        print(f"  {label}: {_mean(deltas)*100:+6.2f} +/- {_std(deltas)*100:5.2f} pp")

    print("=" * 70)
    print(f"  per-arm table -> {agg_path}")


if __name__ == "__main__":
    aggregate()
