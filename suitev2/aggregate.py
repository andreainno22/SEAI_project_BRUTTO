"""
aggregate.py — Post-processing: reads results/summary.csv and produces
mean/std across seeds for each (ansatz, encoding, n_epochs_trained, chunk_id).

Groups rows by (ansatz, encoding, n_epochs_trained, chunk_id). For each
group, computes:
  - mean and std of test_acc and test_loss across seeds
  - summed confusion matrix across seeds (suite-level diagnostic)
  - per-class accuracy from the summed CM

Per-combo confusion matrices are written to results/confusion_matrices/
as <ansatz>_<encoding>__<chunk_id>__ep<N>.csv.

Run AFTER run_suite.py (and optionally extend_run.py).
"""

import csv
import os
import sys
from collections import defaultdict


SUMMARY_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "results", "summary.csv")
AGG_PATH_DEFAULT     = os.path.join(os.path.dirname(__file__), "results", "summary_aggregated.csv")
CM_DIR_DEFAULT       = os.path.join(os.path.dirname(__file__), "results", "confusion_matrices")


def _mean(values):
    return sum(values) / len(values) if values else 0.0

def _std(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def aggregate(summary_path=SUMMARY_PATH_DEFAULT,
              agg_path=AGG_PATH_DEFAULT,
              cm_dir=CM_DIR_DEFAULT):
    if not os.path.exists(summary_path):
        sys.exit(f"summary.csv not found at {summary_path!r}")

    # Group by (ansatz, encoding, n_epochs_trained, chunk_id)
    groups = defaultdict(list)
    with open(summary_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["ansatz"], row["encoding"],
                   int(row["n_epochs_trained"]), row["chunk_id"])
            groups[key].append(row)

    if not groups:
        sys.exit("summary.csv has no data rows")

    os.makedirs(cm_dir, exist_ok=True)
    agg_rows = []
    for key, rows in sorted(groups.items()):
        ansatz, encoding, n_epochs_trained, chunk_id = key
        test_accs   = [float(r["test_acc"])  for r in rows]
        test_losses = [float(r["test_loss"]) for r in rows]
        n_params    = int(rows[0]["n_params"])
        n_seeds     = len(rows)
        test_offset = int(rows[0]["test_offset"])
        test_size   = int(rows[0]["test_size"])

        # Sum confusion matrices across seeds
        cm_sum = [[0] * 4 for _ in range(4)]
        for r in rows:
            for i in range(4):
                for j in range(4):
                    cm_sum[i][j] += int(r[f"cm_{i}{j}"])

        # Per-class accuracy from summed CM
        per_class_acc = []
        for i in range(4):
            row_total = sum(cm_sum[i])
            per_class_acc.append(cm_sum[i][i] / row_total if row_total else 0.0)

        agg_rows.append({
            "ansatz": ansatz,
            "encoding": encoding,
            "n_epochs_trained": n_epochs_trained,
            "chunk_id": chunk_id,
            "test_offset": test_offset,
            "test_size": test_size,
            "n_seeds": n_seeds,
            "n_params": n_params,
            "test_acc_mean":  _mean(test_accs),
            "test_acc_std":   _std(test_accs),
            "test_loss_mean": _mean(test_losses),
            "test_loss_std":  _std(test_losses),
            "class0_acc": per_class_acc[0],
            "class1_acc": per_class_acc[1],
            "class2_acc": per_class_acc[2],
            "class3_acc": per_class_acc[3],
        })

        # Write per-group confusion matrix
        cm_name = f"{ansatz}_{encoding}__{chunk_id}__ep{n_epochs_trained}.csv"
        cm_path = os.path.join(cm_dir, cm_name)
        with open(cm_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true \\ pred", "0(T-shirt)", "1(Trouser)", "2(Sneaker)", "3(Bag)"])
            class_labels = ["0(T-shirt)", "1(Trouser)", "2(Sneaker)", "3(Bag)"]
            for i in range(4):
                w.writerow([class_labels[i]] + cm_sum[i])

    fields = list(agg_rows[0].keys())
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)

    n_runs = sum(len(v) for v in groups.values())
    print(f"Aggregated {n_runs} run rows -> {len(agg_rows)} groups "
          f"(by ansatz, encoding, n_epochs_trained, chunk_id)")
    print(f"  summary_aggregated.csv  -> {agg_path}")
    print(f"  confusion_matrices/*.csv -> {cm_dir}")


if __name__ == "__main__":
    aggregate()
