#!/usr/bin/env python3
"""Summarize the per-epoch CSV logs written by train_mnist --log.

Usage: scripts/plot_results.py [results_dir] [output_png]

Prints a Markdown table (mean and standard deviation over seeds of the final-epoch test
accuracy, hidden-layer parameter count, and mean seconds per training epoch) and saves a
figure with test loss and test accuracy per epoch for every variant.
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VARIANTS = ["none", "dense", "butterfly"]


def load_runs(results_dir):
    """Return {variant: {seed: [row, ...]}} with rows sorted by epoch."""
    runs = defaultdict(dict)
    for path in sorted(Path(results_dir).glob("*.csv")):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            runs[rows[0]["variant"]][int(rows[0]["seed"])] = sorted(rows, key=lambda r: int(r["epoch"]))
    return runs


def mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def print_table(runs):
    print("| Variant | Hidden parameters | Test accuracy (mean ± std) | Seconds per epoch | Seeds |")
    print("|---------|-------------------|----------------------------|-------------------|-------|")
    for variant in VARIANTS:
        if variant not in runs:
            continue
        seeds = runs[variant]
        final_acc = [float(rows[-1]["test_accuracy"]) * 100 for rows in seeds.values()]
        seconds = [float(r["epoch_seconds"]) for rows in seeds.values() for r in rows]
        hidden = int(next(iter(seeds.values()))[0]["hidden_params"])
        acc_mean, acc_std = mean_std(final_acc)
        print(f"| `{variant}` | {hidden:,} | {acc_mean:.2f} % ± {acc_std:.2f} | "
              f"{statistics.mean(seconds):.1f} | {len(seeds)} |")


def plot(runs, output):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    for variant in VARIANTS:
        if variant not in runs:
            continue
        seeds = list(runs[variant].values())
        epochs = [int(r["epoch"]) for r in seeds[0]]
        loss = [statistics.mean(float(rows[i]["test_loss"]) for rows in seeds) for i in range(len(epochs))]
        acc = [statistics.mean(float(rows[i]["test_accuracy"]) * 100 for rows in seeds)
               for i in range(len(epochs))]
        ax_loss.plot(epochs, loss, marker="o", label=variant)
        ax_acc.plot(epochs, acc, marker="o", label=variant)
    ax_loss.set(xlabel="epoch", ylabel="test loss (mean over seeds)", title="Test loss")
    ax_acc.set(xlabel="epoch", ylabel="test accuracy % (mean over seeds)", title="Test accuracy")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output = sys.argv[2] if len(sys.argv) > 2 else "docs/img/test_curves.png"
    runs = load_runs(results_dir)
    if not runs:
        sys.exit(f"no CSV logs found in {results_dir}")
    print_table(runs)
    plot(runs, output)
    print(f"\nfigure written to {output}")


if __name__ == "__main__":
    main()
