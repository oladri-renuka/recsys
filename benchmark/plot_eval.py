"""
Generates evaluation charts from real eval_summary.json results.
Produces two plots:
  1. Precision/Recall/NDCG/Hit curves across K values
  2. Popularity bias histogram - how often each item appears in top-10
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def load_results():
    path = RESULTS_DIR / "eval_summary.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run eval_framework.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def plot_metrics_curves(data):
    """Plot Precision/Recall/NDCG/Hit curves across K values."""
    metrics = data["metrics"]
    k_values = data["k_values"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor("#FAFBFC")
    fig.suptitle("SASRec Evaluation Metrics Across K", fontsize=15,
                 fontweight="bold", color="#2C3E50", y=1.02)

    metric_types = [
        ("Hit", "#E74C3C"),
        ("Precision", "#3498DB"),
        ("Recall", "#27AE60"),
        ("NDCG", "#8E44AD"),
    ]

    for ax, (metric_name, color) in zip(axes, metric_types):
        values = [metrics[f"{metric_name}@{k}"] * 100 for k in k_values]
        ax.plot(k_values, values, "o-", color=color, linewidth=2.5,
                markersize=7, markerfacecolor="white", markeredgewidth=2)
        ax.set_xlabel("K", fontsize=10)
        ax.set_ylabel(f"{metric_name} (%)", fontsize=10)
        ax.set_title(f"{metric_name}@K", fontsize=12, fontweight="bold", color="#2C3E50")
        ax.set_xticks(k_values)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#FAFBFC")

        for k, v in zip(k_values, values):
            ax.annotate(f"{v:.1f}", (k, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color=color, fontweight="bold")

    plt.tight_layout()
    out = DOCS_DIR / "eval_metrics.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#FAFBFC")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


def plot_popularity_bias(data):
    """Plot popularity bias: recommendation frequency vs item popularity."""
    metrics = data["metrics"]
    rec_counts = {int(k): v for k, v in data["recommendation_counts"].items()}
    item_pop = {int(k): v for k, v in data["item_popularity_top100"].items()}
    num_items = data["num_items"]
    coverage = metrics["Coverage@10"]
    pop_bias = metrics["PopularityBias@10"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#FAFBFC")
    fig.suptitle("Catalog Coverage and Popularity Bias", fontsize=15,
                 fontweight="bold", color="#2C3E50", y=1.02)

    # 1. Recommendation frequency distribution
    ax = axes[0]
    counts = sorted(rec_counts.values(), reverse=True)
    ax.bar(range(len(counts)), counts, color="#3498DB", alpha=0.7, width=1.0)
    ax.set_xlabel("Item rank (by recommendation frequency)", fontsize=10)
    ax.set_ylabel("Times recommended (top-10)", fontsize=10)
    ax.set_title(f"Recommendation Distribution\nCoverage@10: {coverage*100:.1f}% of {num_items} items",
                 fontsize=11, fontweight="bold", color="#2C3E50")
    ax.set_facecolor("#FAFBFC")
    ax.set_xlim(-5, min(len(counts), 500))

    # 2. Popularity bias breakdown
    ax = axes[1]
    labels = ["Top 20% popular\nitems", "Remaining 80%\nof catalog"]
    sizes = [pop_bias * 100, (1 - pop_bias) * 100]
    colors_pie = ["#E74C3C", "#27AE60"]
    explode = (0.05, 0)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors_pie,
        explode=explode, startangle=90, textprops={"fontsize": 10},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    ax.set_title(f"Popularity Bias in Top-10 Recommendations\n"
                 f"{pop_bias*100:.1f}% from top-20% popular items",
                 fontsize=11, fontweight="bold", color="#2C3E50")

    plt.tight_layout()
    out = DOCS_DIR / "popularity_bias.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#FAFBFC")
    plt.close(fig)
    print(f"  Saved {out}")
    return out


if __name__ == "__main__":
    os.makedirs(DOCS_DIR, exist_ok=True)
    print("Loading evaluation results...")
    data = load_results()
    print("Generating charts...")
    plot_metrics_curves(data)
    plot_popularity_bias(data)
    print("Done.")
