# src/utils/compare_plots.py — Model comparison visualizations

import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS_DIR


def plot_model_comparison(results: list) -> None:
    names    = [r["name"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    f1       = [r["f1"] for r in results]
    roc_auc  = [r["roc_auc"] for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, accuracy, width, label="Accuracy", color="#3498db")
    ax.bar(x,         f1,       width, label="F1 Score", color="#2ecc71")
    ax.bar(x + width, roc_auc,  width, label="ROC-AUC",  color="#e74c3c")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.95, 1.001)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set")
    ax.legend()

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)

    plt.tight_layout()
    path = OUTPUTS_DIR / "model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plots] Saved -> model_comparison.png")


def plot_roc_comparison(results: list, y_test) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for r, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{r['name']} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = OUTPUTS_DIR / "roc_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plots] Saved -> roc_comparison.png")


def plot_cv_results(cv_results: dict) -> None:
    names  = list(cv_results.keys())
    means  = [cv_results[n].mean() for n in names]
    stds   = [cv_results[n].std()  for n in names]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85)
    ax.set_ylim(0.95, 1.001)
    ax.set_ylabel("F1 Score")
    ax.set_title("5-Fold Cross-Validation F1 Scores")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    plt.tight_layout()
    path = OUTPUTS_DIR / "cv_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plots] Saved -> cv_comparison.png")


def plot_training_time(results: list) -> None:
    names = [r["name"] for r in results]
    times = [r["train_time"] for r in results]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, times, color=colors, alpha=0.85)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training Time Comparison")
    ax.bar_label(bars, fmt="%.1fs", padding=3, fontsize=10)
    plt.tight_layout()
    path = OUTPUTS_DIR / "training_time.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plots] Saved -> training_time.png")
