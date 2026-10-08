# src/utils/plots.py — Confusion matrix + ROC curve plots

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS_DIR


def plot_confusion_matrix(cm, split_name: str = "Test") -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {split_name} Set")
    plt.tight_layout()
    path = OUTPUTS_DIR / f"confusion_matrix_{split_name.lower()}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plots] Saved → confusion_matrix_{split_name.lower()}.png")


def plot_roc_curve(y_true, y_proba, split_name: str = "Test") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3498db", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {split_name} Set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = OUTPUTS_DIR / f"roc_curve_{split_name.lower()}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[plots] Saved → roc_curve_{split_name.lower()}.png")


def plot_top_features(model, vectorizer, n: int = 20) -> None:
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    top_real = np.argsort(coefs)[-n:][::-1]
    top_fake = np.argsort(coefs)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Real news top words
    axes[0].barh(feature_names[top_real], coefs[top_real], color="#2ecc71")
    axes[0].set_title(f"Top {n} words → Real News")
    axes[0].invert_yaxis()

    # Fake news top words
    axes[1].barh(feature_names[top_fake], coefs[top_fake], color="#e74c3c")
    axes[1].set_title(f"Top {n} words → Fake News")
    axes[1].invert_yaxis()

    plt.suptitle("Most Influential Words (Logistic Regression Coefficients)", y=1.02)
    plt.tight_layout()
    path = OUTPUTS_DIR / "top_features.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plots] Saved → top_features.png")
