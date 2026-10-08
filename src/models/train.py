# src/models/train.py — Train Logistic Regression on TF-IDF features

import sys
from pathlib import Path
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_PATH, VECTORIZER_PATH, OUTPUTS_DIR

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)


def train_model(X_train, y_train) -> LogisticRegression:
    print("[train] Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[train] Model saved → {MODEL_PATH}")
    return model


def evaluate_model(model, X, y, split_name: str = "Test") -> dict:
    print(f"\n[eval] Evaluating on {split_name} set...")
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    acc     = accuracy_score(y, y_pred)
    f1      = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)
    cm      = confusion_matrix(y, y_pred)
    report  = classification_report(y, y_pred, target_names=["Fake", "Real"])

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"\n  Classification Report:\n{report}")
    print(f"  Confusion Matrix:\n{cm}")

    return {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "report": report,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def save_report(val_metrics: dict, test_metrics: dict) -> None:
    report_path = OUTPUTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("  Fake News Detector — Phase 2 Report\n")
        f.write("=" * 55 + "\n\n")

        f.write("── Validation Set ──────────────────────\n")
        f.write(f"  Accuracy : {val_metrics['accuracy']:.4f}\n")
        f.write(f"  F1 Score : {val_metrics['f1']:.4f}\n")
        f.write(f"  ROC-AUC  : {val_metrics['roc_auc']:.4f}\n")
        f.write(f"\n{val_metrics['report']}\n")

        f.write("── Test Set ────────────────────────────\n")
        f.write(f"  Accuracy : {test_metrics['accuracy']:.4f}\n")
        f.write(f"  F1 Score : {test_metrics['f1']:.4f}\n")
        f.write(f"  ROC-AUC  : {test_metrics['roc_auc']:.4f}\n")
        f.write(f"\n{test_metrics['report']}\n")

    print(f"\n[report] Saved → {report_path}")
