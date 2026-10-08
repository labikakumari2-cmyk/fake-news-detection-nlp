# src/models/compare.py — Random Forest + XGBoost comparison

import sys
from pathlib import Path
import joblib
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS_DIR

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1
    ),
}


def train_and_evaluate(name, model, X_train, y_train, X_test, y_test) -> dict:
    print(f"\n[compare] Training {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"  Train time : {elapsed:.1f}s")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {roc_auc:.4f}")

    # Save model
    slug = name.lower().replace(" ", "_")
    path = OUTPUTS_DIR / f"{slug}_model.joblib"
    joblib.dump(model, path)
    print(f"  Saved      → {path.name}")

    return {
        "name": name,
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc_auc,
        "train_time": elapsed,
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def cross_validate_models(X_train, y_train, cv: int = 5) -> dict:
    print(f"\n[compare] Running {cv}-fold cross-validation...")
    cv_results = {}
    for name, model in MODELS.items():
        print(f"  CV for {name}...")
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="f1", n_jobs=-1
        )
        cv_results[name] = scores
        print(f"    F1: {scores.mean():.4f} +/- {scores.std():.4f}")
    return cv_results


def save_comparison_report(results: list, cv_results: dict) -> None:
    path = OUTPUTS_DIR / "phase3_comparison_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("  Fake News Detector - Phase 3 Comparison Report\n")
        f.write("=" * 55 + "\n\n")

        f.write("-- Test Set Results --\n")
        f.write(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10} {'Time(s)':>10}\n")
        f.write("-" * 65 + "\n")
        for r in results:
            f.write(
                f"{r['name']:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
                f"{r['roc_auc']:>10.4f} {r['train_time']:>10.1f}\n"
            )

        f.write("\n-- Cross-Validation F1 Scores (5-fold) --\n")
        for name, scores in cv_results.items():
            f.write(f"  {name:<25}: {scores.mean():.4f} +/- {scores.std():.4f}\n")

    print(f"\n[compare] Report saved -> {path}")
