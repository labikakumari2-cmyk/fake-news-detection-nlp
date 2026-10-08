#!/usr/bin/env python3
"""
pipeline_phase3.py — Phase 3 orchestrator
Runs: Load features -> Train all models -> Cross-validate -> Compare -> Plot
Usage: python pipeline_phase3.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from data.split_vectorize import run_split_vectorize
from models.compare import (
    MODELS, train_and_evaluate,
    cross_validate_models, save_comparison_report
)
from utils.compare_plots import (
    plot_model_comparison, plot_roc_comparison,
    plot_cv_results, plot_training_time
)


def main() -> None:
    print("=" * 55)
    print("  Fake News Detector - Phase 3: Model Comparison")
    print("=" * 55)

    # ── Step 1: Load features ─────────────────────────────────
    print("\n[1/4] Loading TF-IDF features...")
    (
        X_train_vec, X_val_vec, X_test_vec,
        y_train, y_val, y_test,
        vectorizer
    ) = run_split_vectorize()

    # ── Step 2: Train & evaluate all models ───────────────────
    print("\n[2/4] Training and evaluating all models...")
    results = []
    for name, model in MODELS.items():
        r = train_and_evaluate(
            name, model,
            X_train_vec, y_train,
            X_test_vec, y_test
        )
        results.append(r)

    # ── Step 3: Cross-validation ──────────────────────────────
    print("\n[3/4] Running cross-validation...")
    cv_results = cross_validate_models(X_train_vec, y_train, cv=5)
    save_comparison_report(results, cv_results)

    # ── Step 4: Plots ─────────────────────────────────────────
    print("\n[4/4] Generating comparison plots...")
    plot_model_comparison(results)
    plot_roc_comparison(results, y_test)
    plot_cv_results(cv_results)
    plot_training_time(results)

    # ── Summary ───────────────────────────────────────────────
    best = max(results, key=lambda r: r["f1"])
    print("\n" + "=" * 55)
    print("  Phase 3 complete!")
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("  " + "-" * 50)
    for r in results:
        print(f"  {r['name']:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")
    print(f"\n  Best model: {best['name']} (F1 = {best['f1']:.4f})")
    print("=" * 55)


if __name__ == "__main__":
    main()
