
"""
pipeline_phase2.py — Phase 2 orchestrator
Runs: Load features → Train LR → Evaluate → Plot
Usage: python pipeline_phase2.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from data.split_vectorize import run_split_vectorize
from models.train import train_model, evaluate_model, save_report
from utils.plots import plot_confusion_matrix, plot_roc_curve, plot_top_features


def main() -> None:
    print("=" * 55)
    print("  Fake News Detector — Phase 2: Model Training")
    print("=" * 55)

    print("\n[1/4] Loading TF-IDF features...")
    (
        X_train_vec, X_val_vec, X_test_vec,
        y_train, y_val, y_test,
        vectorizer
    ) = run_split_vectorize()

    print("\n[2/4] Training model...")
    model = train_model(X_train_vec, y_train)

    print("\n[3/4] Evaluating model...")
    val_metrics  = evaluate_model(model, X_val_vec,  y_val,  "Validation")
    test_metrics = evaluate_model(model, X_test_vec, y_test, "Test")
    save_report(val_metrics, test_metrics)

    print("\n[4/4] Generating evaluation plots...")
    plot_confusion_matrix(val_metrics["confusion_matrix"],  "Validation")
    plot_confusion_matrix(test_metrics["confusion_matrix"], "Test")
    plot_roc_curve(y_val,  val_metrics["y_proba"],  "Validation")
    plot_roc_curve(y_test, test_metrics["y_proba"], "Test")
    plot_top_features(model, vectorizer)

    print("\n" + "=" * 55)
    print("  Phase 2 complete!")
    print(f"  Test Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 Score : {test_metrics['f1']:.4f}")
    print(f"  Test ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
