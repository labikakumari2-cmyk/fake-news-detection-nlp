#!/usr/bin/env python3
"""
pipeline_phase4.py — Phase 4 orchestrator
Runs: Load cleaned data -> Fine-tune DistilBERT -> Evaluate -> Save
Usage: python pipeline_phase4.py

NOTE: CPU training takes 1-2 hours on 8GB RAM.
      For faster training, use Google Colab with GPU.
"""

import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from config import CLEANED_CSV, OUTPUTS_DIR, TEXT_COL, LABEL_COL, RANDOM_STATE
from models.distilbert_model import run_distilbert


def load_splits():
    """Load cleaned data and split — use RAW text for BERT (no TF-IDF)."""
    if not CLEANED_CSV.exists():
        raise FileNotFoundError("Run pipeline.py first to generate cleaned.csv")

    df = pd.read_csv(CLEANED_CSV)
    print(f"[phase4] Loaded {len(df):,} rows")

    # Use a subset for CPU training (full dataset = too slow on CPU)
    SAMPLE_SIZE = 8000
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    print(f"[phase4] Using {len(df):,} samples for CPU-friendly training")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df[TEXT_COL], df[LABEL_COL],
        test_size=0.15, stratify=df[LABEL_COL], random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.15, stratify=y_train_val, random_state=RANDOM_STATE
    )

    print(f"[phase4] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_phase4_report(metrics: dict) -> None:
    path = OUTPUTS_DIR / "phase4_distilbert_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("  Fake News Detector - Phase 4: DistilBERT Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"  Test Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"  Test F1 Score : {metrics['f1']:.4f}\n")
        f.write(f"  Test ROC-AUC  : {metrics['roc_auc']:.4f}\n")
    print(f"[phase4] Report saved -> {path}")


def main():
    print("=" * 55)
    print("  Fake News Detector - Phase 4: DistilBERT")
    print("=" * 55)
    print("\n  WARNING: CPU training is slow.")
    print("  Estimated time: 45-90 minutes on 8GB RAM.")
    print("  Tip: Use Google Colab for GPU-accelerated training.\n")

    # ── Step 1: Load data ─────────────────────────────────────
    print("[1/3] Loading and splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # ── Step 2: Fine-tune DistilBERT ─────────────────────────
    print("\n[2/3] Fine-tuning DistilBERT...")
    test_metrics, tokenizer = run_distilbert(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
    )

    # ── Step 3: Save report ───────────────────────────────────
    print("\n[3/3] Saving report...")
    save_phase4_report(test_metrics)

    print("\n" + "=" * 55)
    print("  Phase 4 complete!")
    print(f"  Test Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 Score : {test_metrics['f1']:.4f}")
    print(f"  Test ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
