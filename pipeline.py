#!/usr/bin/env python3
"""
pipeline.py — Phase 1 orchestrator
Runs: Ingest → EDA → Preprocess → Split + Vectorize
Usage: python pipeline.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from config import COMBINED_CSV, CLEANED_CSV
from data.ingest import run_ingestion
from data.preprocess import load_and_preprocess
from data.split_vectorize import run_split_vectorize
from utils.eda import run_eda
import pandas as pd


def main() -> None:
    print("=" * 55)
    print("  Fake News Detector — Phase 1 Pipeline")
    print("=" * 55)

    print("\n[1/4] Ingesting raw data...")
    run_ingestion()

    print("\n[2/4] Running EDA on combined data...")
    if COMBINED_CSV.exists():
        combined_df = pd.read_csv(COMBINED_CSV)
        run_eda(combined_df)

    print("\n[3/4] Preprocessing text...")
    if not CLEANED_CSV.exists():
        load_and_preprocess()
    else:
        print(f"  Cleaned CSV already exists — skipping.")

    print("\n[4/4] Splitting dataset and building TF-IDF features...")
    result = run_split_vectorize()
    X_train_vec, X_val_vec, X_test_vec, y_train, y_val, y_test, vectorizer = result

    print("\n" + "=" * 55)
    print("  Phase 1 complete! Ready for model training.")
    print(f"  Train matrix : {X_train_vec.shape}")
    print(f"  Val matrix   : {X_val_vec.shape}")
    print(f"  Test matrix  : {X_test_vec.shape}")
    print("=" * 55)


if __name__ == "__main__":
    main()
