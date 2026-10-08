# src/data/ingest.py — Load raw CSVs and combine into one labelled dataset

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRUE_CSV, FAKE_CSV, COMBINED_CSV, TEXT_COL, LABEL_COL


def load_isot() -> pd.DataFrame:
    if not TRUE_CSV.exists() or not FAKE_CSV.exists():
        raise FileNotFoundError(
            f"Place True.csv and Fake.csv inside {TRUE_CSV.parent}"
        )
    true_df = pd.read_csv(TRUE_CSV)
    fake_df = pd.read_csv(FAKE_CSV)
    true_df[LABEL_COL] = 1
    fake_df[LABEL_COL] = 0
    df = pd.concat([true_df, fake_df], ignore_index=True)
    print(f"[ingest] Loaded {len(true_df):,} real + {len(fake_df):,} fake = {len(df):,} total rows")
    return df


def merge_title_body(df: pd.DataFrame) -> pd.DataFrame:
    title_col = "title" if "title" in df.columns else None
    body_col  = "text"  if "text"  in df.columns else None
    if title_col and body_col:
        df[TEXT_COL] = df[title_col].fillna("") + " " + df[body_col].fillna("")
    elif body_col:
        df[TEXT_COL] = df[body_col].fillna("")
    elif title_col:
        df[TEXT_COL] = df[title_col].fillna("")
    else:
        raise ValueError("No 'title' or 'text' column found in dataset.")
    return df


def save_combined(df: pd.DataFrame) -> None:
    COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[[TEXT_COL, LABEL_COL]].to_csv(COMBINED_CSV, index=False)
    print(f"[ingest] Saved combined dataset → {COMBINED_CSV}")


def run_ingestion() -> pd.DataFrame:
    df = load_isot()
    df = merge_title_body(df)
    save_combined(df)
    return df


if __name__ == "__main__":
    run_ingestion()
