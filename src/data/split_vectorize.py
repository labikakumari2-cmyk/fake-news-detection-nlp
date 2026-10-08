# src/data/split_vectorize.py — Train/Val/Test split + TF-IDF feature extraction

import pandas as pd
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CLEANED_CSV, OUTPUTS_DIR,
    TEXT_COL, LABEL_COL,
    MAX_FEATURES, NGRAM_RANGE, MIN_DF, MAX_DF, SUBLINEAR_TF,
    TEST_SIZE, VAL_SIZE, RANDOM_STATE,
    VECTORIZER_PATH,
)


def split_data(df: pd.DataFrame):
    X, y = df[TEXT_COL], df[LABEL_COL]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    relative_val = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val,
        stratify=y_train_val, random_state=RANDOM_STATE)
    print(f"[split] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_vectorizer(X_train: pd.Series) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE,
        min_df=MIN_DF, max_df=MAX_DF, sublinear_tf=SUBLINEAR_TF,
        strip_accents="unicode", analyzer="word",
    )
    vectorizer.fit(X_train)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[vectorize] Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"[vectorize] Saved vectorizer → {VECTORIZER_PATH}")
    return vectorizer


def run_split_vectorize():
    if not CLEANED_CSV.exists():
        raise FileNotFoundError("Cleaned CSV not found. Run preprocess.py first.")
    df = pd.read_csv(CLEANED_CSV)
    print(f"[split_vectorize] Loaded {len(df):,} rows")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    vectorizer = build_vectorizer(X_train)
    X_train_vec = vectorizer.transform(X_train)
    X_val_vec   = vectorizer.transform(X_val)
    X_test_vec  = vectorizer.transform(X_test)
    return X_train_vec, X_val_vec, X_test_vec, y_train, y_val, y_test, vectorizer


if __name__ == "__main__":
    run_split_vectorize()
