# src/data/preprocess.py — Text cleaning + normalisation pipeline

import re
import pandas as pd
import nltk
import sys
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import COMBINED_CSV, CLEANED_CSV, TEXT_COL, LABEL_COL

for resource in ("stopwords", "punkt", "wordnet"):
    nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

_URL_RE     = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE    = re.compile(r"<.*?>")
_MENTION_RE = re.compile(r"@\w+")
_NONALPHA   = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _NONALPHA.sub(" ", text)
    tokens = _WHITESPACE.split(text.strip())
    tokens = [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t and t not in STOP_WORDS and len(t) > 2
    ]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("[preprocess] Cleaning text (this may take ~2 mins)...")
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].progress_apply(clean_text)
    before = len(df)
    df = df[df[TEXT_COL].str.strip().astype(bool)].reset_index(drop=True)
    print(f"[preprocess] Dropped {before - len(df)} empty rows after cleaning.")
    return df


def load_and_preprocess() -> pd.DataFrame:
    if not COMBINED_CSV.exists():
        raise FileNotFoundError(f"Combined CSV not found. Run ingest.py first.")
    df = pd.read_csv(COMBINED_CSV)
    print(f"[preprocess] Loaded {len(df):,} rows")
    df = preprocess_dataframe(df)
    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False)
    print(f"[preprocess] Saved cleaned dataset → {CLEANED_CSV}")
    return df


if __name__ == "__main__":
    load_and_preprocess()
