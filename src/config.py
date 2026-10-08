# config.py — Central configuration for Fake News Detector (Phase 1)

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_RAW       = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS_DIR    = BASE_DIR / "outputs"

# ── Dataset ───────────────────────────────────────────────────────────────────
TRUE_CSV     = DATA_RAW / "True.csv"
FAKE_CSV     = DATA_RAW / "Fake.csv"
COMBINED_CSV = DATA_PROCESSED / "combined.csv"
CLEANED_CSV  = DATA_PROCESSED / "cleaned.csv"

# ── Text columns ──────────────────────────────────────────────────────────────
TEXT_COL  = "text"
LABEL_COL = "label"

# ── Pre-processing ────────────────────────────────────────────────────────────
MAX_FEATURES = 50_000
NGRAM_RANGE  = (1, 2)
MIN_DF       = 2
MAX_DF       = 0.95
SUBLINEAR_TF = True

# ── Train / Val / Test split ──────────────────────────────────────────────────
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
RANDOM_STATE = 42

# ── Model artefacts ───────────────────────────────────────────────────────────
MODEL_PATH      = OUTPUTS_DIR / "lr_model.joblib"
VECTORIZER_PATH = OUTPUTS_DIR / "tfidf_vectorizer.joblib"
REPORT_PATH     = OUTPUTS_DIR / "classification_report.txt"
