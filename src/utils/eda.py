# src/utils/eda.py — Exploratory Data Analysis helpers

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS_DIR, TEXT_COL, LABEL_COL

LABEL_MAP = {1: "Real", 0: "Fake"}


def class_distribution(df: pd.DataFrame) -> None:
    counts = df[LABEL_COL].map(LABEL_MAP).value_counts()
    print("\n── Class Distribution ──────────────────")
    print(counts.to_string())
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index, y=counts.values, palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUTS_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"[eda] Saved → class_distribution.png")


def text_length_distribution(df: pd.DataFrame) -> None:
    df = df.copy()
    df["length"] = df[TEXT_COL].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 4))
    for label_id, color in [(1, "#2ecc71"), (0, "#e74c3c")]:
        ax.hist(df[df[LABEL_COL] == label_id]["length"],
                bins=60, alpha=0.6, color=color, label=LABEL_MAP[label_id])
    ax.set_xlabel("Word count")
    ax.set_ylabel("Frequency")
    ax.set_title("Text Length Distribution by Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "text_length_distribution.png", dpi=150)
    plt.close()
    print(f"[eda] Saved → text_length_distribution.png")


def top_words(df: pd.DataFrame, n: int = 20) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (label_id, color) in zip(axes, [(1, "#2ecc71"), (0, "#e74c3c")]):
        words = " ".join(df[df[LABEL_COL] == label_id][TEXT_COL]).split()
        common = Counter(words).most_common(n)
        terms, freqs = zip(*common)
        sns.barplot(x=list(freqs), y=list(terms), color=color, ax=ax)
        ax.set_title(f"Top {n} words — {LABEL_MAP[label_id]}")
        ax.set_xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "top_words.png", dpi=150)
    plt.close()
    print(f"[eda] Saved → top_words.png")


def run_eda(df: pd.DataFrame) -> None:
    print("\n════ Running EDA ════")
    class_distribution(df)
    text_length_distribution(df)
    top_words(df)
    print("════ EDA complete ════\n")
