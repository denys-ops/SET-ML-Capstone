"""Data loading for training (HW2).

Two modes:
  * load_splits()  — read the DVC-tracked parquet splits (default, local/MinIO).
  * rebuild()      — re-create splits from the public HF dataset (for Colab, where
                     the local MinIO DVC remote is unreachable). Same clean+split
                     logic as the capstone F0, so the result matches load_splits().

Both yield train/val/test DataFrames with columns: text (normalised), label (0/1).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocess import html_to_scoring_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
SEED = 42


def load_splits(raw_dir: Path | str = RAW) -> dict[str, pd.DataFrame]:
    """Load train/val/test parquet (the DVC-tracked splits)."""
    raw = Path(raw_dir)
    out = {}
    for name in ("train", "val", "test"):
        df = pd.read_parquet(raw / f"{name}.parquet")
        out[name] = df[["text", "label"]].reset_index(drop=True)
    print("loaded splits:", {k: len(v) for k, v in out.items()})
    return out


def rebuild(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Rebuild splits from the public HF dataset (Colab path). Mirrors capstone F0."""
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    raw = load_dataset("zefang-liu/phishing-email-dataset", split="train").to_pandas()
    df = raw.rename(columns={"Email Text": "text_raw", "Email Type": "etype"})
    df["text_raw"] = df["text_raw"].fillna("").astype(str)
    df["label"] = (df["etype"].str.strip() == "Phishing Email").astype(int)
    df["text"] = [html_to_scoring_text(t) for t in df["text_raw"]]

    before = len(df)
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"cleaned: {before} -> {len(df)} rows")

    train_df, tmp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=seed)
    out = {
        "train": train_df[["text", "label"]].reset_index(drop=True),
        "val": val_df[["text", "label"]].reset_index(drop=True),
        "test": test_df[["text", "label"]].reset_index(drop=True),
    }
    print("rebuilt splits:", {k: len(v) for k, v in out.items()})
    return out


def get_splits(source: str = "parquet", raw_dir: Path | str = RAW) -> dict[str, pd.DataFrame]:
    """`parquet` (default) or `hf` (rebuild from HuggingFace, for Colab)."""
    return rebuild() if source == "hf" else load_splits(raw_dir)
