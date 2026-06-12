from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_mercari_data(raw_dir: Path, sample_frac: float | None = 0.25, random_state: int = 42) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    train_path_tsv = raw_dir / "train.tsv"
    train_path_csv = raw_dir / "train.csv"

    if train_path_tsv.exists():
        df = pd.read_csv(train_path_tsv, sep="	")
    elif train_path_csv.exists():
        df = pd.read_csv(train_path_csv)
    else:
        raise FileNotFoundError(
            f"Mercari train file not found under {raw_dir}. Run scripts/download_data.sh first."
        )

    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state).copy()

    return df


def clean_mercari(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["name"] = out.get("name", "").fillna("").astype(str)
    out["item_description"] = out.get("item_description", "").fillna("missing_description").astype(str)
    out["category_name"] = out.get("category_name", "unknown").fillna("unknown").astype(str)
    out["brand_name"] = out.get("brand_name", "unknown_brand").fillna("unknown_brand").astype(str)
    out["shipping"] = out.get("shipping", 0).fillna(0).astype(int)
    out["item_condition_id"] = out.get("item_condition_id", 0).fillna(0).astype(int)
    out["price"] = out.get("price", 0).fillna(0).clip(lower=0)
    out = out[out["price"] > 0].copy()
    return out.reset_index(drop=True)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    # Stratify by price quantiles for stable holdout mix.
    bins = pd.qcut(df["price"], q=10, labels=False, duplicates="drop")
    train_df, holdout_df = train_test_split(
        df,
        test_size=test_size,
        stratify=bins,
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


def build_pycaret_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["title_len"] = out["name"].str.len().clip(upper=300)
    out["desc_len"] = out["item_description"].str.len().clip(upper=3000)
    out["category_main"] = out["category_name"].str.split("/").str[0].fillna("unknown")
    out["has_brand"] = (out["brand_name"] != "unknown_brand").astype(int)
    out["log_price"] = np.log1p(out["price"].values)

    keep_cols = [
        "title_len",
        "desc_len",
        "shipping",
        "item_condition_id",
        "category_main",
        "brand_name",
        "has_brand",
        "log_price",
    ]
    return out[keep_cols].copy()
