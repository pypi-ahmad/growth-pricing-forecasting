from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_rossmann_data(
    raw_dir: Path,
    sample_store_frac: float | None = None,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    train = _read(raw_dir / "train.csv")
    test = _read(raw_dir / "test.csv")
    store = _read(raw_dir / "store.csv")

    if train.empty or store.empty:
        raise FileNotFoundError(
            f"Rossmann CSVs not found under {raw_dir}. Run scripts/download_data.sh first."
        )

    if sample_store_frac is not None and 0 < sample_store_frac < 1:
        sampled_stores = (
            train[["Store"]]
            .drop_duplicates()
            .sample(frac=sample_store_frac, random_state=random_state)["Store"]
            .tolist()
        )
        train = train[train["Store"].isin(sampled_stores)].copy()
        test = test[test["Store"].isin(sampled_stores)].copy() if not test.empty else test
        store = store[store["Store"].isin(sampled_stores)].copy()

    return {"train": train, "test": test, "store": store}


def merge_train_store(train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    merged = train_df.merge(store_df, on="Store", how="left")
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged.sort_values(["Store", "Date"]).reset_index(drop=True)
    return merged


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Open"] = out["Open"].fillna(1)
    out["Promo"] = out["Promo"].fillna(0)
    out["SchoolHoliday"] = out["SchoolHoliday"].fillna(0)
    out["StateHoliday"] = out["StateHoliday"].fillna("0")

    # Keep closed-day rows so baselines can compare policy impacts.
    out["Sales"] = out["Sales"].clip(lower=0)
    return out


def time_split(
    df: pd.DataFrame,
    date_col: str = "Date",
    holdout_days: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = df[date_col].max()
    cutoff = max_date - pd.Timedelta(days=holdout_days)
    train_df = df[df[date_col] <= cutoff].copy()
    holdout_df = df[df[date_col] > cutoff].copy()
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)
