from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_insurance_data(
    raw_dir: Path,
    sample_frac: float | None = None,
    random_state: int = 42,
    max_rows: int = 40_000,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"train.csv not found under {raw_dir}. Run scripts/download_data.sh first."
        )

    train_df = pd.read_csv(train_path)
    if sample_frac is not None and 0 < sample_frac < 1:
        train_df = train_df.sample(frac=sample_frac, random_state=random_state).copy()
    if max_rows > 0 and len(train_df) > max_rows:
        train_df = train_df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    return train_df


def basic_clean(df: pd.DataFrame, target_col: str = "Response") -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates().reset_index(drop=True)
    if target_col in out.columns:
        out[target_col] = out[target_col].astype(int)
    return out


def stratified_split(df: pd.DataFrame, target_col: str = "Response", test_size: float = 0.2, random_state: int = 42):
    train_df, holdout_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)
