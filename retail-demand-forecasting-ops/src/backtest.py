from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.forecast_models import regression_metrics


def rolling_origin_backtest(
    df: pd.DataFrame,
    predictor_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    min_train_days: int = 180,
    horizon_days: int = 14,
    step_days: int = 14,
):
    data = df.sort_values("Date").copy()
    unique_dates = sorted(data["Date"].unique())

    rows = []
    start_idx = min_train_days
    while start_idx + horizon_days < len(unique_dates):
        train_end = unique_dates[start_idx]
        horizon_end = unique_dates[start_idx + horizon_days]

        train_slice = data[data["Date"] <= train_end].copy()
        valid_slice = data[(data["Date"] > train_end) & (data["Date"] <= horizon_end)].copy()

        if valid_slice.empty:
            start_idx += step_days
            continue

        preds = predictor_fn(train_slice, valid_slice)
        m = regression_metrics(valid_slice["Sales"], preds)
        rows.append(
            {
                "train_end": train_end,
                "valid_end": horizon_end,
                "smape": m["smape"],
                "mae": m["mae"],
                "rmse": m["rmse"],
            }
        )
        start_idx += step_days

    return pd.DataFrame(rows)


def summarize_backtest(backtest_df: pd.DataFrame) -> pd.DataFrame:
    if backtest_df.empty:
        return pd.DataFrame(
            [{"metric": "smape", "mean": np.nan, "std": np.nan}, {"metric": "mae", "mean": np.nan, "std": np.nan}, {"metric": "rmse", "mean": np.nan, "std": np.nan}]
        )

    rows = []
    for metric in ["smape", "mae", "rmse"]:
        rows.append(
            {
                "metric": metric,
                "mean": float(backtest_df[metric].mean()),
                "std": float(backtest_df[metric].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)
