"""Backtesting utilities for rolling-origin forecast evaluation."""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SplitConfig:
    """Configuration for rolling-origin backtesting."""

    initial_train_size: int
    horizon: int
    step: int


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percentage terms."""
    denom = np.abs(y_true) + np.abs(y_pred)
    safe = np.where(denom == 0, 1.0, denom)
    return float(np.mean(200.0 * np.abs(y_true - y_pred) / safe))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calibration_bias_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Absolute relative bias in % (lower is better)."""
    mean_abs_true = float(np.mean(np.abs(y_true))) or 1.0
    bias = float(np.mean(y_pred - y_true))
    return abs(100.0 * bias / mean_abs_true)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute primary and secondary forecast metrics."""
    return {
        "sMAPE": smape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "calibration_metric": calibration_bias_metric(y_true, y_pred),
    }


def rolling_origin_splits(n_obs: int, config: SplitConfig) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Yield rolling-origin train/test indices."""
    train_end = config.initial_train_size
    fold = 0
    while train_end + config.horizon <= n_obs:
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, train_end + config.horizon)
        yield train_idx, test_idx, fold
        fold += 1
        train_end += config.step


def estimate_pickle_size_mb(model_obj: Any) -> float:
    """Estimate model size using pickle serialization."""
    if model_obj is None:
        return np.nan
    try:
        payload = pickle.dumps(model_obj)
        return float(len(payload) / (1024.0**2))
    except Exception:
        return np.nan


def _parse_model_output(raw_output: Any, horizon: int, elapsed_sec: float) -> dict[str, Any]:
    if isinstance(raw_output, dict):
        y_pred = np.asarray(raw_output.get("y_pred", []), dtype=float)
        fit_time_sec = float(raw_output.get("fit_time_sec", elapsed_sec))
        infer_latency_ms = float(
            raw_output.get("infer_latency_ms", (elapsed_sec * 1000.0) / max(horizon, 1))
        )
        model_obj = raw_output.get("model_object")
    else:
        y_pred = np.asarray(raw_output, dtype=float)
        fit_time_sec = elapsed_sec
        infer_latency_ms = (elapsed_sec * 1000.0) / max(horizon, 1)
        model_obj = None

    if y_pred.shape[0] != horizon:
        raise ValueError(f"Expected {horizon} predictions, got {y_pred.shape[0]}.")

    return {
        "y_pred": y_pred,
        "fit_time_sec": fit_time_sec,
        "infer_latency_ms": infer_latency_ms,
        "model_object": model_obj,
    }


def run_rolling_backtest(
    series: pd.Series,
    model_name: str,
    forecast_fn: Callable[[pd.Series, int, pd.DatetimeIndex], Any],
    config: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run rolling-origin backtest and return:
      1) point-level forecasts
      2) fold-level metric summary
    """
    records: list[dict[str, Any]] = []

    for train_idx, test_idx, fold in rolling_origin_splits(len(series), config):
        train_series = series.iloc[train_idx]
        test_series = series.iloc[test_idx]

        start = time.perf_counter()
        raw_output = forecast_fn(train_series, config.horizon, test_series.index)
        elapsed_sec = time.perf_counter() - start
        parsed = _parse_model_output(raw_output, config.horizon, elapsed_sec)

        for h, (ts, y_true, y_hat) in enumerate(
            zip(test_series.index, test_series.to_numpy(), parsed["y_pred"]), start=1
        ):
            records.append(
                {
                    "model_name": model_name,
                    "fold": fold,
                    "timestamp": ts,
                    "horizon_step": h,
                    "y_true": float(y_true),
                    "y_pred": float(y_hat),
                    "fit_time_sec": parsed["fit_time_sec"],
                    "infer_latency_ms": parsed["infer_latency_ms"],
                    "model_size_mb": estimate_pickle_size_mb(parsed["model_object"]),
                }
            )

    point_df = pd.DataFrame.from_records(records)
    if point_df.empty:
        return point_df, pd.DataFrame()

    fold_metrics = (
        point_df.groupby(["model_name", "fold"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "sMAPE": smape(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
                    "MAE": mae(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
                    "RMSE": rmse(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
                    "calibration_metric": calibration_bias_metric(
                        g["y_true"].to_numpy(), g["y_pred"].to_numpy()
                    ),
                    "fit_time_sec": g["fit_time_sec"].mean(),
                    "infer_latency_ms": g["infer_latency_ms"].mean(),
                    "model_size_mb": g["model_size_mb"].dropna().mean()
                    if g["model_size_mb"].notna().any()
                    else np.nan,
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    return point_df, fold_metrics


def horizon_band_metrics(
    point_df: pd.DataFrame,
    short_max: int = 6,
    medium_max: int = 18,
) -> pd.DataFrame:
    """Aggregate metrics by short/medium/long horizon buckets."""
    if point_df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "horizon_band",
                "sMAPE",
                "MAE",
                "RMSE",
                "calibration_metric",
                "n_points",
            ]
        )

    df = point_df.copy()
    df["horizon_band"] = np.select(
        [
            df["horizon_step"] <= short_max,
            (df["horizon_step"] > short_max) & (df["horizon_step"] <= medium_max),
        ],
        ["short", "medium"],
        default="long",
    )
    out = (
        df.groupby(["model_name", "horizon_band"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "sMAPE": smape(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
                    "MAE": mae(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
                    "RMSE": rmse(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
                    "calibration_metric": calibration_bias_metric(
                        g["y_true"].to_numpy(), g["y_pred"].to_numpy()
                    ),
                    "n_points": len(g),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    return out


def summarize_cv_metrics(fold_metrics_df: pd.DataFrame) -> tuple[float, float]:
    """Return mean/std sMAPE across folds."""
    if fold_metrics_df.empty:
        return np.nan, np.nan
    return (
        float(fold_metrics_df["sMAPE"].mean()),
        float(fold_metrics_df["sMAPE"].std(ddof=0)),
    )


def make_leaderboard_row(
    project_name: str,
    task_type: str,
    library_source: str,
    model_name: str,
    fold_metrics_df: pd.DataFrame,
    holdout_metrics: dict[str, float],
    interpretability_note: str,
) -> dict[str, Any]:
    """Create one row in the final leaderboard schema."""
    cv_mean, cv_std = summarize_cv_metrics(fold_metrics_df)
    return {
        "project_name": project_name,
        "task_type": task_type,
        "library_source": library_source,
        "model_name": model_name,
        "cv_metric_mean": cv_mean,
        "cv_metric_std": cv_std,
        "holdout_primary_metric": holdout_metrics.get("sMAPE", np.nan),
        "holdout_secondary_metric": holdout_metrics.get("MAE", np.nan),
        "holdout_tertiary_metric": holdout_metrics.get("RMSE", np.nan),
        "calibration_metric": holdout_metrics.get("calibration_metric", np.nan),
        "train_time_sec": float(fold_metrics_df["fit_time_sec"].mean())
        if not fold_metrics_df.empty
        else np.nan,
        "infer_latency_ms": float(fold_metrics_df["infer_latency_ms"].mean())
        if not fold_metrics_df.empty
        else np.nan,
        "model_size_mb": float(fold_metrics_df["model_size_mb"].mean())
        if not fold_metrics_df.empty and fold_metrics_df["model_size_mb"].notna().any()
        else np.nan,
        "interpretability_note": interpretability_note,
    }


def finalize_leaderboard(leaderboard_df: pd.DataFrame) -> pd.DataFrame:
    """Apply weighted ranking with sMAPE as primary metric."""
    if leaderboard_df.empty:
        leaderboard_df = leaderboard_df.copy()
        leaderboard_df["rank_score"] = np.nan
        leaderboard_df["final_rank"] = np.nan
        return leaderboard_df

    df = leaderboard_df.copy()
    primary_rank = df["holdout_primary_metric"].rank(method="dense", ascending=True)
    secondary_rank = df["holdout_secondary_metric"].rank(method="dense", ascending=True)
    tertiary_rank = df["holdout_tertiary_metric"].rank(method="dense", ascending=True)
    calibration_rank = df["calibration_metric"].rank(method="dense", ascending=True)

    df["rank_score"] = (
        0.55 * primary_rank
        + 0.20 * secondary_rank
        + 0.15 * tertiary_rank
        + 0.10 * calibration_rank
    )
    df["final_rank"] = df["rank_score"].rank(method="dense", ascending=True).astype(int)
    return df.sort_values(["final_rank", "holdout_primary_metric"], ascending=[True, True]).reset_index(
        drop=True
    )


def horizon_rank_table(horizon_df: pd.DataFrame) -> pd.DataFrame:
    """Build short/medium/long horizon ranking table by sMAPE."""
    if horizon_df.empty:
        return pd.DataFrame(columns=["model_name", "short_horizon_rank", "medium_horizon_rank", "long_horizon_rank"])

    pivot = horizon_df.pivot(index="model_name", columns="horizon_band", values="sMAPE")
    for band in ("short", "medium", "long"):
        if band not in pivot.columns:
            pivot[band] = np.nan

    ranked = pivot.copy()
    ranked["short_horizon_rank"] = ranked["short"].rank(method="dense", ascending=True)
    ranked["medium_horizon_rank"] = ranked["medium"].rank(method="dense", ascending=True)
    ranked["long_horizon_rank"] = ranked["long"].rank(method="dense", ascending=True)
    return ranked[
        ["short_horizon_rank", "medium_horizon_rank", "long_horizon_rank"]
    ].reset_index()


__all__ = [
    "SplitConfig",
    "finalize_leaderboard",
    "horizon_band_metrics",
    "horizon_rank_table",
    "calibration_bias_metric",
    "mae",
    "make_leaderboard_row",
    "metric_bundle",
    "rmse",
    "rolling_origin_splits",
    "run_rolling_backtest",
    "smape",
]
