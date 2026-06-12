from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "project_name",
    "task_type",
    "library_source",
    "model_name",
    "cv_metric_mean",
    "cv_metric_std",
    "holdout_primary_metric",
    "holdout_secondary_metric",
    "holdout_tertiary_metric",
    "calibration_metric",
    "train_time_sec",
    "infer_latency_ms",
    "p95_latency_ms",
    "model_size_mb",
    "retrain_time_sec",
    "interpretability_note",
    "rank_score",
    "final_rank",
]


def build_leaderboard(project_name: str, lazy_results: pd.DataFrame, manual_results: pd.DataFrame, flaml_result: dict, pycaret_result: dict):
    rows = []

    for _, r in lazy_results.iterrows():
        rows.append(
            {
                "project_name": project_name,
                "task_type": "regression_pricing",
                "library_source": "lazypredict",
                "model_name": r["family"],
                "cv_metric_mean": np.nan,
                "cv_metric_std": np.nan,
                "holdout_primary_metric": r["rmsle"],
                "holdout_secondary_metric": r["mae"],
                "holdout_tertiary_metric": r["p95_latency_ms"],
                "calibration_metric": np.nan,
                "train_time_sec": r["train_time_sec"],
                "infer_latency_ms": r["infer_latency_ms"],
                "p95_latency_ms": r["p95_latency_ms"],
                "model_size_mb": np.nan,
                "retrain_time_sec": r["train_time_sec"],
                "interpretability_note": r.get("eligibility_note", "lazy discovery"),
            }
        )

    for _, r in manual_results.iterrows():
        rows.append(
            {
                "project_name": project_name,
                "task_type": "regression_pricing",
                "library_source": "manual",
                "model_name": r["model_name"],
                "cv_metric_mean": np.nan,
                "cv_metric_std": np.nan,
                "holdout_primary_metric": r["rmsle"],
                "holdout_secondary_metric": r["mae"],
                "holdout_tertiary_metric": r["p95_latency_ms"],
                "calibration_metric": np.nan,
                "train_time_sec": r["train_time_sec"],
                "infer_latency_ms": r["infer_latency_ms"],
                "p95_latency_ms": r["p95_latency_ms"],
                "model_size_mb": np.nan,
                "retrain_time_sec": r["train_time_sec"],
                "interpretability_note": r.get("interpretability_note", "manual track"),
            }
        )

    rows.append(
        {
            "project_name": project_name,
            "task_type": "regression_pricing",
            "library_source": "flaml",
            "model_name": flaml_result.get("model_name", "flaml_best"),
            "cv_metric_mean": np.nan,
            "cv_metric_std": np.nan,
            "holdout_primary_metric": flaml_result.get("rmsle"),
            "holdout_secondary_metric": flaml_result.get("mae"),
            "holdout_tertiary_metric": flaml_result.get("p95_latency_ms"),
            "calibration_metric": np.nan,
            "train_time_sec": flaml_result.get("train_time_sec"),
            "infer_latency_ms": flaml_result.get("infer_latency_ms"),
            "p95_latency_ms": flaml_result.get("p95_latency_ms"),
            "model_size_mb": np.nan,
            "retrain_time_sec": flaml_result.get("train_time_sec"),
            "interpretability_note": flaml_result.get("interpretability_note", "flaml track"),
        }
    )

    rows.append(
        {
            "project_name": project_name,
            "task_type": "regression_pricing",
            "library_source": "pycaret",
            "model_name": pycaret_result.get("model_name", "pycaret_best"),
            "cv_metric_mean": np.nan,
            "cv_metric_std": np.nan,
            "holdout_primary_metric": pycaret_result.get("rmsle"),
            "holdout_secondary_metric": pycaret_result.get("mae"),
            "holdout_tertiary_metric": pycaret_result.get("p95_latency_ms"),
            "calibration_metric": np.nan,
            "train_time_sec": pycaret_result.get("train_time_sec"),
            "infer_latency_ms": pycaret_result.get("infer_latency_ms"),
            "p95_latency_ms": pycaret_result.get("p95_latency_ms"),
            "model_size_mb": np.nan,
            "retrain_time_sec": pycaret_result.get("train_time_sec"),
            "interpretability_note": pycaret_result.get("interpretability_note", "pycaret track"),
        }
    )

    df = pd.DataFrame(rows)

    # Lower is better for RMSLE/MAE/latency.
    rmsle_norm = df["holdout_primary_metric"].fillna(df["holdout_primary_metric"].max(skipna=True) or 1)
    mae_norm = df["holdout_secondary_metric"].fillna(df["holdout_secondary_metric"].max(skipna=True) or 1)
    lat_norm = df["holdout_tertiary_metric"].fillna(df["holdout_tertiary_metric"].median(skipna=True) or 0)

    rmsle_norm = rmsle_norm / (rmsle_norm.max() if rmsle_norm.max() else 1)
    mae_norm = mae_norm / (mae_norm.max() if mae_norm.max() else 1)
    lat_norm = lat_norm / (lat_norm.max() if lat_norm.max() else 1)

    df["rank_score"] = 1.0 - (0.62 * rmsle_norm + 0.28 * mae_norm + 0.10 * lat_norm)
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df["final_rank"] = np.arange(1, len(df) + 1)

    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[REQUIRED_COLUMNS]
