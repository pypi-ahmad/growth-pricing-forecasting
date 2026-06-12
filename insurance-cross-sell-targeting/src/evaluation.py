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
                "task_type": "binary_classification_targeting",
                "library_source": "lazypredict",
                "model_name": r["family"],
                "cv_metric_mean": np.nan,
                "cv_metric_std": np.nan,
                "holdout_primary_metric": r["pr_auc"],
                "holdout_secondary_metric": r["roc_auc"],
                "holdout_tertiary_metric": r["precision_at_20pct"],
                "calibration_metric": np.nan,
                "train_time_sec": r["train_time_sec"],
                "infer_latency_ms": np.nan,
                "p95_latency_ms": np.nan,
                "model_size_mb": np.nan,
                "retrain_time_sec": r["train_time_sec"],
                "interpretability_note": r.get("eligibility_note", "lazy discovery"),
            }
        )

    for _, r in manual_results.iterrows():
        rows.append(
            {
                "project_name": project_name,
                "task_type": "binary_classification_targeting",
                "library_source": "manual",
                "model_name": r["model_name"],
                "cv_metric_mean": np.nan,
                "cv_metric_std": np.nan,
                "holdout_primary_metric": r["pr_auc"],
                "holdout_secondary_metric": r["roc_auc"],
                "holdout_tertiary_metric": r["precision_at_20pct"],
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
            "task_type": "binary_classification_targeting",
            "library_source": "flaml",
            "model_name": flaml_result.get("model_name", "flaml_best"),
            "cv_metric_mean": np.nan,
            "cv_metric_std": np.nan,
            "holdout_primary_metric": flaml_result.get("pr_auc"),
            "holdout_secondary_metric": flaml_result.get("roc_auc"),
            "holdout_tertiary_metric": flaml_result.get("precision_at_20pct"),
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
            "task_type": "binary_classification_targeting",
            "library_source": "pycaret",
            "model_name": pycaret_result.get("model_name", "pycaret_best"),
            "cv_metric_mean": np.nan,
            "cv_metric_std": np.nan,
            "holdout_primary_metric": pycaret_result.get("pr_auc"),
            "holdout_secondary_metric": pycaret_result.get("roc_auc"),
            "holdout_tertiary_metric": pycaret_result.get("precision_at_20pct"),
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
    df["rank_score"] = (
        0.55 * df["holdout_primary_metric"].fillna(0)
        + 0.25 * df["holdout_secondary_metric"].fillna(0)
        + 0.20 * df["holdout_tertiary_metric"].fillna(0)
    )
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df["final_rank"] = np.arange(1, len(df) + 1)

    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[REQUIRED_COLUMNS]
