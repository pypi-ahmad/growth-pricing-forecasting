from __future__ import annotations

import numpy as np
import pandas as pd


def map_at_k(y_true_items: list[int], ranked_items: list[int], k: int = 5) -> float:
    if not y_true_items:
        return 0.0
    y_true_set = set(y_true_items)
    hits = 0
    score = 0.0
    for i, item in enumerate(ranked_items[:k], start=1):
        if item in y_true_set:
            hits += 1
            score += hits / i
    return float(score / min(len(y_true_set), k))


def hit_rate_at_k(y_true_items: list[int], ranked_items: list[int], k: int = 5) -> float:
    if not y_true_items:
        return 0.0
    return float(any(item in set(y_true_items) for item in ranked_items[:k]))


def evaluate_ranking_frame(
    df: pd.DataFrame,
    group_col: str = "srch_id",
    item_col: str = "hotel_cluster",
    label_col: str = "label",
    score_col: str = "score",
    k: int = 5,
) -> dict[str, float]:
    maps = []
    hits = []

    for _, g in df.groupby(group_col):
        true_items = g.loc[g[label_col] == 1, item_col].astype(int).tolist()
        ranked_items = g.sort_values(score_col, ascending=False)[item_col].astype(int).tolist()
        if not true_items:
            continue
        maps.append(map_at_k(true_items, ranked_items, k=k))
        hits.append(hit_rate_at_k(true_items, ranked_items, k=k))

    return {
        "map_at_5": float(np.mean(maps)) if maps else 0.0,
        "hit_rate_at_5": float(np.mean(hits)) if hits else 0.0,
    }


def build_leaderboard(
    project_name: str,
    baseline_rows: pd.DataFrame,
    lazy_results: pd.DataFrame,
    manual_results: pd.DataFrame,
    flaml_result: dict,
    pycaret_result: dict,
) -> pd.DataFrame:
    required = [
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

    rows = []

    for _, r in baseline_rows.iterrows():
        rows.append(
            {
                "project_name": project_name,
                "task_type": "ranking_candidate_scoring",
                "library_source": r["library_source"],
                "model_name": r["model_name"],
                "cv_metric_mean": np.nan,
                "cv_metric_std": np.nan,
                "holdout_primary_metric": r["map_at_5"],
                "holdout_secondary_metric": r["hit_rate_at_5"],
                "holdout_tertiary_metric": r["latency_ms"],
                "calibration_metric": np.nan,
                "train_time_sec": np.nan,
                "infer_latency_ms": r["latency_ms"],
                "p95_latency_ms": r["latency_ms"],
                "model_size_mb": np.nan,
                "retrain_time_sec": np.nan,
                "interpretability_note": r.get("interpretability_note", "baseline"),
            }
        )

    for _, r in lazy_results.iterrows():
        rows.append(
            {
                "project_name": project_name,
                "task_type": "ranking_candidate_scoring",
                "library_source": "lazypredict",
                "model_name": r["family"],
                "cv_metric_mean": np.nan,
                "cv_metric_std": np.nan,
                "holdout_primary_metric": r["map_at_5"],
                "holdout_secondary_metric": r["hit_rate_at_5"],
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
                "task_type": "ranking_candidate_scoring",
                "library_source": "manual",
                "model_name": r["model_name"],
                "cv_metric_mean": np.nan,
                "cv_metric_std": np.nan,
                "holdout_primary_metric": r["map_at_5"],
                "holdout_secondary_metric": r["hit_rate_at_5"],
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
            "task_type": "ranking_candidate_scoring",
            "library_source": "flaml",
            "model_name": flaml_result.get("model_name", "flaml_best"),
            "cv_metric_mean": np.nan,
            "cv_metric_std": np.nan,
            "holdout_primary_metric": flaml_result.get("map_at_5"),
            "holdout_secondary_metric": flaml_result.get("hit_rate_at_5"),
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
            "task_type": "ranking_candidate_scoring",
            "library_source": "pycaret",
            "model_name": pycaret_result.get("model_name", "pycaret_best"),
            "cv_metric_mean": np.nan,
            "cv_metric_std": np.nan,
            "holdout_primary_metric": pycaret_result.get("map_at_5"),
            "holdout_secondary_metric": pycaret_result.get("hit_rate_at_5"),
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
    latency = df["holdout_tertiary_metric"].fillna(df["holdout_tertiary_metric"].median(skipna=True) or 0)
    latency_norm = latency / (latency.max() if latency.max() else 1)

    df["rank_score"] = (
        0.68 * df["holdout_primary_metric"].fillna(0)
        + 0.27 * df["holdout_secondary_metric"].fillna(0)
        - 0.05 * latency_norm
    )
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df["final_rank"] = np.arange(1, len(df) + 1)

    for c in required:
        if c not in df.columns:
            df[c] = np.nan
    return df[required]
