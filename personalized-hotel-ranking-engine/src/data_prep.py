from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def load_expedia_data(
    raw_dir: Path,
    sample_frac: float | None = 0.03,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"train.csv not found under {raw_dir}. Run scripts/download_data.sh first."
        )

    train = pd.read_csv(train_path)
    if sample_frac is not None and 0 < sample_frac < 1:
        train = train.sample(frac=sample_frac, random_state=random_state).copy()

    return {"train": train}


def prepare_candidate_scoring_table(
    train_df: pd.DataFrame,
    negative_ratio: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Convert Expedia logs into supervised candidate-scoring rows."""
    required = ["hotel_cluster", "is_booking"]
    for col in required:
        if col not in train_df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = train_df.copy()
    if "srch_id" not in df.columns:
        # Older Expedia snapshots do not expose an explicit search id.
        # Build a deterministic query key for ranking-group evaluation.
        if "user_id" in df.columns and "date_time" in df.columns:
            dt = pd.to_datetime(df["date_time"], errors="coerce").astype(str).str[:10]
            key = df["user_id"].astype(str) + "_" + dt
        elif "user_id" in df.columns:
            key = df["user_id"].astype(str)
        else:
            key = df.index.astype(str)
        df["srch_id"] = pd.factorize(key)[0].astype(int)

    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    else:
        df["date_time"] = pd.NaT

    df["label"] = df["is_booking"].fillna(0).astype(int)

    sampled_parts = []
    rng = np.random.default_rng(random_state)
    for _, g in df.groupby("srch_id"):
        pos = g[g["label"] == 1]
        neg = g[g["label"] == 0]

        if pos.empty:
            n_neg = min(len(neg), 3)
            if n_neg > 0:
                sampled_parts.append(neg.sample(n=n_neg, random_state=random_state))
            continue

        n_neg = min(len(neg), max(1, negative_ratio * len(pos)))
        neg_sample = neg.sample(n=n_neg, random_state=random_state) if n_neg > 0 else neg
        sampled_parts.append(pd.concat([pos, neg_sample], axis=0))

    if not sampled_parts:
        return pd.DataFrame()

    sampled = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    keep_cols = [
        c
        for c in [
            "srch_id",
            "date_time",
            "site_name",
            "posa_continent",
            "user_location_country",
            "user_location_region",
            "user_location_city",
            "orig_destination_distance",
            "user_id",
            "is_mobile",
            "is_package",
            "channel",
            "srch_adults_cnt",
            "srch_children_cnt",
            "srch_rm_cnt",
            "srch_destination_id",
            "srch_destination_type_id",
            "hotel_continent",
            "hotel_country",
            "hotel_market",
            "hotel_cluster",
            "label",
        ]
        if c in sampled.columns
    ]

    return sampled[keep_cols].copy()


def query_time_split(
    df: pd.DataFrame,
    holdout_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()

    if "date_time" in work.columns and work["date_time"].notna().any():
        cutoff = work["date_time"].quantile(1 - holdout_frac)
        train = work[work["date_time"] < cutoff].copy()
        holdout = work[work["date_time"] >= cutoff].copy()
    else:
        srch_ids = work["srch_id"].drop_duplicates().sample(frac=1.0, random_state=random_state)
        n_holdout = int(len(srch_ids) * holdout_frac)
        holdout_ids = set(srch_ids.iloc[:n_holdout].tolist())
        holdout = work[work["srch_id"].isin(holdout_ids)].copy()
        train = work[~work["srch_id"].isin(holdout_ids)].copy()

    return train.reset_index(drop=True), holdout.reset_index(drop=True)


def build_reference_baselines(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.evaluation import evaluate_ranking_frame

    train = train_df.copy()
    holdout = holdout_df.copy()

    popularity = train.groupby("hotel_cluster")["label"].mean()
    holdout["score_popularity"] = holdout["hotel_cluster"].map(popularity).fillna(0.0)

    recent = train.copy()
    if recent["date_time"].notna().any():
        cutoff = recent["date_time"].max() - pd.Timedelta(days=30)
        recent = recent[recent["date_time"] >= cutoff]
    recent_pop = recent.groupby("hotel_cluster")["label"].mean() if not recent.empty else popularity
    holdout["score_recent_booking"] = holdout["hotel_cluster"].map(recent_pop).fillna(0.0)

    pop_metrics = evaluate_ranking_frame(holdout, score_col="score_popularity")
    rec_metrics = evaluate_ranking_frame(holdout, score_col="score_recent_booking")

    baseline_rows = pd.DataFrame(
        [
            {
                "model_name": "popularity_baseline",
                "library_source": "baseline",
                "map_at_5": pop_metrics["map_at_5"],
                "hit_rate_at_5": pop_metrics["hit_rate_at_5"],
                "latency_ms": np.nan,
                "interpretability_note": "Global popularity ranker",
            },
            {
                "model_name": "recent_booking_baseline",
                "library_source": "baseline",
                "map_at_5": rec_metrics["map_at_5"],
                "hit_rate_at_5": rec_metrics["hit_rate_at_5"],
                "latency_ms": np.nan,
                "interpretability_note": "Recent booking popularity ranker",
            },
        ]
    )
    return baseline_rows, holdout
