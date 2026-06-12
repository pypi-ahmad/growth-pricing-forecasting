from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation import evaluate_ranking_frame


try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


FAMILY_ALIASES = {
    "LogisticRegression": "logistic_regression",
    "RandomForestClassifier": "random_forest",
    "ExtraTreesClassifier": "extra_trees",
    "XGBClassifier": "xgboost",
    "LGBMClassifier": "lightgbm",
    "CatBoostClassifier": "catboost",
}


def make_estimator(family: str, random_state: int = 42):
    if family == "logistic_regression":
        return LogisticRegression(max_iter=2500, class_weight="balanced", random_state=random_state)
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=16,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=650,
            max_depth=18,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=800,
            learning_rate=0.04,
            num_leaves=63,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "catboost" and CatBoostClassifier is not None:
        return CatBoostClassifier(
            depth=8,
            learning_rate=0.04,
            iterations=900,
            eval_metric="PRAUC",
            random_seed=random_state,
            verbose=False,
        )
    raise ValueError(f"Unsupported family: {family}")


def _measure_latency(model, X, n_rows: int = 300):
    n = min(n_rows, X.shape[0])
    if n == 0:
        return np.nan, np.nan
    times = []
    for i in range(n):
        row = X[i : i + 1]
        t0 = time.perf_counter()
        _ = model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return float(arr.mean()), float(np.percentile(arr, 95))


def _evaluate_ranking(model, X_holdout, holdout_meta: pd.DataFrame):
    scores = model.predict_proba(X_holdout)[:, 1]
    scored = holdout_meta.copy()
    scored["score"] = scores
    metrics = evaluate_ranking_frame(scored, score_col="score")
    return metrics, scored


def run_lazypredict_discovery(X_train, X_holdout, y_train, y_holdout):
    from lazypredict.Supervised import LazyClassifier

    train_n = min(3000, X_train.shape[0])
    hold_n = min(1200, X_holdout.shape[0])
    if train_n < X_train.shape[0]:
        idx = np.random.default_rng(42).choice(X_train.shape[0], size=train_n, replace=False)
        X_train_lazy = X_train[idx]
        y_train_lazy = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]
    else:
        X_train_lazy, y_train_lazy = X_train, y_train

    if hold_n < X_holdout.shape[0]:
        idxh = np.random.default_rng(43).choice(X_holdout.shape[0], size=hold_n, replace=False)
        X_holdout_lazy = X_holdout[idxh]
        y_holdout_lazy = y_holdout.iloc[idxh] if hasattr(y_holdout, "iloc") else y_holdout[idxh]
    else:
        X_holdout_lazy, y_holdout_lazy = X_holdout, y_holdout

    lazy = LazyClassifier(verbose=0, ignore_warnings=True)
    models, _ = lazy.fit(X_train_lazy, X_holdout_lazy, y_train_lazy, y_holdout_lazy)
    table = models.reset_index().rename(columns={"index": "Model"})
    if "Model" not in table.columns:
        table = table.rename(columns={table.columns[0]: "Model"})
    keep = [c for c in ["Model", "Accuracy", "Balanced Accuracy", "ROC AUC", "F1 Score", "Time Taken"] if c in table.columns]
    return table[keep]


def select_top3_eligible_families(lazy_table, X_train, y_train, X_holdout, holdout_meta, random_state: int = 42):
    order_col = "ROC AUC" if "ROC AUC" in lazy_table.columns else lazy_table.columns[1]
    ranked = lazy_table.sort_values(order_col, ascending=False)

    rows = []
    chosen = []

    for _, row in ranked.iterrows():
        family = FAMILY_ALIASES.get(row["Model"])
        if family is None or family in chosen:
            continue
        try:
            model = make_estimator(family, random_state=random_state)
        except Exception:
            continue

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        metrics, _ = _evaluate_ranking(model, X_holdout, holdout_meta)
        mean_latency, p95_latency = _measure_latency(model, X_holdout)

        eligible = (metrics["map_at_5"] > 0) and (train_time < 1200)
        rows.append(
            {
                "lazy_model": row["Model"],
                "family": family,
                "map_at_5": metrics["map_at_5"],
                "hit_rate_at_5": metrics["hit_rate_at_5"],
                "train_time_sec": train_time,
                "infer_latency_ms": mean_latency,
                "p95_latency_ms": p95_latency,
                "eligible": eligible,
                "eligibility_note": "eligible" if eligible else "filtered: ranking quality/speed",
            }
        )
        if eligible:
            chosen.append(family)
        if len(chosen) >= 3:
            break

    out = pd.DataFrame(rows).sort_values(["map_at_5", "hit_rate_at_5"], ascending=[False, False])
    return out.reset_index(drop=True), chosen[:3]


def run_manual_engineering_track(top3_families, X_train, y_train, X_holdout, holdout_meta, random_state: int = 42):
    rows = []
    models: Dict[str, Any] = {}
    scored_frames: Dict[str, pd.DataFrame] = {}

    for family in top3_families:
        model = make_estimator(family, random_state=random_state)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        metrics, scored = _evaluate_ranking(model, X_holdout, holdout_meta)
        mean_latency, p95_latency = _measure_latency(model, X_holdout)

        rows.append(
            {
                "model_name": family,
                "library_source": "manual",
                "map_at_5": metrics["map_at_5"],
                "hit_rate_at_5": metrics["hit_rate_at_5"],
                "train_time_sec": train_time,
                "infer_latency_ms": mean_latency,
                "p95_latency_ms": p95_latency,
                "interpretability_note": f"Manual candidate-scoring model: {family}",
            }
        )
        models[family] = model
        scored_frames[family] = scored

    result = pd.DataFrame(rows).sort_values(["map_at_5", "hit_rate_at_5"], ascending=[False, False]).reset_index(drop=True)
    return result, models, scored_frames


def run_flaml_track(X_train, y_train, X_holdout, holdout_meta, time_budget: int = 120, random_state: int = 42):
    from flaml import AutoML

    automl = AutoML()
    t0 = time.perf_counter()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="classification",
        metric="ap",
        time_budget=time_budget,
        eval_method="cv",
        n_splits=3,
        estimator_list=["lgbm", "xgboost", "rf", "extra_tree", "lrl1"],
        seed=random_state,
    )
    train_time = time.perf_counter() - t0

    model = automl.model.estimator
    scores = automl.predict_proba(X_holdout)[:, 1]
    scored = holdout_meta.copy()
    scored["score"] = scores
    metrics = evaluate_ranking_frame(scored)

    mean_latency, p95_latency = _measure_latency(model, X_holdout)

    return {
        "model_name": str(automl.best_estimator),
        "library_source": "flaml",
        "map_at_5": metrics["map_at_5"],
        "hit_rate_at_5": metrics["hit_rate_at_5"],
        "train_time_sec": train_time,
        "infer_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "interpretability_note": "FLAML challenger for ranking candidate scoring",
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
        "scored": scored,
    }


def run_pycaret_track(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str,
    session_id: int,
    model_output_path: Path,
):
    try:
        import sys

        original_version_info = sys.version_info
        try:
            if tuple(sys.version_info) >= (3, 12):
                sys.version_info = (3, 11, 9, "final", 0)
            from pycaret.classification import compare_models, create_model, finalize_model, models as pycaret_models, predict_model, save_model, setup, tune_model
        finally:
            sys.version_info = original_version_info

        if len(train_df) > 12000:
            train_df = train_df.sample(n=min(12000, len(train_df)), random_state=session_id).reset_index(drop=True)

        setup(
            data=train_df,
            target=target_col,
            session_id=session_id,
            fold=3,
            fold_strategy="stratifiedkfold",
            preprocess=True,
            normalize=True,
            imputation_type="simple",
            numeric_imputation="median",
            categorical_imputation="most_frequent",
            html=False,
            n_jobs=1,
            verbose=False,
        )
        available_ids = set(pycaret_models().index.tolist())
        preferred_ids = ["lr", "rf"]
        include_ids = [m for m in preferred_ids if m in available_ids]

        if include_ids:
            best = compare_models(sort="AUC", include=include_ids)
        else:
            best = compare_models(sort="AUC")
        if isinstance(best, list):
            if len(best) == 0:
                fallback_id = include_ids[0] if include_ids else "lr"
                best = create_model(fallback_id)
            else:
                best = best[0]

        try:
            tuned = tune_model(best, optimize="AUC")
        except Exception:
            tuned = best
        final_model = finalize_model(tuned)

        pred = predict_model(final_model, data=holdout_df.copy())
        score_col = "prediction_score" if "prediction_score" in pred.columns else "Score"
        pred["score"] = pred[score_col].astype(float)

        scored = pred[["srch_id", "hotel_cluster", target_col, "score"]].rename(columns={target_col: "label"})
        metrics = evaluate_ranking_frame(scored)

        save_model(final_model, str(model_output_path))

        return {
            "model_name": type(final_model).__name__,
            "library_source": "pycaret",
            "map_at_5": metrics["map_at_5"],
            "hit_rate_at_5": metrics["hit_rate_at_5"],
            "train_time_sec": np.nan,
            "infer_latency_ms": np.nan,
            "p95_latency_ms": np.nan,
            "interpretability_note": "PyCaret tuned finalized ranking scorer",
            "scored": scored,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "model_name": "pycaret_failed",
            "library_source": "pycaret",
            "map_at_5": np.nan,
            "hit_rate_at_5": np.nan,
            "train_time_sec": np.nan,
            "infer_latency_ms": np.nan,
            "p95_latency_ms": np.nan,
            "interpretability_note": f"PyCaret unavailable or failed: {exc}",
            "scored": pd.DataFrame(),
            "status": "failed",
        }


def build_topk_recommendations(scored_df: pd.DataFrame, group_col: str = "srch_id", item_col: str = "hotel_cluster", score_col: str = "score", k: int = 5):
    rows = []
    for gid, g in scored_df.groupby(group_col):
        top = g.sort_values(score_col, ascending=False).head(k)
        rows.append(
            {
                group_col: gid,
                "top_k_items": top[item_col].astype(int).tolist(),
                "top_k_scores": [float(x) for x in top[score_col].tolist()],
            }
        )
    return pd.DataFrame(rows)


def save_inference_bundle(model, preprocessor, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "ranking_model.joblib")
    joblib.dump(preprocessor, output_dir / "ranking_preprocessor.joblib")
