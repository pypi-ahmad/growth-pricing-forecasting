from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


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


def precision_at_k(y_true, scores, k_ratio: float = 0.2) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = max(1, int(len(scores) * k_ratio))
    top_idx = np.argsort(-scores)[:k]
    return float(y_true[top_idx].mean())


def recall_at_threshold(y_true, scores, threshold: float) -> float:
    pred = (np.asarray(scores) >= threshold).astype(int)
    return float(recall_score(y_true, pred, zero_division=0))


def binary_metrics(y_true, scores, threshold: float = 0.5):
    pred = (np.asarray(scores) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision_at_20pct": precision_at_k(y_true, scores, k_ratio=0.2),
    }


def optimize_threshold(y_true, scores):
    best_thr = 0.5
    best_val = -1.0
    for thr in np.linspace(0.1, 0.9, 81):
        pred = (scores >= thr).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        utility = 0.7 * p + 0.3 * r
        if utility > best_val:
            best_thr = float(thr)
            best_val = float(utility)
    return best_thr, best_val


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


def make_estimator(family: str, random_state: int = 42):
    if family == "logistic_regression":
        return LogisticRegression(max_iter=2500, class_weight="balanced", random_state=random_state)
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=140,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=160,
            learning_rate=0.08,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "catboost" and CatBoostClassifier is not None:
        return CatBoostClassifier(
            depth=5,
            learning_rate=0.08,
            iterations=160,
            eval_metric="PRAUC",
            random_seed=random_state,
            verbose=False,
        )
    raise ValueError(f"Unsupported family: {family}")


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


def select_top3_eligible_families(lazy_table, X_train, y_train, X_holdout, y_holdout, random_state: int = 42):
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
        scores = model.predict_proba(X_holdout)[:, 1]

        m = binary_metrics(y_holdout, scores)
        thr, utility = optimize_threshold(y_holdout, scores)

        eligible = (m["pr_auc"] > 0) and (train_time < 900)
        rows.append(
            {
                "lazy_model": row["Model"],
                "family": family,
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "precision_at_20pct": m["precision_at_20pct"],
                "threshold": thr,
                "policy_utility": utility,
                "train_time_sec": train_time,
                "eligible": eligible,
                "eligibility_note": "eligible" if eligible else "filtered: quality/speed",
            }
        )

        if eligible:
            chosen.append(family)
        if len(chosen) >= 3:
            break

    out = pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc", "precision_at_20pct"], ascending=[False, False, False])
    return out.reset_index(drop=True), chosen[:3]


def run_manual_engineering_track(top3_families, X_train, y_train, X_holdout, y_holdout, random_state: int = 42):
    rows = []
    models: Dict[str, Any] = {}
    score_map: Dict[str, np.ndarray] = {}

    for family in top3_families:
        base = make_estimator(family, random_state=random_state)
        t0 = time.perf_counter()

        if family in {"logistic_regression"}:
            base.fit(X_train, y_train)
            model = base
        else:
            model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            model.fit(X_train, y_train)

        train_time = time.perf_counter() - t0
        scores = model.predict_proba(X_holdout)[:, 1]
        thr, utility = optimize_threshold(y_holdout, scores)
        m = binary_metrics(y_holdout, scores, threshold=thr)

        mean_latency, p95_latency = _measure_latency(model, X_holdout)

        rows.append(
            {
                "model_name": family,
                "library_source": "manual",
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "precision_at_20pct": m["precision_at_20pct"],
                "recall_at_operating": m["recall"],
                "threshold": thr,
                "policy_utility": utility,
                "train_time_sec": train_time,
                "infer_latency_ms": mean_latency,
                "p95_latency_ms": p95_latency,
                "interpretability_note": f"Manual calibrated model: {family}",
            }
        )

        models[family] = model
        score_map[family] = scores

    out = pd.DataFrame(rows).sort_values(["pr_auc", "precision_at_20pct"], ascending=[False, False]).reset_index(drop=True)
    return out, models, score_map


def run_flaml_track(X_train, y_train, X_holdout, y_holdout, time_budget: int = 120, random_state: int = 42):
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

    scores = automl.predict_proba(X_holdout)[:, 1]
    thr, utility = optimize_threshold(y_holdout, scores)
    m = binary_metrics(y_holdout, scores, threshold=thr)

    return {
        "model_name": str(automl.best_estimator),
        "library_source": "flaml",
        "pr_auc": m["pr_auc"],
        "roc_auc": m["roc_auc"],
        "precision_at_20pct": m["precision_at_20pct"],
        "recall_at_operating": m["recall"],
        "threshold": thr,
        "policy_utility": utility,
        "train_time_sec": train_time,
        "infer_latency_ms": np.nan,
        "p95_latency_ms": np.nan,
        "interpretability_note": "FLAML budget-aware targeting challenger",
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
        "scores": scores,
    }


def run_pycaret_track(train_df: pd.DataFrame, holdout_df: pd.DataFrame, target_col: str, session_id: int, model_output_path: Path):
    try:
        import sys

        original_version_info = sys.version_info
        try:
            if tuple(sys.version_info) >= (3, 12):
                sys.version_info = (3, 11, 9, "final", 0)
            from pycaret.classification import calibrate_model, compare_models, create_model, finalize_model, models as pycaret_models, predict_model, save_model, setup, tune_model
        finally:
            sys.version_info = original_version_info

        if len(train_df) > 7000:
            train_df = train_df.sample(n=min(8000, len(train_df)), random_state=session_id).reset_index(drop=True)

        setup(
            data=train_df,
            target=target_col,
            session_id=session_id,
            fold=3,
            fold_strategy="stratifiedkfold",
            preprocess=True,
            normalize=True,
            remove_multicollinearity=True,
            multicollinearity_threshold=0.95,
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
        try:
            calibrated = calibrate_model(tuned)
        except Exception:
            calibrated = tuned
        final_model = finalize_model(calibrated)

        pred = predict_model(final_model, data=holdout_df.copy())
        score_col = "prediction_score" if "prediction_score" in pred.columns else "Score"
        scores = pred[score_col].astype(float).values

        thr, utility = optimize_threshold(holdout_df[target_col].astype(int).values, scores)
        m = binary_metrics(holdout_df[target_col].astype(int).values, scores, threshold=thr)

        save_model(final_model, str(model_output_path))

        return {
            "model_name": type(final_model).__name__,
            "library_source": "pycaret",
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "precision_at_20pct": m["precision_at_20pct"],
            "recall_at_operating": m["recall"],
            "threshold": thr,
            "policy_utility": utility,
            "train_time_sec": np.nan,
            "infer_latency_ms": np.nan,
            "p95_latency_ms": np.nan,
            "interpretability_note": "PyCaret tuned+calibrated finalized model",
            "scores": scores,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "model_name": "pycaret_failed",
            "library_source": "pycaret",
            "pr_auc": np.nan,
            "roc_auc": np.nan,
            "precision_at_20pct": np.nan,
            "recall_at_operating": np.nan,
            "threshold": np.nan,
            "policy_utility": np.nan,
            "train_time_sec": np.nan,
            "infer_latency_ms": np.nan,
            "p95_latency_ms": np.nan,
            "interpretability_note": f"PyCaret unavailable or failed: {exc}",
            "scores": np.array([]),
            "status": "failed",
        }
