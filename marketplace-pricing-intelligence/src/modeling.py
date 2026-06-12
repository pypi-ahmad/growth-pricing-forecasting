from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error


try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


def rmsle(y_true_price, y_pred_price):
    y_true = np.asarray(y_true_price)
    y_pred = np.clip(np.asarray(y_pred_price), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def pricing_metrics(y_true_price, y_pred_price):
    y_pred = np.clip(np.asarray(y_pred_price), 0, None)
    return {
        "rmsle": rmsle(y_true_price, y_pred),
        "mae": float(mean_absolute_error(y_true_price, y_pred)),
    }


def _measure_latency(model, X, n_rows: int = 300):
    n = min(n_rows, X.shape[0])
    if n == 0:
        return np.nan, np.nan
    times = []
    for i in range(n):
        row = X[i : i + 1]
        t0 = time.perf_counter()
        _ = model.predict(row)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return float(arr.mean()), float(np.percentile(arr, 95))


FAMILY_ALIASES = {
    "Ridge": "ridge",
    "ElasticNet": "elastic_net",
    "RandomForestRegressor": "random_forest",
    "XGBRegressor": "xgboost",
    "LGBMRegressor": "lightgbm",
}


def make_estimator(family: str, random_state: int = 42):
    if family == "ridge":
        return Ridge(alpha=3.0, random_state=random_state)
    if family == "elastic_net":
        return ElasticNet(alpha=0.0007, l1_ratio=0.15, random_state=random_state, max_iter=5000)
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=22,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "xgboost" and XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=900,
            learning_rate=0.04,
            max_depth=9,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "lightgbm" and LGBMRegressor is not None:
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.04,
            num_leaves=63,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported family: {family}")


def run_lazypredict_discovery(X_train_lazy, X_holdout_lazy, y_train_log, y_holdout_price):
    from lazypredict.Supervised import LazyRegressor

    train_n = min(3000, X_train_lazy.shape[0])
    hold_n = min(1200, X_holdout_lazy.shape[0])
    if train_n < X_train_lazy.shape[0]:
        idx = np.random.default_rng(42).choice(X_train_lazy.shape[0], size=train_n, replace=False)
        X_train_lp = X_train_lazy[idx]
        y_train_lp = y_train_log[idx]
    else:
        X_train_lp, y_train_lp = X_train_lazy, y_train_log

    if hold_n < X_holdout_lazy.shape[0]:
        idxh = np.random.default_rng(43).choice(X_holdout_lazy.shape[0], size=hold_n, replace=False)
        X_holdout_lp = X_holdout_lazy[idxh]
        y_holdout_price_lp = y_holdout_price[idxh]
    else:
        X_holdout_lp, y_holdout_price_lp = X_holdout_lazy, y_holdout_price

    reg = LazyRegressor(verbose=0, ignore_warnings=True)
    y_holdout_log = np.log1p(y_holdout_price_lp)
    models, _ = reg.fit(X_train_lp, X_holdout_lp, y_train_lp, y_holdout_log)
    table = models.reset_index().rename(columns={"index": "Model"})
    if "Model" not in table.columns:
        table = table.rename(columns={table.columns[0]: "Model"})

    keep = [c for c in ["Model", "R-Squared", "RMSE", "Time Taken"] if c in table.columns]
    return table[keep]


def select_top3_eligible_families(lazy_table, X_train_lazy, y_train_log, X_holdout_lazy, y_holdout_price, random_state: int = 42):
    ranked = lazy_table.sort_values("RMSE", ascending=True) if "RMSE" in lazy_table.columns else lazy_table.copy()
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
        model.fit(X_train_lazy, y_train_log)
        train_time = time.perf_counter() - t0

        pred_price = np.expm1(model.predict(X_holdout_lazy)).clip(min=0)
        m = pricing_metrics(y_holdout_price, pred_price)
        mean_latency, p95_latency = _measure_latency(model, X_holdout_lazy)

        eligible = (m["rmsle"] < 1.5) and (train_time < 1500)
        rows.append(
            {
                "lazy_model": row["Model"],
                "family": family,
                "rmsle": m["rmsle"],
                "mae": m["mae"],
                "train_time_sec": train_time,
                "infer_latency_ms": mean_latency,
                "p95_latency_ms": p95_latency,
                "eligible": eligible,
                "eligibility_note": "eligible" if eligible else "filtered: quality/speed",
            }
        )
        if eligible:
            chosen.append(family)
        if len(chosen) >= 3:
            break

    out = pd.DataFrame(rows).sort_values(["rmsle", "mae", "p95_latency_ms"], ascending=[True, True, True])
    return out.reset_index(drop=True), chosen[:3]


def run_manual_engineering_track(top3_families, X_train_full, y_train_log, X_holdout_full, y_holdout_price, holdout_meta, random_state: int = 42):
    rows = []
    models: Dict[str, Any] = {}
    preds_map: Dict[str, np.ndarray] = {}

    for family in top3_families:
        model = make_estimator(family, random_state=random_state)

        t0 = time.perf_counter()
        model.fit(X_train_full, y_train_log)
        train_time = time.perf_counter() - t0

        pred_price = np.expm1(model.predict(X_holdout_full)).clip(min=0)
        m = pricing_metrics(y_holdout_price, pred_price)
        mean_latency, p95_latency = _measure_latency(model, X_holdout_full)

        rows.append(
            {
                "model_name": family,
                "library_source": "manual",
                "rmsle": m["rmsle"],
                "mae": m["mae"],
                "train_time_sec": train_time,
                "infer_latency_ms": mean_latency,
                "p95_latency_ms": p95_latency,
                "interpretability_note": f"Manual text+structured model: {family}",
            }
        )

        models[family] = model
        preds_map[family] = pred_price

    result = pd.DataFrame(rows).sort_values(["rmsle", "mae"], ascending=[True, True]).reset_index(drop=True)

    return result, models, preds_map


def run_flaml_track(X_train_full, y_train_log, X_holdout_full, y_holdout_price, time_budget: int = 150, random_state: int = 42):
    from flaml import AutoML

    automl = AutoML()
    t0 = time.perf_counter()
    automl.fit(
        X_train=X_train_full,
        y_train=y_train_log,
        task="regression",
        metric="rmse",
        time_budget=time_budget,
        eval_method="cv",
        n_splits=3,
        estimator_list=["lgbm", "xgboost", "rf", "extra_tree", "xgb_limitdepth"],
        seed=random_state,
    )
    train_time = time.perf_counter() - t0

    pred_price = np.expm1(automl.predict(X_holdout_full)).clip(min=0)
    m = pricing_metrics(y_holdout_price, pred_price)

    return {
        "model_name": str(automl.best_estimator),
        "library_source": "flaml",
        "rmsle": m["rmsle"],
        "mae": m["mae"],
        "train_time_sec": train_time,
        "infer_latency_ms": np.nan,
        "p95_latency_ms": np.nan,
        "interpretability_note": "FLAML budget-aware pricing challenger",
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
        "predictions": pred_price,
    }


def run_pycaret_track(train_table: pd.DataFrame, holdout_table: pd.DataFrame, session_id: int, model_output_path: Path):
    try:
        import sys

        original_version_info = sys.version_info
        try:
            if tuple(sys.version_info) >= (3, 12):
                sys.version_info = (3, 11, 9, "final", 0)
            from pycaret.regression import compare_models, create_model, finalize_model, models as pycaret_models, predict_model, save_model, setup, tune_model
        finally:
            sys.version_info = original_version_info

        if len(train_table) > 12000:
            train_table = train_table.sample(n=min(12000, len(train_table)), random_state=session_id).reset_index(drop=True)

        setup(
            data=train_table,
            target="log_price",
            session_id=session_id,
            fold=5,
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
        preferred_ids = ["lr", "ridge", "rf"]
        include_ids = [m for m in preferred_ids if m in available_ids]

        if include_ids:
            best = compare_models(sort="RMSE", include=include_ids)
        else:
            best = compare_models(sort="RMSE")
        if isinstance(best, list):
            if len(best) == 0:
                fallback_id = include_ids[0] if include_ids else "lr"
                best = create_model(fallback_id)
            else:
                best = best[0]

        try:
            tuned = tune_model(best, optimize="RMSE")
        except Exception:
            tuned = best
        final_model = finalize_model(tuned)

        pred = predict_model(final_model, data=holdout_table.copy())
        label_col = "prediction_label" if "prediction_label" in pred.columns else "Label"
        pred_price = np.expm1(pred[label_col].astype(float).values).clip(min=0)

        y_holdout_price = np.expm1(holdout_table["log_price"].astype(float).values)
        m = pricing_metrics(y_holdout_price, pred_price)

        save_model(final_model, str(model_output_path))

        return {
            "model_name": type(final_model).__name__,
            "library_source": "pycaret",
            "rmsle": m["rmsle"],
            "mae": m["mae"],
            "train_time_sec": np.nan,
            "infer_latency_ms": np.nan,
            "p95_latency_ms": np.nan,
            "interpretability_note": "PyCaret tuned finalized pricing model",
            "predictions": pred_price,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "model_name": "pycaret_failed",
            "library_source": "pycaret",
            "rmsle": np.nan,
            "mae": np.nan,
            "train_time_sec": np.nan,
            "infer_latency_ms": np.nan,
            "p95_latency_ms": np.nan,
            "interpretability_note": f"PyCaret unavailable or failed: {exc}",
            "predictions": np.array([]),
            "status": "failed",
        }


def category_error_analysis(holdout_meta: pd.DataFrame, preds_price) -> pd.DataFrame:
    df = holdout_meta.copy()
    df["pred_price"] = preds_price
    df["abs_error"] = (df["pred_price"] - df["price"]).abs()

    out = (
        df.groupby("category_name")
        .agg(
            n_items=("price", "size"),
            mae=("abs_error", "mean"),
            median_price=("price", "median"),
        )
        .sort_values("mae", ascending=False)
        .reset_index()
    )
    return out


def seller_pricing_logic(pred_price: float, global_mae: float):
    low = max(0.0, pred_price - 0.8 * global_mae)
    high = pred_price + 0.8 * global_mae
    return {"suggested_price": float(pred_price), "suggested_low": float(low), "suggested_high": float(high)}


def save_inference_bundle(model, preprocessor, residuals, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "pricing_model.joblib")
    joblib.dump(preprocessor, output_dir / "pricing_preprocessor.joblib")

    q = np.quantile(np.abs(residuals), [0.5, 0.9]).tolist() if len(residuals) else [5.0, 12.0]
    meta = {"abs_error_p50": float(q[0]), "abs_error_p90": float(q[1])}
    (output_dir / "pricing_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
