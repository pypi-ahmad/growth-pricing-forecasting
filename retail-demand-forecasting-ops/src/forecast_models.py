from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "smape": smape(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def run_classical_baselines(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    hold = holdout_df.copy()

    global_mean = train["Sales"].mean()

    # Naive baseline: last observed sales by store.
    last_sales = train.sort_values("Date").groupby("Store")["Sales"].last()
    hold["pred_naive"] = hold["Store"].map(last_sales).fillna(global_mean)

    # Seasonal naive baseline: mean by store x dayofweek.
    store_dow = train.groupby(["Store", "dayofweek"])["Sales"].mean()
    hold["pred_seasonal_naive"] = [
        store_dow.get((s, d), global_mean) for s, d in zip(hold["Store"], hold["dayofweek"])
    ]

    # SARIMA reference on aggregate daily sales.
    daily = train.groupby("Date")["Sales"].sum().asfreq("D").fillna(0)
    sarima_ok = True
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        sarima = SARIMAX(
            daily,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarima_fit = sarima.fit(disp=False)
        forecast_dates = sorted(hold["Date"].unique())
        sarima_fore = sarima_fit.forecast(steps=len(forecast_dates))
        date_to_pred = dict(zip(forecast_dates, sarima_fore))
    except Exception:
        sarima_ok = False
        date_to_pred = {d: global_mean * hold["Store"].nunique() for d in sorted(hold["Date"].unique())}

    store_share = train.groupby("Store")["Sales"].mean()
    store_share = store_share / store_share.sum()
    hold["pred_sarima"] = [
        date_to_pred.get(d, global_mean * hold["Store"].nunique()) * store_share.get(s, 1 / max(len(store_share), 1))
        for s, d in zip(hold["Store"], hold["Date"])
    ]

    # Prophet reference on aggregate daily sales.
    prophet_ok = True
    try:
        from prophet import Prophet

        prophet_df = daily.reset_index().rename(columns={"Date": "ds", "Sales": "y"})
        model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=len(sorted(hold["Date"].unique())), freq="D")
        fcst = model.predict(future)
        pred_series = fcst.tail(len(sorted(hold["Date"].unique())))[["ds", "yhat"]]
        date_to_pred_prophet = dict(zip(pred_series["ds"], pred_series["yhat"]))
    except Exception:
        prophet_ok = False
        date_to_pred_prophet = {d: global_mean * hold["Store"].nunique() for d in sorted(hold["Date"].unique())}

    hold["pred_prophet"] = [
        date_to_pred_prophet.get(d, global_mean * hold["Store"].nunique()) * store_share.get(s, 1 / max(len(store_share), 1))
        for s, d in zip(hold["Store"], hold["Date"])
    ]

    rows = []
    for name, col, note in [
        ("naive", "pred_naive", "Last observed sales per store"),
        ("seasonal_naive", "pred_seasonal_naive", "Store/day-of-week seasonal baseline"),
        ("sarima", "pred_sarima", "Aggregate SARIMA reference" if sarima_ok else "SARIMA fallback used"),
        ("prophet", "pred_prophet", "Aggregate Prophet reference" if prophet_ok else "Prophet fallback used"),
    ]:
        m = regression_metrics(hold["Sales"], hold[col])
        rows.append(
            {
                "model_name": name,
                "library_source": "classical_baseline",
                "smape": m["smape"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "interpretability_note": note,
            }
        )

    return pd.DataFrame(rows), hold


FAMILY_ALIASES = {
    "LinearRegression": "linear_regression",
    "Ridge": "ridge",
    "ElasticNet": "elastic_net",
    "RandomForestRegressor": "random_forest",
    "ExtraTreesRegressor": "extra_trees",
    "XGBRegressor": "xgboost",
    "LGBMRegressor": "lightgbm",
    "CatBoostRegressor": "catboost",
}


def make_estimator(family: str, random_state: int = 42):
    if family == "linear_regression":
        return LinearRegression(n_jobs=-1)
    if family == "ridge":
        return Ridge(alpha=4.0, random_state=random_state)
    if family == "elastic_net":
        return ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=random_state, max_iter=5000)
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=18,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=600,
            max_depth=20,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "xgboost" and XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=700,
            learning_rate=0.04,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "lightgbm" and LGBMRegressor is not None:
        return LGBMRegressor(
            n_estimators=800,
            learning_rate=0.04,
            num_leaves=63,
            random_state=random_state,
            n_jobs=-1,
        )
    if family == "catboost" and CatBoostRegressor is not None:
        return CatBoostRegressor(
            depth=8,
            learning_rate=0.04,
            iterations=900,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
        )
    raise ValueError(f"Unsupported family: {family}")


def run_lazypredict_discovery(X_train, X_holdout, y_train, y_holdout) -> pd.DataFrame:
    from lazypredict.Supervised import LazyRegressor

    train_n = min(7000, X_train.shape[0])
    hold_n = min(3000, X_holdout.shape[0])
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

    reg = LazyRegressor(verbose=0, ignore_warnings=True)
    models, _ = reg.fit(X_train_lazy, X_holdout_lazy, y_train_lazy, y_holdout_lazy)
    table = models.reset_index().rename(columns={"index": "Model"})
    if "Model" not in table.columns:
        table = table.rename(columns={table.columns[0]: "Model"})

    keep = [c for c in ["Model", "R-Squared", "RMSE", "Time Taken"] if c in table.columns]
    return table[keep].reset_index(drop=True)


def select_top3_eligible_families(lazy_table, X_train, y_train, X_holdout, y_holdout, random_state: int = 42):
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
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        preds = np.clip(model.predict(X_holdout), 0, None)
        m = regression_metrics(y_holdout, preds)

        eligible = (m["smape"] < 1.0) and (train_time < 900)
        rows.append(
            {
                "lazy_model": row["Model"],
                "family": family,
                "smape": m["smape"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "train_time_sec": train_time,
                "eligible": eligible,
                "eligibility_note": "eligible" if eligible else "filtered: instability/speed",
            }
        )
        if eligible:
            chosen.append(family)
        if len(chosen) >= 3:
            break

    table = pd.DataFrame(rows).sort_values(["smape", "mae", "rmse"], ascending=[True, True, True])
    return table.reset_index(drop=True), chosen[:3]


def run_manual_engineering_track(top3_families, X_train, y_train, X_holdout, y_holdout, random_state: int = 42):
    rows = []
    models: Dict[str, Any] = {}
    preds_map: Dict[str, np.ndarray] = {}

    for family in top3_families:
        model = make_estimator(family, random_state=random_state)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        preds = np.clip(model.predict(X_holdout), 0, None)
        m = regression_metrics(y_holdout, preds)

        latencies = []
        n_rows = min(300, X_holdout.shape[0])
        for i in range(n_rows):
            row_x = X_holdout[i : i + 1]
            l0 = time.perf_counter()
            _ = model.predict(row_x)
            latencies.append((time.perf_counter() - l0) * 1000)

        rows.append(
            {
                "model_name": family,
                "library_source": "manual",
                "smape": m["smape"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "train_time_sec": train_time,
                "infer_latency_ms": float(np.mean(latencies)) if latencies else np.nan,
                "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else np.nan,
                "interpretability_note": f"Manual lag-feature model: {family}",
            }
        )
        models[family] = model
        preds_map[family] = preds

    out = pd.DataFrame(rows).sort_values("smape", ascending=True).reset_index(drop=True)
    return out, models, preds_map


def run_flaml_track(X_train, y_train, X_holdout, y_holdout, time_budget: int = 300, random_state: int = 42):
    from flaml import AutoML

    automl = AutoML()
    t0 = time.perf_counter()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="regression",
        metric="mae",
        time_budget=time_budget,
        eval_method="cv",
        n_splits=3,
        estimator_list=["lgbm", "xgboost", "rf", "extra_tree", "xgb_limitdepth"],
        seed=random_state,
    )
    train_time = time.perf_counter() - t0

    preds = np.clip(automl.predict(X_holdout), 0, None)
    m = regression_metrics(y_holdout, preds)

    return {
        "model_name": str(automl.best_estimator),
        "library_source": "flaml",
        "smape": m["smape"],
        "mae": m["mae"],
        "rmse": m["rmse"],
        "train_time_sec": train_time,
        "infer_latency_ms": np.nan,
        "p95_latency_ms": np.nan,
        "interpretability_note": "FLAML budget-aware challenger",
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
        "time_budget": time_budget,
        "predictions": preds,
    }


def run_pycaret_track(train_df: pd.DataFrame, holdout_df: pd.DataFrame, target_col: str, session_id: int, model_output_path: Path):
    try:
        import sys

        original_version_info = sys.version_info
        try:
            if tuple(sys.version_info) >= (3, 12):
                sys.version_info = (3, 11, 9, "final", 0)
            from pycaret.regression import compare_models, create_model, finalize_model, models as pycaret_models, predict_model, save_model, setup, tune_model
        finally:
            sys.version_info = original_version_info

        if len(train_df) > 35000:
            train_df = train_df.sample(n=35000, random_state=session_id).reset_index(drop=True)

        setup(
            data=train_df,
            target=target_col,
            session_id=session_id,
            fold_strategy="timeseries",
            fold=3,
            data_split_shuffle=False,
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
        preferred_ids = ["lr", "ridge", "lasso", "rf", "et", "xgboost", "lightgbm", "catboost"]
        include_ids = [m for m in preferred_ids if m in available_ids]

        if include_ids:
            best = compare_models(sort="MAE", include=include_ids)
        else:
            best = compare_models(sort="MAE")
        if isinstance(best, list):
            if len(best) == 0:
                fallback_id = include_ids[0] if include_ids else "lr"
                best = create_model(fallback_id)
            else:
                best = best[0]

        try:
            tuned = tune_model(best, optimize="MAE")
        except Exception:
            tuned = best
        final_model = finalize_model(tuned)

        pred = predict_model(final_model, data=holdout_df.copy())
        y_pred_col = "prediction_label" if "prediction_label" in pred.columns else "Label"
        preds = np.clip(pred[y_pred_col].astype(float).values, 0, None)
        m = regression_metrics(holdout_df[target_col].astype(float).values, preds)

        save_model(final_model, str(model_output_path))

        return {
            "model_name": type(final_model).__name__,
            "library_source": "pycaret",
            "smape": m["smape"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "train_time_sec": np.nan,
            "infer_latency_ms": np.nan,
            "p95_latency_ms": np.nan,
            "interpretability_note": "PyCaret tuned finalized regression model",
            "predictions": preds,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "model_name": "pycaret_failed",
            "library_source": "pycaret",
            "smape": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "train_time_sec": np.nan,
            "infer_latency_ms": np.nan,
            "p95_latency_ms": np.nan,
            "interpretability_note": f"PyCaret unavailable or failed: {exc}",
            "predictions": np.array([]),
            "status": "failed",
        }


def build_leaderboard(
    project_name: str,
    lazy_results: pd.DataFrame,
    manual_results: pd.DataFrame,
    flaml_result: dict,
    pycaret_result: dict,
    classical_baselines: pd.DataFrame,
):
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

    for _, r in classical_baselines.iterrows():
        rows.append(
            {
                "project_name": project_name,
                "task_type": "time_series_regression",
                "library_source": r.get("library_source", "classical_baseline"),
                "model_name": r["model_name"],
                "cv_metric_mean": np.nan,
                "cv_metric_std": np.nan,
                "holdout_primary_metric": r["smape"],
                "holdout_secondary_metric": r["mae"],
                "holdout_tertiary_metric": r["rmse"],
                "calibration_metric": np.nan,
                "train_time_sec": np.nan,
                "infer_latency_ms": np.nan,
                "p95_latency_ms": np.nan,
                "model_size_mb": np.nan,
                "retrain_time_sec": np.nan,
                "interpretability_note": r.get("interpretability_note", "classical baseline"),
            }
        )

    for _, r in lazy_results.iterrows():
        rows.append(
            {
                "project_name": project_name,
                "task_type": "time_series_regression",
                "library_source": "lazypredict",
                "model_name": r["family"],
                "cv_metric_mean": np.nan,
                "cv_metric_std": np.nan,
                "holdout_primary_metric": r["smape"],
                "holdout_secondary_metric": r["mae"],
                "holdout_tertiary_metric": r["rmse"],
                "calibration_metric": np.nan,
                "train_time_sec": r.get("train_time_sec"),
                "infer_latency_ms": np.nan,
                "p95_latency_ms": np.nan,
                "model_size_mb": np.nan,
                "retrain_time_sec": r.get("train_time_sec"),
                "interpretability_note": r.get("eligibility_note", "Lazy discovery"),
            }
        )

    for _, r in manual_results.iterrows():
        rows.append(
            {
                "project_name": project_name,
                "task_type": "time_series_regression",
                "library_source": "manual",
                "model_name": r["model_name"],
                "cv_metric_mean": np.nan,
                "cv_metric_std": np.nan,
                "holdout_primary_metric": r["smape"],
                "holdout_secondary_metric": r["mae"],
                "holdout_tertiary_metric": r["rmse"],
                "calibration_metric": np.nan,
                "train_time_sec": r.get("train_time_sec"),
                "infer_latency_ms": r.get("infer_latency_ms"),
                "p95_latency_ms": r.get("p95_latency_ms"),
                "model_size_mb": np.nan,
                "retrain_time_sec": r.get("train_time_sec"),
                "interpretability_note": r.get("interpretability_note", "manual track"),
            }
        )

    rows.append(
        {
            "project_name": project_name,
            "task_type": "time_series_regression",
            "library_source": "flaml",
            "model_name": flaml_result.get("model_name", "flaml_best"),
            "cv_metric_mean": np.nan,
            "cv_metric_std": np.nan,
            "holdout_primary_metric": flaml_result.get("smape"),
            "holdout_secondary_metric": flaml_result.get("mae"),
            "holdout_tertiary_metric": flaml_result.get("rmse"),
            "calibration_metric": np.nan,
            "train_time_sec": flaml_result.get("train_time_sec"),
            "infer_latency_ms": flaml_result.get("infer_latency_ms"),
            "p95_latency_ms": flaml_result.get("p95_latency_ms"),
            "model_size_mb": np.nan,
            "retrain_time_sec": flaml_result.get("train_time_sec"),
            "interpretability_note": flaml_result.get("interpretability_note", "FLAML track"),
        }
    )

    rows.append(
        {
            "project_name": project_name,
            "task_type": "time_series_regression",
            "library_source": "pycaret",
            "model_name": pycaret_result.get("model_name", "pycaret_best"),
            "cv_metric_mean": np.nan,
            "cv_metric_std": np.nan,
            "holdout_primary_metric": pycaret_result.get("smape"),
            "holdout_secondary_metric": pycaret_result.get("mae"),
            "holdout_tertiary_metric": pycaret_result.get("rmse"),
            "calibration_metric": np.nan,
            "train_time_sec": pycaret_result.get("train_time_sec"),
            "infer_latency_ms": pycaret_result.get("infer_latency_ms"),
            "p95_latency_ms": pycaret_result.get("p95_latency_ms"),
            "model_size_mb": np.nan,
            "retrain_time_sec": pycaret_result.get("train_time_sec"),
            "interpretability_note": pycaret_result.get("interpretability_note", "PyCaret track"),
        }
    )

    df = pd.DataFrame(rows)

    # Lower errors are better.
    smape_norm = df["holdout_primary_metric"] / (df["holdout_primary_metric"].max(skipna=True) or 1)
    mae_norm = df["holdout_secondary_metric"] / (df["holdout_secondary_metric"].max(skipna=True) or 1)
    rmse_norm = df["holdout_tertiary_metric"] / (df["holdout_tertiary_metric"].max(skipna=True) or 1)

    df["rank_score"] = 1.0 - (0.6 * smape_norm.fillna(1) + 0.25 * mae_norm.fillna(1) + 0.15 * rmse_norm.fillna(1))
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df["final_rank"] = np.arange(1, len(df) + 1)

    for c in required:
        if c not in df.columns:
            df[c] = np.nan
    return df[required]


def create_ops_signals(pred_df: pd.DataFrame, pred_col: str, baseline_col: str | None = None) -> pd.DataFrame:
    out = pred_df.copy()
    pred = out[pred_col].astype(float)

    median_sales = max(pred.median(), 1.0)
    out["staffing_proxy"] = np.ceil(pred / (0.18 * median_sales)).astype(int)

    rolling_base = pred.rolling(14, min_periods=1).mean()
    out["replenishment_signal"] = np.where(pred > rolling_base * 1.10, "increase_stock", "normal")

    spike_threshold = pred.quantile(0.90)
    out["demand_spike_alert"] = pred >= spike_threshold

    if baseline_col and baseline_col in out.columns:
        out["uplift_vs_baseline"] = out[pred_col] - out[baseline_col]

    return out
