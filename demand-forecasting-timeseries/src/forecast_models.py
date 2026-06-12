"""Forecasting model wrappers and lag-feature utilities."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import holidays
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


def _holiday_set(country: str) -> holidays.HolidayBase:
    return holidays.country_holidays(country)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (percentage). Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    safe = np.where(denom == 0, 1.0, denom)
    return float(np.mean(200.0 * np.abs(y_true - y_pred) / safe))


def calibration_bias_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Absolute relative bias in % (lower is better)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_abs_true = float(np.mean(np.abs(y_true))) or 1.0
    bias = float(np.mean(y_pred - y_true))
    return abs(100.0 * bias / mean_abs_true)


def build_time_features(index: pd.DatetimeIndex, holiday_country: str = "FR") -> pd.DataFrame:
    """Construct calendar features from timestamps."""
    hdays = _holiday_set(holiday_country)
    out = pd.DataFrame(index=index)
    out["hour"] = index.hour
    out["day_of_week"] = index.dayofweek
    out["day_of_month"] = index.day
    out["month"] = index.month
    out["is_weekend"] = (index.dayofweek >= 5).astype(int)
    out["is_holiday"] = pd.Series(index.date, index=index).map(lambda d: int(d in hdays))
    return out


def build_lagged_frame(
    series: pd.Series,
    lags: tuple[int, ...] = (1, 2, 3, 24, 48, 168),
    rolling_windows: tuple[int, ...] = (3, 24, 168),
    holiday_country: str = "FR",
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create supervised tabular frame from a univariate series."""
    df = pd.DataFrame({"target": series.astype(float)})
    for lag in lags:
        df[f"lag_{lag}"] = df["target"].shift(lag)

    lagged_target = df["target"].shift(1)
    for win in rolling_windows:
        df[f"roll_mean_{win}"] = lagged_target.rolling(window=win).mean()
        df[f"roll_std_{win}"] = lagged_target.rolling(window=win).std()
        df[f"roll_min_{win}"] = lagged_target.rolling(window=win).min()
        df[f"roll_max_{win}"] = lagged_target.rolling(window=win).max()

    calendar = build_time_features(df.index, holiday_country=holiday_country)
    for col in calendar.columns:
        df[col] = calendar[col]

    if dropna:
        df = df.dropna()

    y = df["target"].copy()
    X = df.drop(columns=["target"])
    return X, y


def _one_step_lag_features(
    history: pd.Series,
    ts: pd.Timestamp,
    lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
    holiday_country: str,
) -> pd.DataFrame:
    values = history.astype(float).to_numpy()
    row: dict[str, float] = {}
    for lag in lags:
        row[f"lag_{lag}"] = float(values[-lag]) if len(values) >= lag else float(values[-1])

    for win in rolling_windows:
        slice_vals = values[-win:] if len(values) >= win else values
        row[f"roll_mean_{win}"] = float(np.mean(slice_vals))
        row[f"roll_std_{win}"] = float(np.std(slice_vals, ddof=0))
        row[f"roll_min_{win}"] = float(np.min(slice_vals))
        row[f"roll_max_{win}"] = float(np.max(slice_vals))

    cal = build_time_features(pd.DatetimeIndex([ts]), holiday_country=holiday_country).iloc[0].to_dict()
    row.update({k: float(v) for k, v in cal.items()})
    return pd.DataFrame([row], index=[ts])


def _safe_test_index(train: pd.Series, horizon: int, test_index: pd.DatetimeIndex | None) -> pd.DatetimeIndex:
    if test_index is not None:
        return test_index
    inferred = pd.infer_freq(train.index)
    freq = inferred or "h"
    start = train.index[-1] + pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=start, periods=horizon, freq=freq)


def recursive_lag_forecast(
    model: Any,
    history: pd.Series,
    horizon: int,
    test_index: pd.DatetimeIndex | None = None,
    lags: tuple[int, ...] = (1, 2, 3, 24, 48, 168),
    rolling_windows: tuple[int, ...] = (3, 24, 168),
    holiday_country: str = "FR",
    feature_order: list[str] | None = None,
) -> tuple[np.ndarray, float]:
    """Forecast recursively, feeding predictions back as lags."""
    idx = _safe_test_index(history, horizon, test_index)
    hist = history.copy()
    preds: list[float] = []
    start = time.perf_counter()
    for ts in idx:
        row = _one_step_lag_features(
            hist,
            ts=ts,
            lags=lags,
            rolling_windows=rolling_windows,
            holiday_country=holiday_country,
        )
        if feature_order is not None:
            row = row.reindex(columns=feature_order, fill_value=0.0)
        y_hat = float(np.asarray(model.predict(row), dtype=float).ravel()[0])
        preds.append(y_hat)
        hist.loc[ts] = y_hat
    infer_ms = ((time.perf_counter() - start) * 1000.0) / max(horizon, 1)
    return np.asarray(preds, dtype=float), infer_ms


def make_recursive_forecast_fn(
    estimator_factory: Callable[[], Any],
    lags: tuple[int, ...] = (1, 2, 3, 24, 48, 168),
    rolling_windows: tuple[int, ...] = (3, 24, 168),
    holiday_country: str = "FR",
) -> Callable[[pd.Series, int, pd.DatetimeIndex | None], dict[str, Any]]:
    """Build a forecast callable compatible with rolling backtest helpers."""

    def _forecast(train: pd.Series, horizon: int, test_index: pd.DatetimeIndex | None = None) -> dict[str, Any]:
        X_train, y_train = build_lagged_frame(
            train,
            lags=lags,
            rolling_windows=rolling_windows,
            holiday_country=holiday_country,
            dropna=True,
        )
        model = estimator_factory()
        fit_start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_start
        y_pred, infer_ms = recursive_lag_forecast(
            model=model,
            history=train,
            horizon=horizon,
            test_index=test_index,
            lags=lags,
            rolling_windows=rolling_windows,
            holiday_country=holiday_country,
            feature_order=list(X_train.columns),
        )
        return {
            "y_pred": y_pred,
            "fit_time_sec": fit_time,
            "infer_latency_ms": infer_ms,
            "model_object": model,
        }

    return _forecast


def naive_forecast(train: pd.Series, horizon: int, test_index: pd.DatetimeIndex | None = None) -> dict[str, Any]:
    """Last-value carry-forward baseline."""
    value = float(train.iloc[-1])
    y_pred = np.repeat(value, horizon)
    return {"y_pred": y_pred, "fit_time_sec": 0.0, "infer_latency_ms": 0.01}


def seasonal_naive_forecast(
    train: pd.Series,
    horizon: int,
    test_index: pd.DatetimeIndex | None = None,
    season_length: int = 24,
) -> dict[str, Any]:
    """Repeat prior seasonal cycle."""
    values = train.astype(float).to_numpy()
    if len(values) < season_length:
        return naive_forecast(train, horizon, test_index)
    tail = values[-season_length:]
    repeats = int(np.ceil(horizon / season_length))
    y_pred = np.tile(tail, repeats)[:horizon]
    return {"y_pred": y_pred, "fit_time_sec": 0.0, "infer_latency_ms": 0.02}


def sarima_forecast(
    train: pd.Series,
    horizon: int,
    test_index: pd.DatetimeIndex | None = None,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
) -> dict[str, Any]:
    """SARIMA wrapper with fallback to seasonal naive."""
    if len(train) < 7 * 24:
        return seasonal_naive_forecast(train, horizon, test_index, season_length=24)

    start = time.perf_counter()
    try:
        model = SARIMAX(
            train.astype(float),
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=80)
        pred_start = time.perf_counter()
        y_pred = fitted.forecast(steps=horizon).to_numpy()
        infer_ms = ((time.perf_counter() - pred_start) * 1000.0) / max(horizon, 1)
        return {
            "y_pred": y_pred,
            "fit_time_sec": time.perf_counter() - start,
            "infer_latency_ms": infer_ms,
            "model_object": fitted,
        }
    except Exception:
        return seasonal_naive_forecast(train, horizon, test_index, season_length=24)


def prophet_forecast(
    train: pd.Series,
    horizon: int,
    test_index: pd.DatetimeIndex | None = None,
) -> dict[str, Any]:
    """Prophet wrapper configured for hourly demand."""
    idx = _safe_test_index(train, horizon, test_index)
    df = pd.DataFrame({"ds": train.index, "y": train.astype(float).to_numpy()})

    start = time.perf_counter()
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode="additive",
    )
    model.fit(df)
    pred_start = time.perf_counter()
    future = pd.DataFrame({"ds": idx})
    forecast = model.predict(future)
    y_pred = forecast["yhat"].to_numpy()
    infer_ms = ((time.perf_counter() - pred_start) * 1000.0) / max(horizon, 1)
    return {
        "y_pred": y_pred,
        "fit_time_sec": time.perf_counter() - start,
        "infer_latency_ms": infer_ms,
        "model_object": model,
    }


@dataclass
class GradientBoostingLagForecaster:
    """Recursive gradient boosting forecaster on lag features."""

    lags: tuple[int, ...] = (1, 2, 3, 24, 48, 168)
    rolling_windows: tuple[int, ...] = (3, 24, 168)
    holiday_country: str = "FR"
    random_state: int = 42

    def __post_init__(self) -> None:
        self.model = GradientBoostingRegressor(random_state=self.random_state)

    def fit(self, series: pd.Series) -> tuple[float, pd.DataFrame, pd.Series]:
        X, y = build_lagged_frame(
            series,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            holiday_country=self.holiday_country,
            dropna=True,
        )
        start = time.perf_counter()
        self.model.fit(X, y)
        fit_time = time.perf_counter() - start
        self.feature_order_ = list(X.columns)
        return fit_time, X, y

    def predict(
        self, history: pd.Series, horizon: int, test_index: pd.DatetimeIndex | None = None
    ) -> tuple[np.ndarray, float]:
        return recursive_lag_forecast(
            self.model,
            history=history,
            horizon=horizon,
            test_index=test_index,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            holiday_country=self.holiday_country,
            feature_order=self.feature_order_,
        )


def gradient_boosting_forecast(
    train: pd.Series,
    horizon: int,
    test_index: pd.DatetimeIndex | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for backtesting API."""
    forecaster = GradientBoostingLagForecaster()
    fit_time, _, _ = forecaster.fit(train)
    y_pred, infer_ms = forecaster.predict(train, horizon, test_index=test_index)
    return {
        "y_pred": y_pred,
        "fit_time_sec": fit_time,
        "infer_latency_ms": infer_ms,
        "model_object": forecaster.model,
    }


def make_lag_train_test(
    series_train: pd.Series,
    series_eval: pd.Series,
    lags: tuple[int, ...] = (1, 2, 3, 24, 48, 168),
    rolling_windows: tuple[int, ...] = (3, 24, 168),
    holiday_country: str = "FR",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Create time-aware train/eval lag-feature matrices.

    Note:
    This is useful for one-step-ahead evaluation where lag features can use
    recently observed actuals. For strict multi-step deployment simulation,
    use `recursive_lag_forecast` on the holdout horizon.
    """
    full = pd.concat([series_train, series_eval]).sort_index()
    X_full, y_full = build_lagged_frame(
        full,
        lags=lags,
        rolling_windows=rolling_windows,
        holiday_country=holiday_country,
        dropna=True,
    )
    cutoff = series_train.index.max()
    train_mask = X_full.index <= cutoff
    eval_mask = X_full.index > cutoff
    X_train, y_train = X_full.loc[train_mask], y_full.loc[train_mask]
    X_eval, y_eval = X_full.loc[eval_mask], y_full.loc[eval_mask]
    return X_train, y_train, X_eval, y_eval


def evaluate_sklearn_estimator(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    name: str,
) -> dict[str, Any]:
    """Train and evaluate a scikit-learn style regressor."""
    model = clone(estimator)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    pred_start = time.perf_counter()
    y_pred = model.predict(X_eval)
    infer_ms = ((time.perf_counter() - pred_start) * 1000.0) / max(len(y_eval), 1)

    return {
        "model_name": name,
        "y_pred": np.asarray(y_pred, dtype=float),
        "fit_time_sec": fit_time,
        "infer_latency_ms": infer_ms,
        "model_object": model,
        "sMAPE": float(smape(y_eval.to_numpy(), np.asarray(y_pred, dtype=float))),
        "MAE": float(mean_absolute_error(y_eval, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_eval, y_pred))),
        "calibration_metric": float(calibration_bias_metric(y_eval.to_numpy(), np.asarray(y_pred, dtype=float))),
    }


def run_lazypredict_discovery(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run LazyPredict and return:
      1) ranked benchmark table with project metrics
      2) prediction matrix by model family
    """
    from lazypredict.Supervised import LazyRegressor

    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=True,
    )
    models_df, preds_df = reg.fit(X_train, X_eval, y_train, y_eval)
    table = models_df.reset_index()
    if "model_family" not in table.columns:
        if "Model" in table.columns:
            table = table.rename(columns={"Model": "model_family"})
        elif "index" in table.columns:
            table = table.rename(columns={"index": "model_family"})
        else:
            table["model_family"] = table.iloc[:, 0].astype(str)

    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        family = str(row["model_family"])
        if family not in preds_df.columns:
            continue
        y_pred = preds_df[family].to_numpy(dtype=float)
        rows.append(
            {
                "model_family": family,
                "lazy_r2": float(row.get("R-Squared", np.nan)),
                "lazy_rmse": float(row.get("RMSE", np.nan)),
                "fit_time_sec": float(row.get("Time Taken", np.nan)),
                "holdout_sMAPE": float(smape(y_eval.to_numpy(), y_pred)),
                "holdout_MAE": float(mean_absolute_error(y_eval, y_pred)),
                "holdout_RMSE": float(np.sqrt(mean_squared_error(y_eval, y_pred))),
                "calibration_metric": float(calibration_bias_metric(y_eval.to_numpy(), y_pred)),
            }
        )

    ranked = pd.DataFrame(rows).sort_values(
        by=["holdout_sMAPE", "holdout_MAE", "holdout_RMSE"],
        ascending=[True, True, True],
    )
    ranked["library_source"] = "lazypredict"
    ranked = ranked.reset_index(drop=True)
    return ranked, preds_df


def select_eligible_lazypredict_models(
    discovery_df: pd.DataFrame,
    top_k: int = 3,
    max_fit_time_sec: float = 120.0,
    blocked_families: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter LazyPredict output to deployment-eligible model families."""
    blocked = blocked_families or {
        "GaussianProcessRegressor",
        "KernelRidge",
        "MLPRegressor",
        "SVR",
        "NuSVR",
        "LinearSVR",
        "TransformedTargetRegressor",
    }
    df = discovery_df.copy()
    df["is_eligible"] = True
    df["eligibility_reason"] = "eligible"

    bad_metric = ~np.isfinite(df["holdout_sMAPE"])
    too_slow = df["fit_time_sec"].fillna(np.inf) > max_fit_time_sec
    blocked_name = df["model_family"].isin(blocked)

    df.loc[bad_metric, ["is_eligible", "eligibility_reason"]] = [False, "invalid_metric"]
    df.loc[too_slow, ["is_eligible", "eligibility_reason"]] = [False, "excessive_fit_time"]
    df.loc[blocked_name, ["is_eligible", "eligibility_reason"]] = [False, "blocked_family"]

    eligible = (
        df[df["is_eligible"]]
        .drop_duplicates(subset=["model_family"])
        .sort_values(["holdout_sMAPE", "holdout_MAE", "holdout_RMSE"], ascending=True)
        .head(top_k)
        .reset_index(drop=True)
    )
    return eligible, df


def build_manual_estimator(model_family: str, random_state: int = 42) -> Any:
    """Map a model family name to a manually engineered estimator."""
    family = model_family.replace("LazyPredict::", "").strip()

    linear_pipeline = lambda model: Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", model),
        ]
    )

    factories: dict[str, Callable[[], Any]] = {
        "LinearRegression": lambda: linear_pipeline(LinearRegression()),
        "Ridge": lambda: linear_pipeline(Ridge(alpha=1.0, random_state=random_state)),
        "Lasso": lambda: linear_pipeline(Lasso(alpha=0.001, random_state=random_state, max_iter=6000)),
        "ElasticNet": lambda: linear_pipeline(
            ElasticNet(alpha=0.001, l1_ratio=0.25, random_state=random_state, max_iter=8000)
        ),
        "GradientBoostingRegressor": lambda: GradientBoostingRegressor(random_state=random_state),
        "RandomForestRegressor": lambda: RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTreesRegressor": lambda: ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if family == "XGBRegressor":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
            )
        except Exception:
            pass

    if family in factories:
        return factories[family]()
    raise ValueError(f"Unsupported manual model family: {model_family}")


def run_lazypredict_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Backwards-compatible wrapper returning top LazyPredict predictions."""
    discovery_df, preds_df = run_lazypredict_discovery(X_train, y_train, X_eval, y_eval)
    outputs: list[dict[str, Any]] = []
    for _, row in discovery_df.head(top_n).iterrows():
        name = row["model_family"]
        if name not in preds_df.columns:
            continue
        y_pred = preds_df[name].to_numpy(dtype=float)
        outputs.append(
            {
                "model_name": f"LazyPredict::{name}",
                "y_pred": y_pred,
                "fit_time_sec": float(row.get("fit_time_sec", np.nan)),
                "infer_latency_ms": np.nan,
                "model_object": None,
                "sMAPE": float(row.get("holdout_sMAPE", np.nan)),
                "MAE": float(row.get("holdout_MAE", np.nan)),
                "RMSE": float(row.get("holdout_RMSE", np.nan)),
                "calibration_metric": float(row.get("calibration_metric", np.nan)),
            }
        )
    return outputs


def flaml_smape_metric(
    X_val: Any,
    y_val: Any,
    estimator: Any,
    labels: Any,
    X_train: Any,
    y_train: Any,
    weight_val: Any | None = None,
    weight_train: Any | None = None,
    config: dict[str, Any] | None = None,
    groups_val: Any | None = None,
    groups_train: Any | None = None,
) -> tuple[float, dict[str, float]]:
    """Custom FLAML objective using project primary metric: sMAPE."""
    y_val_arr = np.asarray(y_val, dtype=float)
    y_pred = np.asarray(estimator.predict(X_val), dtype=float)
    val_smape = smape(y_val_arr, y_pred)
    val_mae = float(mean_absolute_error(y_val_arr, y_pred))
    val_rmse = float(np.sqrt(mean_squared_error(y_val_arr, y_pred)))
    return val_smape, {
        "val_smape": val_smape,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
    }


def run_flaml_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    time_budget_sec: int = 180,
    seed: int = 42,
    estimator_list: list[str] | None = None,
    n_splits: int = 4,
) -> dict[str, Any]:
    """Run FLAML AutoML with sMAPE optimization and time-aware CV."""
    from flaml import AutoML

    automl = AutoML()
    start = time.perf_counter()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="regression",
        metric=flaml_smape_metric,
        time_budget=time_budget_sec,
        eval_method="cv",
        split_type="time",
        n_splits=n_splits,
        estimator_list=estimator_list
        or ["lgbm", "xgboost", "xgb_limitdepth", "rf", "extra_tree"],
        seed=seed,
        n_jobs=-1,
        model_history=True,
        log_training_metric=True,
        log_file_name="",
        verbose=0,
    )
    fit_time = time.perf_counter() - start
    return {
        "automl": automl,
        "model_name": f"FLAML::{automl.best_estimator}",
        "fit_time_sec": fit_time,
        "model_object": automl.model,
        "best_estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
        "best_config_per_estimator": automl.best_config_per_estimator,
        "best_loss_per_estimator": automl.best_loss_per_estimator,
        "config_history": automl.config_history,
    }


def run_flaml_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    time_budget_sec: int = 90,
) -> dict[str, Any]:
    """Backwards-compatible FLAML benchmark wrapper."""
    out = run_flaml_optimization(
        X_train=X_train,
        y_train=y_train,
        time_budget_sec=time_budget_sec,
        seed=42,
        estimator_list=["lgbm", "xgboost", "xgb_limitdepth", "rf", "extra_tree"],
        n_splits=3,
    )
    pred_start = time.perf_counter()
    y_pred = out["automl"].predict(X_eval)
    infer_ms = ((time.perf_counter() - pred_start) * 1000.0) / max(len(X_eval), 1)
    return {
        "model_name": out["model_name"],
        "y_pred": np.asarray(y_pred, dtype=float),
        "fit_time_sec": out["fit_time_sec"],
        "infer_latency_ms": infer_ms,
        "model_object": out["model_object"],
        "best_config": out["best_config"],
        "best_estimator": out["best_estimator"],
    }


def run_pycaret_regression_workflow(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target_col: str = "target",
    session_id: int = 42,
    compare_include: list[str] | None = None,
    tune_iter: int = 25,
    save_path: str | None = None,
    blend_top_n: int = 2,
) -> dict[str, Any]:
    """Run a full PyCaret regression workflow in OOP API."""
    from pycaret.regression import RegressionExperiment

    exp = RegressionExperiment(
        target=target_col,
        session_id=session_id,
        fold=4,
        fold_strategy="timeseries",
        preprocess=True,
        normalize=True,
        feature_selection=False,
        remove_outliers=False,
        n_jobs=-1,
        verbose=False,
    )
    exp.fit(train_df)

    compare = exp.compare_models(
        include=compare_include,
        sort="MAE",
        n_select=max(1, blend_top_n),
        turbo=True,
        errors="ignore",
        verbose=False,
    )

    tuned = exp.tune_model(
        compare.best,
        n_iter=tune_iter,
        optimize="MAE",
        verbose=False,
    )
    selected_pipeline = tuned.pipeline
    selected_label = f"PyCaretReg::Tuned::{type(tuned.pipeline.steps[-1][1]).__name__}"

    blend_result = None
    if len(compare.models) >= 2:
        try:
            blend_result = exp.blend_models(compare.models[:2], verbose=False)
            tune_mae = float(tuned.metrics.loc["Mean", "MAE"])
            blend_mae = float(blend_result.metrics.loc["Mean", "MAE"])
            if blend_mae < tune_mae:
                selected_pipeline = blend_result.pipeline
                selected_label = "PyCaretReg::BlendTop2"
        except Exception:
            blend_result = None

    final = exp.finalize_model(selected_pipeline)
    pred_start = time.perf_counter()
    pred_result = exp.predict_model(final.pipeline, data=eval_df.copy(), verbose=False)
    infer_ms = ((time.perf_counter() - pred_start) * 1000.0) / max(len(eval_df), 1)

    pred_col = "prediction_label"
    if pred_col not in pred_result.predictions.columns:
        fallback = [c for c in pred_result.predictions.columns if "prediction" in c.lower()]
        pred_col = fallback[0] if fallback else pred_result.predictions.columns[-1]
    y_pred = pred_result.predictions[pred_col].to_numpy(dtype=float)

    saved_path = None
    if save_path:
        saved_path = exp.save_model(final.pipeline, save_path)

    return {
        "experiment": exp,
        "compare_result": compare,
        "compare_leaderboard": compare.leaderboard.copy(),
        "tune_result": tuned,
        "blend_result": blend_result,
        "selected_pipeline": selected_pipeline,
        "selected_label": selected_label,
        "final_result": final,
        "predictions_df": pred_result.predictions.copy(),
        "y_pred": y_pred,
        "infer_latency_ms": infer_ms,
        "model_object": final.pipeline,
        "saved_path": str(saved_path) if saved_path else None,
    }


def run_pycaret_regression_benchmark(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target_col: str = "target",
) -> dict[str, Any]:
    """Backwards-compatible wrapper around the PyCaret regression workflow."""
    out = run_pycaret_regression_workflow(
        train_df=train_df,
        eval_df=eval_df,
        target_col=target_col,
        session_id=42,
        compare_include=None,
        tune_iter=15,
        save_path=None,
        blend_top_n=2,
    )
    return {
        "model_name": out["selected_label"],
        "y_pred": out["y_pred"],
        "fit_time_sec": np.nan,
        "infer_latency_ms": out["infer_latency_ms"],
        "model_object": out["model_object"],
        "compare_leaderboard": out["compare_leaderboard"],
    }


def run_pycaret_timeseries_workflow(
    train_series: pd.Series,
    horizon: int,
    session_id: int = 42,
    compare_include: list[str] | None = None,
    tune_iter: int = 20,
    seasonal_period: int = 24,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Run a full PyCaret time-series workflow (compare -> tune -> finalize)."""
    from pycaret.time_series import TimeSeriesExperiment

    exp = TimeSeriesExperiment(
        fh=horizon,
        seasonal_period=seasonal_period,
        session_id=session_id,
        fold=3,
        fold_strategy="expanding",
        preprocess=True,
        n_jobs=-1,
        verbose=False,
    )
    exp.fit(train_series.astype(float))

    compare = exp.compare_models(
        include=compare_include,
        sort="MAE",
        n_select=1,
        turbo=True,
        errors="ignore",
        verbose=False,
    )
    tuned = exp.tune_model(
        compare.best,
        n_iter=tune_iter,
        optimize="MAE",
        verbose=False,
    )
    final = exp.finalize_model(tuned.pipeline)
    pred_start = time.perf_counter()
    pred_result = exp.predict_model(final.pipeline, verbose=False)
    infer_ms = ((time.perf_counter() - pred_start) * 1000.0) / max(horizon, 1)

    if isinstance(pred_result.predictions, pd.DataFrame):
        pred_col = "y_pred" if "y_pred" in pred_result.predictions.columns else pred_result.predictions.columns[0]
        y_pred = pred_result.predictions[pred_col].to_numpy(dtype=float)
    else:
        y_pred = np.asarray(pred_result.predictions, dtype=float)

    saved_path = None
    if save_path:
        saved_path = exp.save_model(final.pipeline, save_path)

    return {
        "experiment": exp,
        "compare_result": compare,
        "compare_leaderboard": compare.leaderboard.copy(),
        "tune_result": tuned,
        "final_result": final,
        "model_name": f"PyCaretTS::{type(final.pipeline.steps[-1][1]).__name__}",
        "y_pred": y_pred[:horizon],
        "infer_latency_ms": infer_ms,
        "model_object": final.pipeline,
        "saved_path": str(saved_path) if saved_path else None,
    }


def run_pycaret_timeseries_benchmark(train_series: pd.Series, horizon: int) -> dict[str, Any]:
    """Backwards-compatible wrapper for PyCaret time-series benchmark."""
    out = run_pycaret_timeseries_workflow(
        train_series=train_series,
        horizon=horizon,
        session_id=42,
        compare_include=["naive", "arima", "exp_smooth", "ets"],
        tune_iter=10,
        seasonal_period=24,
        save_path=None,
    )
    return {
        "model_name": out["model_name"],
        "y_pred": out["y_pred"],
        "fit_time_sec": np.nan,
        "infer_latency_ms": out["infer_latency_ms"],
        "model_object": out["model_object"],
        "compare_leaderboard": out["compare_leaderboard"],
    }


class _LSTMModel(nn.Module if nn is not None else object):  # pragma: no cover - optional
    def __init__(self, input_size: int = 1, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def lstm_forecast(
    train: pd.Series,
    horizon: int,
    test_index: pd.DatetimeIndex | None = None,
    lookback: int = 24,
    epochs: int = 6,
    learning_rate: float = 0.01,
) -> dict[str, Any]:
    """Optional lightweight LSTM forecaster; falls back if torch unavailable."""
    if torch is None or nn is None or len(train) <= lookback + 5:
        return seasonal_naive_forecast(train, horizon, test_index, season_length=24)

    values = train.astype(float).to_numpy()
    mu = values.mean()
    sigma = values.std() if values.std() > 1e-8 else 1.0
    scaled = (values - mu) / sigma

    X_list, y_list = [], []
    for i in range(lookback, len(scaled)):
        X_list.append(scaled[i - lookback : i])
        y_list.append(scaled[i])

    X_train = torch.tensor(np.asarray(X_list), dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(np.asarray(y_list), dtype=torch.float32).unsqueeze(-1)

    model = _LSTMModel(input_size=1, hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
    fit_time = time.perf_counter() - start

    model.eval()
    history = list(scaled.copy())
    preds_scaled: list[float] = []
    pred_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(np.asarray(history[-lookback:]), dtype=torch.float32).view(1, lookback, 1)
            y_hat = float(model(x).item())
            preds_scaled.append(y_hat)
            history.append(y_hat)
    infer_ms = ((time.perf_counter() - pred_start) * 1000.0) / max(horizon, 1)

    y_pred = np.asarray(preds_scaled) * sigma + mu
    return {
        "y_pred": y_pred,
        "fit_time_sec": fit_time,
        "infer_latency_ms": infer_ms,
        "model_object": model,
    }


__all__ = [
    "GradientBoostingLagForecaster",
    "build_lagged_frame",
    "build_manual_estimator",
    "build_time_features",
    "calibration_bias_metric",
    "evaluate_sklearn_estimator",
    "flaml_smape_metric",
    "gradient_boosting_forecast",
    "lstm_forecast",
    "make_lag_train_test",
    "make_recursive_forecast_fn",
    "naive_forecast",
    "prophet_forecast",
    "recursive_lag_forecast",
    "run_flaml_benchmark",
    "run_flaml_optimization",
    "run_lazypredict_benchmark",
    "run_lazypredict_discovery",
    "run_pycaret_regression_benchmark",
    "run_pycaret_regression_workflow",
    "run_pycaret_timeseries_benchmark",
    "run_pycaret_timeseries_workflow",
    "sarima_forecast",
    "seasonal_naive_forecast",
    "select_eligible_lazypredict_models",
    "smape",
]
