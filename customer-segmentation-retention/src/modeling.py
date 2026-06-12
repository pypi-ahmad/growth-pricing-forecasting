"""Modeling utilities for segmentation, churn experiments, ranking, and deployment-oriented outputs."""

from __future__ import annotations

import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from flaml import AutoML
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import ClassificationExperiment
from scipy.special import expit
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42

UNIFIED_LEADERBOARD_COLUMNS = [
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
    "model_size_mb",
    "interpretability_note",
    "rank_score",
    "final_rank",
]


@dataclass
class ThresholdConfig:
    retention_offer_cost: float = 12.0
    prevented_churn_value: float = 80.0
    retention_success_rate: float = 0.30
    max_target_share: float = 0.40


def split_train_valid_test(
    frame: pd.DataFrame,
    target_col: str = "churn",
    test_size: float = 0.20,
    valid_size: float = 0.25,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create a stratified train/validation/test split."""
    X = frame.drop(columns=[target_col]).copy()
    y = frame[target_col].astype(int).copy()

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_size,
        random_state=random_state,
        stratify=y_train_valid,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def _build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, object, list[str]]] = [
        ("num", Pipeline(steps=numeric_steps), numeric_cols),
    ]

    if categorical_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _predict_scores(estimator: object, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 1:
            return np.asarray(proba, dtype=float)
        return np.asarray(proba[:, -1], dtype=float)

    if hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(X)
        return expit(np.asarray(decision, dtype=float))

    predictions = estimator.predict(X)
    return np.asarray(predictions, dtype=float)


def _normalize_metric(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = series.astype(float)
    if values.isna().all() or np.isclose(values.max(), values.min()):
        return pd.Series(np.full(len(values), 0.5), index=series.index)

    norm = (values - values.min()) / (values.max() - values.min())
    if not higher_is_better:
        norm = 1 - norm
    return norm.fillna(0.0)


def _estimate_infer_latency_ms(
    predict_fn: Callable[[pd.DataFrame | np.ndarray], np.ndarray],
    X_sample: pd.DataFrame | np.ndarray,
    runs: int = 20,
) -> float:
    if isinstance(X_sample, pd.DataFrame):
        sample = X_sample.iloc[: min(len(X_sample), 256)].copy()
    else:
        sample = X_sample[: min(len(X_sample), 256)]

    if len(sample) == 0:
        return float("nan")

    start = time.perf_counter()
    for _ in range(runs):
        _ = predict_fn(sample)
    elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0


def _model_size_mb(model_obj: object) -> float:
    try:
        return len(pickle.dumps(model_obj, protocol=pickle.HIGHEST_PROTOCOL)) / (1024 * 1024)
    except Exception:
        return float("nan")


def _set_random_state_recursive(estimator: object, seed: int) -> object:
    if not hasattr(estimator, "get_params"):
        return estimator

    params = estimator.get_params(deep=True)
    rand_params = {
        key: seed
        for key in params
        if key.endswith("random_state")
    }
    if rand_params:
        estimator.set_params(**rand_params)
    return estimator


def evaluate_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    y_pred_arr = (y_score_arr >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true_arr, y_score_arr)),
        "threshold": float(threshold),
    }


def threshold_profit_curve(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    config: ThresholdConfig | None = None,
) -> pd.DataFrame:
    """Compute campaign-level economics across probability thresholds."""
    if config is None:
        config = ThresholdConfig()

    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    thresholds = np.linspace(0.05, 0.95, 19)

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        preds = (y_score_arr >= threshold).astype(int)
        tp = int(((preds == 1) & (y_true_arr == 1)).sum())
        fp = int(((preds == 1) & (y_true_arr == 0)).sum())
        targeted = int(preds.sum())
        target_share = targeted / len(preds)

        precision = precision_score(y_true_arr, preds, zero_division=0)
        recall = recall_score(y_true_arr, preds, zero_division=0)
        expected_profit = (
            tp * config.prevented_churn_value * config.retention_success_rate
            - targeted * config.retention_offer_cost
        )

        rows.append(
            {
                "threshold": float(threshold),
                "targeted_customers": targeted,
                "target_share": float(target_share),
                "tp": tp,
                "fp": fp,
                "precision": float(precision),
                "recall": float(recall),
                "expected_profit": float(expected_profit),
            }
        )

    return pd.DataFrame(rows)


def pick_profit_optimal_threshold(threshold_table: pd.DataFrame, max_target_share: float) -> pd.Series:
    constrained = threshold_table.loc[threshold_table["target_share"] <= max_target_share].copy()
    if constrained.empty:
        return threshold_table.sort_values("expected_profit", ascending=False).iloc[0]
    return constrained.sort_values("expected_profit", ascending=False).iloc[0]


def build_calibration_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(
        np.asarray(y_true).astype(int),
        np.asarray(y_score).astype(float),
        n_bins=n_bins,
        strategy="quantile",
    )
    return pd.DataFrame(
        {
            "predicted_probability": prob_pred,
            "observed_frequency": prob_true,
        }
    )


def build_error_analysis(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)
    y_pred_arr = (y_score_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    return pd.DataFrame(
        {
            "metric": ["true_negative", "false_positive", "false_negative", "true_positive"],
            "count": [int(tn), int(fp), int(fn), int(tp)],
        }
    )


def save_model_artifact(model: object, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(model, f)
    return output_path


def load_model_artifact(path: str | Path) -> object:
    with Path(path).open("rb") as f:
        return pickle.load(f)


# ============================
# Segmentation Utilities
# ============================

def _safe_segmentation_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    silhouette = float(silhouette_score(X_scaled, labels))
    calinski = float(calinski_harabasz_score(X_scaled, labels))
    davies = float(davies_bouldin_score(X_scaled, labels))
    return silhouette, calinski, davies


def rank_segmentation_leaderboard(leaderboard: pd.DataFrame) -> pd.DataFrame:
    ranked = leaderboard.copy()
    primary = _normalize_metric(ranked["holdout_primary_metric"], higher_is_better=True)
    secondary = _normalize_metric(ranked["holdout_secondary_metric"], higher_is_better=True)
    tertiary = _normalize_metric(ranked["calibration_metric"], higher_is_better=False)

    ranked["rank_score"] = 0.60 * primary + 0.25 * secondary + 0.15 * tertiary
    ranked["final_rank"] = ranked["rank_score"].rank(method="dense", ascending=False).astype(int)
    return ranked.sort_values(["final_rank", "model_name"]).reset_index(drop=True)


def benchmark_segmentation_models(
    customer_frame: pd.DataFrame,
    feature_cols: list[str],
    project_name: str = "customer-segmentation-retention",
    random_state: int = SEED,
    min_clusters: int = 2,
    max_clusters: int = 7,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Run KMeans and Gaussian Mixture segmentation and rank by clustering quality."""
    features = customer_frame[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    features = features.fillna(features.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    rows: list[dict[str, float | str]] = []
    registry: dict[str, dict[str, object]] = {}

    for n_clusters in range(min_clusters, max_clusters + 1):
        start = time.perf_counter()
        kmeans = KMeans(n_clusters=n_clusters, n_init=25, random_state=random_state)
        labels = kmeans.fit_predict(X_scaled)
        train_time = time.perf_counter() - start
        sil, cal, dav = _safe_segmentation_metrics(X_scaled, labels)

        model_name = f"KMeans_k{n_clusters}"
        rows.append(
            {
                "project_name": project_name,
                "task_type": "segmentation",
                "library_source": "sklearn",
                "model_name": model_name,
                "cv_metric_mean": sil,
                "cv_metric_std": 0.0,
                "holdout_primary_metric": sil,
                "holdout_secondary_metric": cal,
                "holdout_tertiary_metric": 1.0 / (1.0 + dav) if np.isfinite(dav) else float("nan"),
                "calibration_metric": dav,
                "train_time_sec": float(train_time),
                "infer_latency_ms": _estimate_infer_latency_ms(kmeans.predict, X_scaled),
                "model_size_mb": _model_size_mb(kmeans),
                "interpretability_note": "Centroids provide direct segment archetypes.",
            }
        )
        registry[model_name] = {
            "model": kmeans,
            "labels": labels,
            "scaler": scaler,
            "feature_cols": feature_cols,
        }

        start = time.perf_counter()
        gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=random_state)
        labels = gmm.fit_predict(X_scaled)
        train_time = time.perf_counter() - start
        sil, cal, dav = _safe_segmentation_metrics(X_scaled, labels)

        model_name = f"GaussianMixture_k{n_clusters}"
        rows.append(
            {
                "project_name": project_name,
                "task_type": "segmentation",
                "library_source": "sklearn",
                "model_name": model_name,
                "cv_metric_mean": sil,
                "cv_metric_std": 0.0,
                "holdout_primary_metric": sil,
                "holdout_secondary_metric": cal,
                "holdout_tertiary_metric": 1.0 / (1.0 + dav) if np.isfinite(dav) else float("nan"),
                "calibration_metric": dav,
                "train_time_sec": float(train_time),
                "infer_latency_ms": _estimate_infer_latency_ms(gmm.predict, X_scaled),
                "model_size_mb": _model_size_mb(gmm),
                "interpretability_note": "Soft cluster probabilities support overlap-aware targeting.",
            }
        )
        registry[model_name] = {
            "model": gmm,
            "labels": labels,
            "scaler": scaler,
            "feature_cols": feature_cols,
        }

    leaderboard = rank_segmentation_leaderboard(pd.DataFrame(rows))
    return leaderboard[UNIFIED_LEADERBOARD_COLUMNS], registry


def label_segments_with_business_names(
    customer_frame: pd.DataFrame,
    segment_col: str = "segment_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map numeric segments to business labels."""
    profile = (
        customer_frame.groupby(segment_col, as_index=False)
        .agg(
            customers=("CustomerID", "nunique"),
            recency_days=("recency_days", "mean"),
            frequency_orders=("frequency_orders", "mean"),
            monetary_value=("monetary_value", "mean"),
            churn_rate=("churn", "mean"),
            avg_ltv=("ltv_proxy", "mean"),
        )
        .sort_values("avg_ltv", ascending=False)
        .reset_index(drop=True)
    )

    recency_q = profile["recency_days"].quantile([0.35, 0.70]).to_dict()
    freq_q = profile["frequency_orders"].quantile([0.35, 0.70]).to_dict()
    ltv_q = profile["avg_ltv"].quantile([0.35, 0.70]).to_dict()

    def _name(row: pd.Series) -> str:
        if row["avg_ltv"] >= ltv_q[0.70] and row["recency_days"] <= recency_q[0.35]:
            return "Champions"
        if row["recency_days"] >= recency_q[0.70] and row["churn_rate"] >= profile["churn_rate"].median():
            return "At-Risk"
        if row["frequency_orders"] <= freq_q[0.35] and row["avg_ltv"] <= ltv_q[0.35]:
            return "Low-Value Dormant"
        return "Steady Core"

    profile["segment_label"] = profile.apply(_name, axis=1)
    mapping = dict(zip(profile[segment_col], profile["segment_label"]))

    labeled = customer_frame.copy()
    labeled["segment_label"] = labeled[segment_col].map(mapping).fillna("Unassigned")
    return labeled, profile


# ============================
# Churn Modeling Track Helpers
# ============================

def _build_churn_row(
    project_name: str,
    library_source: str,
    model_name: str,
    cv_metric_mean: float,
    cv_metric_std: float,
    holdout_pr_auc: float,
    holdout_recall: float,
    holdout_precision: float,
    brier_score: float,
    train_time_sec: float,
    infer_latency_ms: float,
    model_size_mb: float,
    interpretability_note: str,
) -> dict[str, float | str]:
    return {
        "project_name": project_name,
        "task_type": "churn_prediction",
        "library_source": library_source,
        "model_name": model_name,
        "cv_metric_mean": float(cv_metric_mean),
        "cv_metric_std": float(cv_metric_std),
        "holdout_primary_metric": float(holdout_pr_auc),
        "holdout_secondary_metric": float(holdout_recall),
        "holdout_tertiary_metric": float(holdout_precision),
        "calibration_metric": float(brier_score),
        "train_time_sec": float(train_time_sec),
        "infer_latency_ms": float(infer_latency_ms),
        "model_size_mb": float(model_size_mb),
        "interpretability_note": interpretability_note,
    }


def _average_precision_scorer(
    estimator: object,
    X: pd.DataFrame | np.ndarray,
    y_true: pd.Series | np.ndarray,
) -> float:
    scores = _predict_scores(estimator, X)
    return float(average_precision_score(np.asarray(y_true).astype(int), scores))


def _rank_churn_candidates(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    primary = _normalize_metric(ranked["holdout_primary_metric"], higher_is_better=True)
    secondary = _normalize_metric(ranked["holdout_secondary_metric"], higher_is_better=True)
    tertiary = _normalize_metric(ranked["holdout_tertiary_metric"], higher_is_better=True)
    calibration = _normalize_metric(ranked["calibration_metric"], higher_is_better=False)

    ranked["rank_score"] = 0.50 * primary + 0.20 * secondary + 0.15 * tertiary + 0.15 * calibration
    ranked["final_rank"] = ranked["rank_score"].rank(method="dense", ascending=False).astype(int)
    return ranked.sort_values(["final_rank", "model_name"]).reset_index(drop=True)


def _map_lazy_model_to_family(model_name: str) -> str | None:
    mapping = {
        "LogisticRegression": "logistic_regression",
        "RidgeClassifier": "ridge_classifier",
        "RidgeClassifierCV": "ridge_classifier",
        "LinearDiscriminantAnalysis": "lda",
        "LinearSVC": "linear_svc",
        "CalibratedClassifierCV": "linear_svc",
        "BernoulliNB": "bernoulli_nb",
        "GaussianNB": "gaussian_nb",
        "RandomForestClassifier": "random_forest",
        "ExtraTreesClassifier": "extra_trees",
        "GradientBoostingClassifier": "gradient_boosting",
        "AdaBoostClassifier": "adaboost",
        "DecisionTreeClassifier": "decision_tree",
        "KNeighborsClassifier": "knn",
        "XGBClassifier": "xgboost",
    }
    return mapping.get(model_name)


def _build_manual_estimator(
    family: str,
    X_reference: pd.DataFrame,
    seed: int,
) -> tuple[Pipeline, str]:
    scale_numeric = family in {
        "logistic_regression",
        "ridge_classifier",
        "knn",
        "linear_svc",
        "lda",
    }
    preprocessor = _build_preprocessor(X_reference, scale_numeric=scale_numeric)

    if family == "logistic_regression":
        model = LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)
        note = "Linear baseline with high interpretability and stable calibration behavior."
    elif family == "ridge_classifier":
        model = RidgeClassifier(alpha=1.0, random_state=seed)
        note = "Fast linear margin model; good safety fallback when latency is strict."
    elif family == "lda":
        model = LinearDiscriminantAnalysis()
        note = "Low-latency probabilistic linear discriminant baseline."
    elif family == "linear_svc":
        model = LinearSVC(C=1.0, class_weight="balanced", random_state=seed)
        note = "Linear margin classifier suitable for fast scoring and robust separation."
    elif family == "bernoulli_nb":
        model = BernoulliNB()
        note = "Simple probabilistic baseline; useful robustness comparator."
    elif family == "gaussian_nb":
        model = GaussianNB()
        note = "Distributional baseline for quick probabilistic checks."
    elif family == "random_forest":
        model = RandomForestClassifier(
            n_estimators=450,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        note = "Bagged trees with robust performance and feature importance support."
    elif family == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        note = "High-variance tree ensemble; often strong for sparse/nonlinear tabular behavior."
    elif family == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=seed)
        note = "Boosted tree sequence balancing bias and variance in structured data."
    elif family == "adaboost":
        model = AdaBoostClassifier(random_state=seed)
        note = "Lightweight boosting baseline; interpretable ensemble depth profile."
    elif family == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=seed,
        )
        note = "High interpretability rules baseline with clear split diagnostics."
    elif family == "knn":
        model = KNeighborsClassifier(n_neighbors=35, weights="distance")
        note = "Local-neighborhood behavior baseline; useful for similarity-driven cases."
    elif family == "xgboost":
        model = XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
        note = "Strong nonlinear benchmark with high recall potential after threshold tuning."
    else:
        raise ValueError(f"Unsupported manual model family: {family}")

    estimator = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return estimator, note


def build_baseline_row(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_config: ThresholdConfig,
    project_name: str,
) -> tuple[pd.DataFrame, dict[str, Callable[[int], dict[str, object]]], dict[str, pd.DataFrame]]:
    """Create a simple baseline candidate for leaderboard context."""
    preprocessor = _build_preprocessor(X_train, scale_numeric=True)
    baseline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", DummyClassifier(strategy="prior", random_state=SEED)),
        ]
    )

    start = time.perf_counter()
    baseline.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    valid_scores = _predict_scores(baseline, X_valid)
    threshold_table = threshold_profit_curve(y_valid, valid_scores, config=threshold_config)
    chosen = pick_profit_optimal_threshold(threshold_table, max_target_share=threshold_config.max_target_share)
    threshold = float(chosen["threshold"])

    test_scores = _predict_scores(baseline, X_test)
    metrics = evaluate_at_threshold(y_test, test_scores, threshold=threshold)

    row = _build_churn_row(
        project_name=project_name,
        library_source="Baseline",
        model_name="Baseline_DummyPrior",
        cv_metric_mean=float("nan"),
        cv_metric_std=float("nan"),
        holdout_pr_auc=metrics["pr_auc"],
        holdout_recall=metrics["recall"],
        holdout_precision=metrics["precision"],
        brier_score=metrics["brier_score"],
        train_time_sec=train_time,
        infer_latency_ms=_estimate_infer_latency_ms(lambda data: _predict_scores(baseline, data), X_test),
        model_size_mb=_model_size_mb(baseline),
        interpretability_note="Reference baseline to quantify incremental gain from advanced tracks.",
    )

    def _baseline_retrainer(seed: int) -> dict[str, object]:
        local_model = clone(baseline)
        _set_random_state_recursive(local_model, seed)
        local_model.fit(X_train, y_train)

        local_valid_scores = _predict_scores(local_model, X_valid)
        local_threshold_table = threshold_profit_curve(y_valid, local_valid_scores, config=threshold_config)
        local_threshold = float(
            pick_profit_optimal_threshold(local_threshold_table, max_target_share=threshold_config.max_target_share)[
                "threshold"
            ]
        )
        local_test_scores = _predict_scores(local_model, X_test)
        local_metrics = evaluate_at_threshold(y_test, local_test_scores, threshold=local_threshold)

        return {
            "metrics": local_metrics,
            "threshold": local_threshold,
            "threshold_table": local_threshold_table,
            "test_scores": local_test_scores,
            "score_fn": lambda frame, mdl=local_model: _predict_scores(mdl, frame),
        }

    return pd.DataFrame([row]), {"Baseline_DummyPrior": _baseline_retrainer}, {"Baseline_DummyPrior": threshold_table}


def run_lazypredict_discovery_lab(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_config: ThresholdConfig,
    project_name: str = "customer-segmentation-retention",
    top_n_manual_families: int = 3,
    max_models: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Callable[[int], dict[str, object]]], dict[str, object]]:
    """Track 1: LazyPredict discovery and eligible-family selection."""
    lazy = LazyClassifier(verbose=0, ignore_warnings=True, predictions=True)
    model_table, _ = lazy.fit(X_train, X_valid, y_train, y_valid)
    model_table = model_table.sort_values("ROC AUC", ascending=False).head(max_models)

    rows: list[dict[str, float | str | int | None]] = []
    retrainers: dict[str, Callable[[int], dict[str, object]]] = {}
    threshold_tables: dict[str, pd.DataFrame] = {}

    for base_name in model_table.index.tolist():
        estimator = lazy.models.get(base_name)
        if estimator is None:
            continue

        try:
            valid_scores = _predict_scores(estimator, X_valid)
            threshold_table = threshold_profit_curve(y_valid, valid_scores, config=threshold_config)
            chosen = pick_profit_optimal_threshold(
                threshold_table,
                max_target_share=threshold_config.max_target_share,
            )
            threshold = float(chosen["threshold"])
            test_scores = _predict_scores(estimator, X_test)
            metrics = evaluate_at_threshold(y_test, test_scores, threshold=threshold)
        except Exception:
            continue

        model_name = f"Lazy_{base_name}"
        family = _map_lazy_model_to_family(base_name)
        eligible_supported = family is not None

        threshold_tables[model_name] = threshold_table
        row = _build_churn_row(
            project_name=project_name,
            library_source="LazyPredict",
            model_name=model_name,
            cv_metric_mean=float(model_table.loc[base_name, "ROC AUC"]),
            cv_metric_std=0.0,
            holdout_pr_auc=metrics["pr_auc"],
            holdout_recall=metrics["recall"],
            holdout_precision=metrics["precision"],
            brier_score=metrics["brier_score"],
            train_time_sec=float(model_table.loc[base_name, "Time Taken"]),
            infer_latency_ms=_estimate_infer_latency_ms(
                lambda data, est=estimator: _predict_scores(est, data),
                X_test,
            ),
            model_size_mb=_model_size_mb(estimator),
            interpretability_note=(
                "Eligible for manual lab" if eligible_supported else "Excluded: unsupported manual family"
            ),
        )
        row["manual_family"] = family
        row["eligible_for_manual"] = int(eligible_supported)
        rows.append(row)

        def _make_retrainer(
            estimator_template: object,
        ) -> Callable[[int], dict[str, object]]:
            def _run(seed: int) -> dict[str, object]:
                local_estimator = clone(estimator_template)
                _set_random_state_recursive(local_estimator, seed)
                local_estimator.fit(X_train, y_train)
                valid_scores_local = _predict_scores(local_estimator, X_valid)
                local_threshold_table = threshold_profit_curve(
                    y_valid,
                    valid_scores_local,
                    config=threshold_config,
                )
                local_threshold = float(
                    pick_profit_optimal_threshold(
                        local_threshold_table,
                        max_target_share=threshold_config.max_target_share,
                    )["threshold"]
                )
                test_scores_local = _predict_scores(local_estimator, X_test)
                metrics_local = evaluate_at_threshold(y_test, test_scores_local, threshold=local_threshold)

                return {
                    "metrics": metrics_local,
                    "threshold": local_threshold,
                    "threshold_table": local_threshold_table,
                    "test_scores": test_scores_local,
                    "score_fn": lambda frame,
                    est=local_estimator: _predict_scores(est, frame),
                }

            return _run

        retrainers[model_name] = _make_retrainer(estimator)

    benchmark = pd.DataFrame(rows)
    if benchmark.empty:
        return benchmark, benchmark, [], retrainers, {"lazy_raw_table": model_table}

    supported = benchmark.loc[benchmark["eligible_for_manual"] == 1].copy()
    if not supported.empty:
        pr_floor = supported["holdout_primary_metric"].quantile(0.35)
        supported["eligible_for_manual"] = (supported["holdout_primary_metric"] >= pr_floor).astype(int)
        benchmark = benchmark.drop(columns=["eligible_for_manual"]).merge(
            supported[["model_name", "eligible_for_manual"]], on="model_name", how="left"
        )
        benchmark["eligible_for_manual"] = benchmark["eligible_for_manual"].fillna(0).astype(int)

    benchmark = _rank_churn_candidates(benchmark)

    eligible = benchmark.loc[benchmark["eligible_for_manual"] == 1].copy()
    eligible = eligible.sort_values(["final_rank", "model_name"]).reset_index(drop=True)

    top_families: list[str] = []
    for _, row in eligible.iterrows():
        family = row.get("manual_family")
        if isinstance(family, str) and family not in top_families:
            top_families.append(family)
        if len(top_families) >= top_n_manual_families:
            break

    # If strict eligibility yields fewer than requested families, backfill from supported families.
    if len(top_families) < top_n_manual_families:
        supported_fallback = benchmark.loc[
            benchmark["manual_family"].notna()
        ].sort_values(["final_rank", "model_name"])
        for _, row in supported_fallback.iterrows():
            family = row.get("manual_family")
            if isinstance(family, str) and family not in top_families:
                top_families.append(family)
            if len(top_families) >= top_n_manual_families:
                break

    details = {
        "lazy_raw_table": model_table,
        "lazy_threshold_tables": threshold_tables,
    }
    return benchmark, eligible, top_families, retrainers, details


def run_manual_engineering_lab(
    manual_families: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_config: ThresholdConfig,
    project_name: str = "customer-segmentation-retention",
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, Callable[[int], dict[str, object]]], dict[str, pd.DataFrame]]:
    """Track 2: Manual training for top 3 LazyPredict-selected families only."""
    rows: list[dict[str, float | str]] = []
    details: dict[str, dict[str, object]] = {}
    retrainers: dict[str, Callable[[int], dict[str, object]]] = {}
    threshold_tables: dict[str, pd.DataFrame] = {}

    for family in manual_families:
        try:
            model, note = _build_manual_estimator(family, X_train, SEED)
        except ValueError:
            continue

        model_name = f"Manual_{family}"

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        cv_model, _ = _build_manual_estimator(family, X_train, SEED)
        cv_scores = cross_val_score(
            cv_model,
            pd.concat([X_train, X_valid], axis=0),
            pd.concat([y_train, y_valid], axis=0),
            scoring=_average_precision_scorer,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
            n_jobs=-1,
        )

        valid_scores = _predict_scores(model, X_valid)
        threshold_table = threshold_profit_curve(y_valid, valid_scores, config=threshold_config)
        chosen = pick_profit_optimal_threshold(threshold_table, max_target_share=threshold_config.max_target_share)
        threshold = float(chosen["threshold"])
        test_scores = _predict_scores(model, X_test)
        metrics = evaluate_at_threshold(y_test, test_scores, threshold=threshold)

        artifact_path = save_model_artifact(model, Path("artifacts") / "manual_models" / f"{model_name}.pkl")
        calibration_table = build_calibration_table(y_test, test_scores)
        error_table = build_error_analysis(y_test, test_scores, threshold)

        row = _build_churn_row(
            project_name=project_name,
            library_source="ManualEngineering",
            model_name=model_name,
            cv_metric_mean=float(np.mean(cv_scores)),
            cv_metric_std=float(np.std(cv_scores)),
            holdout_pr_auc=metrics["pr_auc"],
            holdout_recall=metrics["recall"],
            holdout_precision=metrics["precision"],
            brier_score=metrics["brier_score"],
            train_time_sec=train_time,
            infer_latency_ms=_estimate_infer_latency_ms(lambda data: _predict_scores(model, data), X_test),
            model_size_mb=_model_size_mb(model),
            interpretability_note=note,
        )
        rows.append(row)
        threshold_tables[model_name] = threshold_table

        details[model_name] = {
            "family": family,
            "model": model,
            "artifact_path": str(artifact_path),
            "metrics": metrics,
            "threshold": threshold,
            "threshold_table": threshold_table,
            "calibration_table": calibration_table,
            "error_table": error_table,
        }

        def _make_retrainer(family_name: str) -> Callable[[int], dict[str, object]]:
            def _run(seed: int) -> dict[str, object]:
                local_model, _ = _build_manual_estimator(family_name, X_train, seed)
                local_model.fit(X_train, y_train)

                local_valid_scores = _predict_scores(local_model, X_valid)
                local_threshold_table = threshold_profit_curve(
                    y_valid,
                    local_valid_scores,
                    config=threshold_config,
                )
                local_threshold = float(
                    pick_profit_optimal_threshold(
                        local_threshold_table,
                        max_target_share=threshold_config.max_target_share,
                    )["threshold"]
                )
                local_test_scores = _predict_scores(local_model, X_test)
                local_metrics = evaluate_at_threshold(y_test, local_test_scores, threshold=local_threshold)

                return {
                    "metrics": local_metrics,
                    "threshold": local_threshold,
                    "threshold_table": local_threshold_table,
                    "test_scores": local_test_scores,
                    "score_fn": lambda frame, mdl=local_model: _predict_scores(mdl, frame),
                }

            return _run

        retrainers[model_name] = _make_retrainer(family)

    return pd.DataFrame(rows), details, retrainers, threshold_tables


def run_flaml_optimization_lab(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_config: ThresholdConfig,
    project_name: str = "customer-segmentation-retention",
    time_budget_sec: int = 180,
    estimator_list: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, Callable[[int], dict[str, object]]], dict[str, pd.DataFrame]]:
    """Track 3: FLAML optimization with explicit budget and search diagnostics."""
    if estimator_list is None:
        estimator_list = ["rf", "xgboost", "extra_tree", "xgb_limitdepth", "sgd"]

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    preprocessor = _build_preprocessor(X_train, scale_numeric=False)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_valid_enc = preprocessor.transform(X_valid)
    X_test_enc = preprocessor.transform(X_test)

    automl = AutoML()
    start = time.perf_counter()
    automl.fit(
        X_train=X_train_enc,
        y_train=y_train,
        task="classification",
        metric="ap",
        time_budget=time_budget_sec,
        estimator_list=estimator_list,
        seed=SEED,
        log_file_name="artifacts/flaml_churn.log",
        verbose=0,
    )
    train_time = time.perf_counter() - start

    valid_scores = automl.predict_proba(X_valid_enc)[:, 1]
    threshold_table = threshold_profit_curve(y_valid, valid_scores, config=threshold_config)
    chosen = pick_profit_optimal_threshold(threshold_table, max_target_share=threshold_config.max_target_share)
    threshold = float(chosen["threshold"])

    test_scores = automl.predict_proba(X_test_enc)[:, 1]
    metrics = evaluate_at_threshold(y_test, test_scores, threshold=threshold)

    model_name = f"FLAML_{automl.best_estimator}"
    row = _build_churn_row(
        project_name=project_name,
        library_source="FLAML",
        model_name=model_name,
        cv_metric_mean=float(1.0 - automl.best_loss) if automl.best_loss is not None else float("nan"),
        cv_metric_std=0.0,
        holdout_pr_auc=metrics["pr_auc"],
        holdout_recall=metrics["recall"],
        holdout_precision=metrics["precision"],
        brier_score=metrics["brier_score"],
        train_time_sec=train_time,
        infer_latency_ms=_estimate_infer_latency_ms(lambda data: automl.predict_proba(data)[:, 1], X_test_enc),
        model_size_mb=_model_size_mb((preprocessor, automl.model)),
        interpretability_note="FLAML-searched candidate under explicit AP/time budget trade-off.",
    )

    details = {
        "searched_estimators": estimator_list,
        "best_estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
        "best_config_per_estimator": getattr(automl, "best_config_per_estimator", {}),
        "time_budget_sec": time_budget_sec,
        "threshold": threshold,
        "metrics": metrics,
    }

    def _flaml_retrainer(seed: int) -> dict[str, object]:
        local_preprocessor = clone(preprocessor)
        X_train_local = local_preprocessor.fit_transform(X_train)
        X_valid_local = local_preprocessor.transform(X_valid)
        X_test_local = local_preprocessor.transform(X_test)

        local_automl = AutoML()
        local_automl.fit(
            X_train=X_train_local,
            y_train=y_train,
            task="classification",
            metric="ap",
            time_budget=min(120, time_budget_sec),
            estimator_list=estimator_list,
            seed=seed,
            log_file_name=f"artifacts/flaml_churn_seed_{seed}.log",
            verbose=0,
        )

        local_valid_scores = local_automl.predict_proba(X_valid_local)[:, 1]
        local_threshold_table = threshold_profit_curve(
            y_valid,
            local_valid_scores,
            config=threshold_config,
        )
        local_threshold = float(
            pick_profit_optimal_threshold(
                local_threshold_table,
                max_target_share=threshold_config.max_target_share,
            )["threshold"]
        )

        local_test_scores = local_automl.predict_proba(X_test_local)[:, 1]
        local_metrics = evaluate_at_threshold(y_test, local_test_scores, threshold=local_threshold)

        return {
            "metrics": local_metrics,
            "threshold": local_threshold,
            "threshold_table": local_threshold_table,
            "test_scores": local_test_scores,
            "score_fn": lambda frame,
            prep=local_preprocessor,
            automl_model=local_automl: automl_model.predict_proba(prep.transform(frame))[:, 1],
        }

    return (
        pd.DataFrame([row]),
        details,
        {model_name: _flaml_retrainer},
        {model_name: threshold_table},
    )


def run_pycaret_experiment_lab(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_config: ThresholdConfig,
    project_name: str = "customer-segmentation-retention",
    include_models: list[str] | None = None,
    compare_n_select: int = 3,
    tune_iterations: int = 15,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, Callable[[int], dict[str, object]]], dict[str, pd.DataFrame]]:
    """Track 4: PyCaret orchestration lab with compare/tune/calibrate/finalize/save."""
    if include_models is None:
        include_models = ["lr", "rf", "et", "xgboost", "gbc", "ada", "dt", "knn", "ridge"]

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["churn"] = y_train.values

    if len(train_df) > 7000:
        train_df = (
            train_df.groupby("churn", group_keys=False)
            .apply(lambda g: g.sample(n=max(1, int(7000 * len(g) / len(train_df))), random_state=SEED))
            .reset_index(drop=True)
        )

    exp = ClassificationExperiment(
        target="churn",
        session_id=SEED,
        fold=3,
        n_jobs=1,
        verbose=False,
    )

    setup_start = time.perf_counter()
    exp.fit(train_df)
    setup_time = time.perf_counter() - setup_start

    available_ids = set(exp.models().index.astype(str).tolist())
    active_ids = [model_id for model_id in include_models if model_id in available_ids]
    if not active_ids:
        raise RuntimeError("No requested PyCaret models are available in this environment.")

    compare_start = time.perf_counter()
    compare_result = exp.compare_models(
        include=active_ids,
        n_select=min(compare_n_select, len(active_ids)),
        sort="AUC",
        turbo=True,
        verbose=False,
    )
    compare_time = time.perf_counter() - compare_start

    compare_leaderboard = getattr(compare_result, "leaderboard", exp.pull().copy())
    best_model = compare_result.best if hasattr(compare_result, "best") else compare_result
    ranked_ids = list(getattr(compare_result, "ranked_ids", []))
    best_model_id = ranked_ids[0] if ranked_ids else "best"

    tune_start = time.perf_counter()
    tune_result = exp.tune_model(best_model, optimize="AUC", n_iter=tune_iterations, verbose=False)
    tune_time = time.perf_counter() - tune_start
    tuned_model = getattr(tune_result, "pipeline", tune_result)
    tune_metrics = getattr(tune_result, "metrics", exp.pull().copy())

    calibrate_result_obj = None
    calibrate_metrics = None
    calibrate_time = 0.0
    try:
        calibrate_start = time.perf_counter()
        calibrate_result_obj = exp.calibrate_model(tuned_model, method="sigmoid", verbose=False)
        calibrate_time = time.perf_counter() - calibrate_start
        calibrated_model = getattr(calibrate_result_obj, "pipeline", calibrate_result_obj)
        calibrate_metrics = getattr(calibrate_result_obj, "metrics", None)
    except Exception:
        calibrated_model = tuned_model

    finalize_start = time.perf_counter()
    finalize_result = exp.finalize_model(calibrated_model)
    finalize_time = time.perf_counter() - finalize_start
    final_model = getattr(finalize_result, "pipeline", finalize_result)

    saved_model_path = exp.save_model(final_model, "artifacts/pycaret_final_model", verbose=False)

    valid_scores = _predict_scores(final_model, X_valid)
    threshold_table = threshold_profit_curve(y_valid, valid_scores, config=threshold_config)
    chosen = pick_profit_optimal_threshold(threshold_table, max_target_share=threshold_config.max_target_share)
    threshold = float(chosen["threshold"])

    test_scores = _predict_scores(final_model, X_test)
    metrics = evaluate_at_threshold(y_test, test_scores, threshold=threshold)

    cv_auc_mean = float("nan")
    cv_auc_std = float("nan")
    if isinstance(tune_metrics, pd.DataFrame) and "AUC" in tune_metrics.columns:
        if "Mean" in tune_metrics.index:
            cv_auc_mean = float(tune_metrics.loc["Mean", "AUC"])
            cv_auc_std = float(tune_metrics.loc["Std", "AUC"]) if "Std" in tune_metrics.index else 0.0
        else:
            cv_auc_mean = float(tune_metrics["AUC"].mean())
            cv_auc_std = float(tune_metrics["AUC"].std(ddof=0))

    model_name = f"PyCaret_Final_{best_model_id}"
    total_train_time = setup_time + compare_time + tune_time + calibrate_time + finalize_time

    row = _build_churn_row(
        project_name=project_name,
        library_source="PyCaret",
        model_name=model_name,
        cv_metric_mean=cv_auc_mean,
        cv_metric_std=cv_auc_std,
        holdout_pr_auc=metrics["pr_auc"],
        holdout_recall=metrics["recall"],
        holdout_precision=metrics["precision"],
        brier_score=metrics["brier_score"],
        train_time_sec=total_train_time,
        infer_latency_ms=_estimate_infer_latency_ms(lambda data: _predict_scores(final_model, data), X_test),
        model_size_mb=_model_size_mb(final_model),
        interpretability_note="PyCaret-orchestrated candidate (compare/tune/calibrate/finalize).",
    )

    details = {
        "setup_time_sec": setup_time,
        "compare_time_sec": compare_time,
        "tune_time_sec": tune_time,
        "calibrate_time_sec": calibrate_time,
        "finalize_time_sec": finalize_time,
        "include_models": active_ids,
        "compare_leaderboard": compare_leaderboard,
        "tune_metrics": tune_metrics,
        "calibrate_metrics": calibrate_metrics,
        "best_model_id": best_model_id,
        "saved_model_path": saved_model_path,
        "threshold": threshold,
        "metrics": metrics,
    }

    def _pycaret_retrainer(seed: int) -> dict[str, object]:
        local_model = clone(final_model)
        _set_random_state_recursive(local_model, seed)
        local_model.fit(X_train, y_train)

        local_valid_scores = _predict_scores(local_model, X_valid)
        local_threshold_table = threshold_profit_curve(
            y_valid,
            local_valid_scores,
            config=threshold_config,
        )
        local_threshold = float(
            pick_profit_optimal_threshold(
                local_threshold_table,
                max_target_share=threshold_config.max_target_share,
            )["threshold"]
        )

        local_test_scores = _predict_scores(local_model, X_test)
        local_metrics = evaluate_at_threshold(y_test, local_test_scores, threshold=local_threshold)

        return {
            "metrics": local_metrics,
            "threshold": local_threshold,
            "threshold_table": local_threshold_table,
            "test_scores": local_test_scores,
            "score_fn": lambda frame, mdl=local_model: _predict_scores(mdl, frame),
        }

    return (
        pd.DataFrame([row]),
        details,
        {model_name: _pycaret_retrainer},
        {model_name: threshold_table},
    )


def build_unified_leaderboard(
    lazy_benchmark: pd.DataFrame,
    manual_results: pd.DataFrame,
    flaml_result: pd.DataFrame,
    pycaret_result: pd.DataFrame,
    baseline_result: pd.DataFrame | None = None,
    lazy_top_n: int = 5,
) -> pd.DataFrame:
    """Combine the serious tracks into a single ranked leaderboard."""
    lazy_top = lazy_benchmark.sort_values("final_rank").head(lazy_top_n).copy()

    frames = [lazy_top, manual_results, flaml_result, pycaret_result]
    if baseline_result is not None and not baseline_result.empty:
        frames.append(baseline_result)

    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    required_cols = [
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
        "model_size_mb",
        "interpretability_note",
    ]
    for col in required_cols:
        if col not in combined.columns:
            combined[col] = np.nan

    ranked = _rank_churn_candidates(combined[required_cols])
    return ranked[UNIFIED_LEADERBOARD_COLUMNS]


def rerun_top_candidates_across_seeds(
    leaderboard: pd.DataFrame,
    retrainers: dict[str, Callable[[int], dict[str, object]]],
    seeds: tuple[int, int, int] = (42, 77, 2026),
    top_n: int = 3,
) -> pd.DataFrame:
    """Re-run top leaderboard candidates to validate seed robustness."""
    top_models = leaderboard.sort_values("final_rank").head(top_n)

    rows: list[dict[str, float | str | int]] = []
    for _, candidate in top_models.iterrows():
        model_name = str(candidate["model_name"])
        retrainer = retrainers.get(model_name)
        if retrainer is None:
            rows.append(
                {
                    "model_name": model_name,
                    "library_source": candidate["library_source"],
                    "final_rank": int(candidate["final_rank"]),
                    "seeds_used": ",".join(str(seed) for seed in seeds),
                    "status": "not_retrainable",
                    "pr_auc_mean": float("nan"),
                    "pr_auc_std": float("nan"),
                    "recall_mean": float("nan"),
                    "recall_std": float("nan"),
                    "precision_mean": float("nan"),
                    "precision_std": float("nan"),
                    "brier_mean": float("nan"),
                    "brier_std": float("nan"),
                }
            )
            continue

        seed_metrics: list[dict[str, float]] = []
        for seed in seeds:
            result = retrainer(seed)
            seed_metrics.append(
                {
                    "pr_auc": float(result["metrics"]["pr_auc"]),
                    "recall": float(result["metrics"]["recall"]),
                    "precision": float(result["metrics"]["precision"]),
                    "brier": float(result["metrics"]["brier_score"]),
                }
            )

        metric_df = pd.DataFrame(seed_metrics)
        rows.append(
            {
                "model_name": model_name,
                "library_source": candidate["library_source"],
                "final_rank": int(candidate["final_rank"]),
                "seeds_used": ",".join(str(seed) for seed in seeds),
                "status": "ok",
                "pr_auc_mean": float(metric_df["pr_auc"].mean()),
                "pr_auc_std": float(metric_df["pr_auc"].std(ddof=0)),
                "recall_mean": float(metric_df["recall"].mean()),
                "recall_std": float(metric_df["recall"].std(ddof=0)),
                "precision_mean": float(metric_df["precision"].mean()),
                "precision_std": float(metric_df["precision"].std(ddof=0)),
                "brier_mean": float(metric_df["brier"].mean()),
                "brier_std": float(metric_df["brier"].std(ddof=0)),
            }
        )

    return pd.DataFrame(rows).sort_values("final_rank").reset_index(drop=True)


def segment_kpi_summary(
    frame: pd.DataFrame,
    segment_col: str = "segment_label",
    churn_col: str = "churn",
    ltv_col: str = "ltv_proxy",
    revenue_col: str = "monetary_value",
) -> pd.DataFrame:
    return (
        frame.groupby(segment_col, as_index=False)
        .agg(
            customers=("CustomerID", "nunique"),
            churn_rate=(churn_col, "mean"),
            avg_ltv=(ltv_col, "mean"),
            avg_revenue=(revenue_col, "mean"),
        )
        .sort_values(["avg_ltv", "churn_rate"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_action_policy(
    frame: pd.DataFrame,
    churn_score_col: str,
    ltv_col: str = "ltv_proxy",
    retention_threshold: float = 0.55,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    high_ltv_cut = output[ltv_col].quantile(0.70)
    low_ltv_cut = output[ltv_col].quantile(0.30)

    conditions = [
        output[churn_score_col] >= retention_threshold,
        (output[churn_score_col] < retention_threshold * 0.65) & (output[ltv_col] >= high_ltv_cut),
        (output[churn_score_col] < retention_threshold * 0.65) & (output[ltv_col] <= low_ltv_cut),
    ]
    choices = ["retention_offer", "premium_upsell", "low_priority"]
    output["action_policy"] = np.select(conditions, choices, default="monitor")

    summary = (
        output.groupby("action_policy", as_index=False)
        .agg(
            customers=("CustomerID", "nunique"),
            avg_churn_score=(churn_score_col, "mean"),
            avg_ltv=(ltv_col, "mean"),
            observed_churn_rate=("churn", "mean"),
        )
        .sort_values("customers", ascending=False)
        .reset_index(drop=True)
    )
    return output, summary


# Backward-compatible wrapper for older notebook versions.
def repeat_top_churn_models(
    churn_leaderboard: pd.DataFrame,
    retrainers: dict[str, Callable[[int], dict[str, object]]],
    seeds: tuple[int, int, int] = (42, 77, 2026),
    top_n: int = 3,
) -> pd.DataFrame:
    return rerun_top_candidates_across_seeds(
        churn_leaderboard,
        retrainers,
        seeds=seeds,
        top_n=top_n,
    )
