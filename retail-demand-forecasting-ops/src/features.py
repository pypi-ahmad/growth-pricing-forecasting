from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def engineer_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    out["weekofyear"] = out["Date"].dt.isocalendar().week.astype(int)
    out["day"] = out["Date"].dt.day
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_month_start"] = out["Date"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["Date"].dt.is_month_end.astype(int)
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Sales",
    lags: tuple[int, ...] = (1, 7, 14, 28),
    windows: tuple[int, ...] = (7, 14, 28),
) -> pd.DataFrame:
    out = df.copy().sort_values(["Store", "Date"]) 
    group = out.groupby("Store", group_keys=False)

    for lag in lags:
        out[f"lag_{lag}"] = group[target_col].shift(lag)

    for window in windows:
        out[f"roll_mean_{window}"] = group[target_col].shift(1).rolling(window).mean()
        out[f"roll_std_{window}"] = group[target_col].shift(1).rolling(window).std()

    return out


def build_supervised_dataset(df: pd.DataFrame, target_col: str = "Sales") -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=[target_col])
    lag_cols = [c for c in out.columns if c.startswith("lag_") or c.startswith("roll_")]
    out = out.dropna(subset=lag_cols)
    return out.reset_index(drop=True)


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    feature_df = feature_df.copy()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        feature_df[col] = feature_df[col].fillna("missing").astype(str)
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)],
        remainder="drop",
    )


def prepare_matrices(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str,
    preprocessor: ColumnTransformer,
):
    drop_cols = [target_col, "Date"]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df[target_col].astype(float)

    X_holdout = holdout_df.drop(columns=[c for c in drop_cols if c in holdout_df.columns])
    y_holdout = holdout_df[target_col].astype(float)

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        X_train[col] = X_train[col].fillna("missing").astype(str)
        if col in X_holdout.columns:
            X_holdout[col] = X_holdout[col].fillna("missing").astype(str)

    X_train_enc = preprocessor.fit_transform(X_train)
    X_holdout_enc = preprocessor.transform(X_holdout)

    return X_train_enc, X_holdout_enc, y_train, y_holdout
