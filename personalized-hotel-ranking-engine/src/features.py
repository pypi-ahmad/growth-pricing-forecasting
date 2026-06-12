from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date_time" in out.columns:
        out["search_month"] = out["date_time"].dt.month.fillna(0).astype(int)
        out["search_dayofweek"] = out["date_time"].dt.dayofweek.fillna(0).astype(int)

    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].fillna("unknown").astype(str)

    return out


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
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


def prepare_model_inputs(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    target_col: str = "label",
    group_col: str = "srch_id",
    item_col: str = "hotel_cluster",
):
    train = build_feature_frame(train_df)
    holdout = build_feature_frame(holdout_df)

    drop_cols = [target_col, group_col, "date_time"]
    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    y_train = train[target_col].astype(int)

    X_holdout = holdout.drop(columns=[c for c in drop_cols if c in holdout.columns])
    y_holdout = holdout[target_col].astype(int)

    X_train_enc = preprocessor.fit_transform(X_train)
    X_holdout_enc = preprocessor.transform(X_holdout)

    holdout_meta = holdout[[group_col, item_col, target_col]].copy()
    return X_train_enc, X_holdout_enc, y_train, y_holdout, holdout_meta
