from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def prepare_model_inputs(train_df: pd.DataFrame, holdout_df: pd.DataFrame, target_col: str, preprocessor: ColumnTransformer):
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    X_holdout = holdout_df.drop(columns=[target_col])
    y_holdout = holdout_df[target_col].astype(int)

    X_train_enc = preprocessor.fit_transform(X_train)
    X_holdout_enc = preprocessor.transform(X_holdout)

    return X_train_enc, X_holdout_enc, y_train, y_holdout
