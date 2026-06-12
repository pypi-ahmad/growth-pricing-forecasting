from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "name_tfidf",
                TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2),
                "name",
            ),
            (
                "desc_tfidf",
                TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=3),
                "item_description",
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                ["category_name", "brand_name"],
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                ["shipping", "item_condition_id"],
            ),
        ],
        remainder="drop",
    )


def prepare_feature_matrices(train_df: pd.DataFrame, holdout_df: pd.DataFrame, preprocessor: ColumnTransformer, random_state: int = 42):
    feature_cols = [
        "name",
        "item_description",
        "category_name",
        "brand_name",
        "shipping",
        "item_condition_id",
    ]

    X_train_raw = train_df[feature_cols].copy()
    X_holdout_raw = holdout_df[feature_cols].copy()

    y_train_log = np.log1p(train_df["price"].values)
    y_holdout_price = holdout_df["price"].values

    X_train_full = preprocessor.fit_transform(X_train_raw)
    X_holdout_full = preprocessor.transform(X_holdout_raw)

    max_comp = min(160, X_train_full.shape[1] - 1) if X_train_full.shape[1] > 2 else 2
    svd = TruncatedSVD(n_components=max_comp, random_state=random_state)
    X_train_lazy = svd.fit_transform(X_train_full)
    X_holdout_lazy = svd.transform(X_holdout_full)

    holdout_meta = holdout_df[["category_name", "price"]].copy()

    return {
        "X_train_full": X_train_full,
        "X_holdout_full": X_holdout_full,
        "X_train_lazy": X_train_lazy,
        "X_holdout_lazy": X_holdout_lazy,
        "y_train_log": y_train_log,
        "y_holdout_price": y_holdout_price,
        "holdout_meta": holdout_meta,
        "svd": svd,
    }
