"""Feature engineering helpers for customer segmentation and retention analytics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 42

REQUIRED_RETAIL_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}

SEGMENTATION_FEATURE_COLUMNS = [
    "recency_days",
    "frequency_orders",
    "monetary_value",
    "avg_order_value",
    "active_months",
    "avg_days_between_orders",
    "order_quantity_total",
    "distinct_products",
    "return_rate",
    "inactivity_ratio",
]

MODEL_FEATURE_COLUMNS = [
    "recency_days",
    "frequency_orders",
    "monetary_value",
    "avg_order_value",
    "active_months",
    "avg_days_between_orders",
    "std_days_between_orders",
    "order_quantity_total",
    "distinct_products",
    "return_rate",
    "inactivity_ratio",
    "country_txn_share",
]


@dataclass(frozen=True)
class FeatureBuildConfig:
    """Controls history/label windows for churn feature generation."""

    history_window_days: int = 365
    churn_horizon_days: int = 90
    min_orders: int = 2
    min_customer_tenure_days: int = 30


def load_online_retail(path: str) -> pd.DataFrame:
    """Load the Online Retail dataset and validate expected schema."""
    frame = pd.read_csv(path, parse_dates=["InvoiceDate"], dayfirst=False, encoding="ISO-8859-1")
    missing = REQUIRED_RETAIL_COLUMNS - set(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")
    return frame


def clean_online_retail(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean raw transaction rows and create transaction-level behavior flags."""
    cleaned = frame.copy()
    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype(str).str.strip()
    cleaned["CustomerID"] = pd.to_numeric(cleaned["CustomerID"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")

    cleaned = cleaned.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]).copy()
    cleaned = cleaned.loc[cleaned["UnitPrice"] > 0].copy()
    cleaned["CustomerID"] = cleaned["CustomerID"].astype("int64").astype("string")

    cleaned["is_return"] = cleaned["InvoiceNo"].str.startswith("C") | (cleaned["Quantity"] < 0)
    cleaned["line_amount"] = cleaned["Quantity"] * cleaned["UnitPrice"]
    cleaned["purchase_amount"] = np.where(
        (~cleaned["is_return"]) & (cleaned["Quantity"] > 0),
        cleaned["line_amount"],
        0.0,
    )
    cleaned["return_amount"] = np.where(cleaned["is_return"], np.abs(cleaned["line_amount"]), 0.0)
    return cleaned.reset_index(drop=True)


def build_data_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a compact, notebook-friendly data dictionary summary."""
    rows: list[dict[str, object]] = []
    for col in frame.columns:
        series = frame[col]
        sample = series.dropna().iloc[0] if series.notna().any() else None
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_pct": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
                "example_value": sample,
            }
        )
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def run_leakage_checks(
    transactions: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    config: FeatureBuildConfig,
) -> pd.DataFrame:
    """Return a compact leakage diagnostic table for notebook reporting."""
    history_start = snapshot_date - pd.Timedelta(days=config.history_window_days)
    label_end = snapshot_date + pd.Timedelta(days=config.churn_horizon_days)

    history_rows = transactions["InvoiceDate"].between(history_start, snapshot_date).sum()
    post_snapshot_rows = (transactions["InvoiceDate"] > snapshot_date).sum()
    label_window_rows = transactions["InvoiceDate"].between(snapshot_date, label_end).sum()
    future_beyond_label_rows = (transactions["InvoiceDate"] > label_end).sum()

    return pd.DataFrame(
        {
            "check": [
                "history_window_rows",
                "rows_after_snapshot",
                "rows_within_label_window",
                "rows_after_label_window",
            ],
            "value": [
                int(history_rows),
                int(post_snapshot_rows),
                int(label_window_rows),
                int(future_beyond_label_rows),
            ],
            "expected": [
                "Used for features",
                "Should be excluded from features",
                "Used for churn labels only",
                "Should not affect labels",
            ],
        }
    )


def _order_interval_features(invoice_level: pd.DataFrame) -> pd.DataFrame:
    order_dates = (
        invoice_level[["CustomerID", "order_date"]]
        .drop_duplicates()
        .sort_values(["CustomerID", "order_date"])
    )
    order_dates["prev_order_date"] = order_dates.groupby("CustomerID")["order_date"].shift(1)
    order_dates["days_between_orders"] = (
        order_dates["order_date"] - order_dates["prev_order_date"]
    ).dt.days

    return (
        order_dates.groupby("CustomerID", as_index=False)["days_between_orders"]
        .agg(
            avg_days_between_orders="mean",
            std_days_between_orders="std",
            max_days_between_orders="max",
        )
        .fillna(0.0)
    )


def build_customer_features(
    transactions: pd.DataFrame,
    config: FeatureBuildConfig | None = None,
    snapshot_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build customer-level RFM + behavior features.

    Churn label logic:
    - Build features from [snapshot - history_window, snapshot].
    - Label churn=1 if customer has no positive purchase in (snapshot, snapshot + churn_horizon].
    """
    if config is None:
        config = FeatureBuildConfig()

    positive_purchase_mask = (~transactions["is_return"]) & (transactions["Quantity"] > 0)
    purchases = transactions.loc[positive_purchase_mask].copy()
    if purchases.empty:
        raise ValueError("No positive purchase rows available after cleaning.")

    if snapshot_date is None:
        snapshot_date = purchases["InvoiceDate"].max() - pd.Timedelta(days=config.churn_horizon_days)
    snapshot_date = pd.Timestamp(snapshot_date)

    history_start = snapshot_date - pd.Timedelta(days=config.history_window_days)
    label_end = snapshot_date + pd.Timedelta(days=config.churn_horizon_days)

    history_all = transactions.loc[
        transactions["InvoiceDate"].between(history_start, snapshot_date)
    ].copy()
    history_purchases = purchases.loc[
        purchases["InvoiceDate"].between(history_start, snapshot_date)
    ].copy()
    future_purchases = purchases.loc[
        (purchases["InvoiceDate"] > snapshot_date) & (purchases["InvoiceDate"] <= label_end)
    ].copy()

    if history_purchases.empty:
        raise ValueError("History window contains no purchases. Adjust snapshot/window.")

    invoice_level = (
        history_purchases.groupby(["CustomerID", "InvoiceNo"], as_index=False)
        .agg(
            order_date=("InvoiceDate", "max"),
            order_value=("purchase_amount", "sum"),
            order_quantity=("Quantity", "sum"),
            unique_products=("StockCode", "nunique"),
            order_lines=("StockCode", "count"),
        )
        .sort_values(["CustomerID", "order_date"])
    )

    customer_features = (
        invoice_level.groupby("CustomerID", as_index=False)
        .agg(
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max"),
            frequency_orders=("InvoiceNo", "nunique"),
            monetary_value=("order_value", "sum"),
            avg_order_value=("order_value", "mean"),
            order_quantity_total=("order_quantity", "sum"),
            avg_lines_per_order=("order_lines", "mean"),
        )
        .assign(
            recency_days=lambda x: (snapshot_date - x["last_order_date"]).dt.days,
            customer_tenure_days=lambda x: (snapshot_date - x["first_order_date"]).dt.days.clip(lower=0)
            + 1,
        )
    )

    product_features = (
        history_purchases.groupby("CustomerID", as_index=False)["StockCode"]
        .nunique()
        .rename(columns={"StockCode": "distinct_products"})
    )
    customer_features = customer_features.merge(product_features, on="CustomerID", how="left")

    active_months = (
        history_purchases.assign(month=history_purchases["InvoiceDate"].dt.to_period("M"))
        .groupby("CustomerID", as_index=False)["month"]
        .nunique()
        .rename(columns={"month": "active_months"})
    )
    customer_features = customer_features.merge(active_months, on="CustomerID", how="left")

    customer_features["order_frequency_per_month"] = (
        customer_features["frequency_orders"]
        / (customer_features["customer_tenure_days"] / 30.0).clip(lower=1 / 30)
    )

    interval_features = _order_interval_features(invoice_level)
    customer_features = customer_features.merge(interval_features, on="CustomerID", how="left")

    return_features = (
        history_all.groupby("CustomerID", as_index=False)
        .agg(
            return_events=("is_return", "sum"),
            return_amount=("return_amount", "sum"),
            total_lines=("InvoiceNo", "count"),
        )
        .assign(return_rate=lambda x: x["return_events"] / x["total_lines"].clip(lower=1))
    )
    customer_features = customer_features.merge(
        return_features[["CustomerID", "return_rate", "return_amount"]],
        on="CustomerID",
        how="left",
    )

    country_share = (
        history_all.groupby(["CustomerID", "Country"], as_index=False)
        .size()
        .rename(columns={"size": "country_txn_count"})
    )
    top_country = (
        country_share.sort_values(["CustomerID", "country_txn_count"], ascending=[True, False])
        .drop_duplicates("CustomerID")
        .rename(columns={"Country": "primary_country"})
    )
    total_txn = history_all.groupby("CustomerID", as_index=False).size().rename(columns={"size": "total_txn"})
    top_country = top_country.merge(total_txn, on="CustomerID", how="left")
    top_country["country_txn_share"] = top_country["country_txn_count"] / top_country["total_txn"].clip(lower=1)
    customer_features = customer_features.merge(
        top_country[["CustomerID", "primary_country", "country_txn_share"]],
        on="CustomerID",
        how="left",
    )

    future_orders = (
        future_purchases.groupby("CustomerID", as_index=False)["InvoiceNo"]
        .nunique()
        .rename(columns={"InvoiceNo": "future_orders"})
    )
    customer_features = customer_features.merge(future_orders, on="CustomerID", how="left")
    customer_features["future_orders"] = customer_features["future_orders"].fillna(0).astype(int)
    customer_features["churn"] = (customer_features["future_orders"] == 0).astype(int)

    customer_features["inactivity_ratio"] = customer_features["recency_days"] / (
        customer_features["avg_days_between_orders"].replace(0, np.nan).fillna(30) + 1
    )

    customer_features = customer_features.loc[
        (customer_features["frequency_orders"] >= config.min_orders)
        & (customer_features["customer_tenure_days"] >= config.min_customer_tenure_days)
    ].copy()

    customer_features["observation_end"] = snapshot_date
    customer_features["label_window_end"] = label_end

    numeric_cols = customer_features.select_dtypes(include=[np.number]).columns
    customer_features[numeric_cols] = customer_features[numeric_cols].replace([np.inf, -np.inf], np.nan)
    fill_zero_cols = [
        "return_rate",
        "return_amount",
        "avg_days_between_orders",
        "std_days_between_orders",
        "max_days_between_orders",
        "country_txn_share",
        "inactivity_ratio",
        "distinct_products",
    ]
    for col in fill_zero_cols:
        if col in customer_features.columns:
            customer_features[col] = customer_features[col].fillna(0.0)

    customer_features["primary_country"] = customer_features["primary_country"].fillna("Unknown")
    return customer_features.sort_values("CustomerID").reset_index(drop=True)


def add_ltv_proxy(
    customer_df: pd.DataFrame,
    margin_rate: float = 0.35,
    annualization_factor: int = 12,
) -> pd.DataFrame:
    """Add a practical LTV proxy for retention + upsell prioritization."""
    enriched = customer_df.copy()

    monthly_revenue = enriched["monetary_value"] / enriched["active_months"].clip(lower=1)
    frequency_component = enriched["frequency_orders"].rank(pct=True)
    order_value_component = enriched["avg_order_value"].rank(pct=True)
    assortment_component = enriched["distinct_products"].rank(pct=True)
    engagement_score = 0.45 * frequency_component + 0.35 * order_value_component + 0.20 * assortment_component

    recency_risk = enriched["recency_days"].rank(pct=True)
    inactivity_risk = enriched["inactivity_ratio"].rank(pct=True)
    churn_risk_proxy = 0.6 * recency_risk + 0.4 * inactivity_risk

    enriched["ltv_proxy"] = (
        monthly_revenue
        * annualization_factor
        * margin_rate
        * (0.7 + engagement_score)
        * (1 - 0.65 * churn_risk_proxy)
    ).clip(lower=0.0)
    enriched["ltv_percentile"] = enriched["ltv_proxy"].rank(pct=True)
    return enriched
