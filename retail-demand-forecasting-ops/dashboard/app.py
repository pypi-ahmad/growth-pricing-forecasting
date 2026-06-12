from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

st.set_page_config(page_title="Retail Forecasting Ops", layout="wide")
st.title("Retail Demand Forecasting Ops Dashboard")

leaderboard_path = ARTIFACT_DIR / "leaderboard_retail_forecasting.csv"
ops_path = ARTIFACT_DIR / "ops_signals.csv"

if leaderboard_path.exists():
    st.subheader("Unified Leaderboard")
    leaderboard = pd.read_csv(leaderboard_path)
    st.dataframe(leaderboard.head(20), use_container_width=True)
else:
    st.info("Leaderboard artifact not found yet. Run notebook first.")

if ops_path.exists():
    st.subheader("Operations Signals")
    ops = pd.read_csv(ops_path)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(ops))
    c2.metric("Spike Alerts", int(ops.get("demand_spike_alert", pd.Series(dtype=bool)).sum()))
    c3.metric("Avg Staffing Proxy", round(float(ops.get("staffing_proxy", pd.Series([0])).mean()), 2))

    st.line_chart(ops[[c for c in ["Date", "prediction", "Sales"] if c in ops.columns]].set_index("Date") if "Date" in ops.columns else ops[[c for c in ["prediction", "Sales"] if c in ops.columns]])
    st.dataframe(ops.head(200), use_container_width=True)
else:
    st.info("Operations signal artifact not found yet. Run notebook first.")
