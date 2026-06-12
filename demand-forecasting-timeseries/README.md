# Demand Forecasting Time Series

Business-oriented, production-aware forecasting project for hourly electricity demand.

The notebook is structured as one end-to-end project with four serious modeling tracks:
1. LazyPredict Discovery Lab
2. Manual Engineering Lab
3. FLAML Optimization Lab
4. PyCaret Experiment Lab

## Project Goal

Forecast demand with strong operational trade-off analysis:
- Accuracy (`sMAPE`, `MAE`, `RMSE`)
- Stability/calibration (`calibration_metric`)
- Latency and deployability
- Interpretability vs performance decisions

## Setup and Dataset

```bash
git clone https://github.com/pypi-ahmad/demand-forecasting-timeseries.git
cd demand-forecasting-timeseries
```


Primary dataset:
- UCI Individual Household Electric Power Consumption
- Source: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

Expected data file:
- `data/raw/uci/household_power_consumption.txt`

Download (if needed):

```bash
cd demand-forecasting-timeseries
mkdir -p data/raw/uci
wget -O data/raw/uci/household_power.zip "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
unzip -o data/raw/uci/household_power.zip -d data/raw/uci
```

Environment:
- Python: `3.12.10` (pinned)
- Package manager: `uv`

## Current Workflow

Main notebook:
- `demand_forecasting_timeseries.ipynb`

Notebook sections:
1. Business Problem and Success Criteria
2. Dataset Access and Data Dictionary
3. Data Cleaning and Leakage Checks
4. Feature Engineering
5. Validation Strategy
6. LazyPredict Discovery Lab
7. Selection of Top 3 Eligible Models
8. Manual Engineering Lab
9. FLAML Optimization Lab
10. PyCaret Experiment Lab
11. Unified Leaderboard and Final Model Ranking
12. Business Recommendation
13. Inference / Deployment Path
14. Monitoring / Drift / Retraining Plan
15. Limitations and Next Steps

## LazyPredict -> Top 3 Manual Rule

Strict selection rule implemented in notebook:
- LazyPredict runs only after feature matrix and time-aware validation are defined.
- A ranked discovery table is created.
- Ineligible families are filtered (speed/stability/support constraints).
- Top 3 eligible families are selected.
- Manual lab uses only those top 3 families.

## FLAML Optimization Workflow

FLAML is a full optimization track, not a single compare row:
- Explicit `time_budget`
- Custom objective uses primary metric (`sMAPE`)
- Time-aware CV (`split_type='time'`)
- Best estimator/config inspection
- Search history table
- Recursive holdout evaluation
- Comparison against manual-track best

## PyCaret Experiment Workflow

PyCaret is used as a full experiment orchestration track:
- Regression experiment with compare/tune/finalize/save (blend when justified)
- Time-series experiment with compare/tune/finalize/save
- Recursive holdout scoring for fair comparison
- Saved model artifacts for deployable candidates

Note:
- PyCaret 4 uses OOP `Experiment(...).fit(...)`; legacy functional `setup()` is not used in this codebase.

## Final Leaderboard Logic

Unified leaderboard includes:
- Top LazyPredict discovery results
- Manual top-3 engineered models
- Best FLAML result
- Best PyCaret finalized result(s)
- Baseline models (naive/seasonal naive/SARIMA and optional Prophet)

Saved file:
- `artifacts/leaderboard_forecasting.csv`

Columns:
- `project_name`
- `task_type`
- `library_source`
- `model_name`
- `cv_metric_mean`
- `cv_metric_std`
- `holdout_primary_metric`
- `holdout_secondary_metric`
- `holdout_tertiary_metric`
- `calibration_metric`
- `train_time_sec`
- `infer_latency_ms`
- `model_size_mb`
- `interpretability_note`
- `rank_score`
- `final_rank`

Ranking:
- Weighted by metric ranks with primary emphasis on `sMAPE`
- Top 10 shown in notebook
- Top 3 final candidates are re-evaluated with multiple seeds where relevant

## Deployment and Monitoring Path

Deployment artifacts:
- model `.pkl` files saved in `artifacts/models/`

Operational outputs:
- `artifacts/horizon_rankings.csv`
- `artifacts/horizon_metrics.csv`
- `artifacts/top3_seed_stability.csv`
- `artifacts/monitoring_plan.csv`

Monitoring plan includes:
- quality degradation triggers
- drift triggers
- peak miss risk rules
- scheduled retraining cadence

## Exact Run Instructions

```bash
cd demand-forecasting-timeseries
uv sync
uv run jupyter notebook demand_forecasting_timeseries.ipynb
```

Optional kernel registration:

```bash
uv run python -m ipykernel install --user --name demand-forecasting-timeseries --display-name "Demand Forecasting (uv)"
```

Optional deep-learning extra (for LSTM extensions):

```bash
uv sync --extra deep-learning
```

