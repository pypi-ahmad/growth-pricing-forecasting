# retail-demand-forecasting-ops

## Project goal
Build a demand forecasting and store-operations planning system with time-aware backtesting and business action signals (staffing, replenishment, spike alerts).

## Dataset
- Competition: Rossmann Store Sales
- Link: https://www.kaggle.com/competitions/rossmann-store-sales

Download commands:
```bash
mkdir -p data/raw/rossmann
kaggle competitions download -c rossmann-store-sales -p data/raw/rossmann
unzip data/raw/rossmann/rossmann-store-sales.zip -d data/raw/rossmann
```

## Setup (uv + Python 3.12.10)

```bash
git clone https://github.com/pypi-ahmad/retail-demand-forecasting-ops.git
cd retail-demand-forecasting-ops
```

```bash
cd retail-demand-forecasting-ops
uv venv --python 3.12.10 .venv
source .venv/bin/activate
uv sync
```

## Activate environment
```bash
cd retail-demand-forecasting-ops
source .venv/bin/activate
```

## Run notebook
```bash
jupyter lab retail_demand_forecasting_ops.ipynb
```

## Model-selection policy
- LazyPredict discovery runs on lag-feature regression table after time-aware splitting.
- Only top 3 eligible families move into manual engineering.
- Filters remove unstable/slow/non-operational candidates.

## FLAML optimization workflow summary
- Explicit `time_budget` and budget-aware AutoML search.
- Compares estimator tradeoffs under constrained runtime.
- Evaluates whether FLAML challenger improves forecasting business use.

## PyCaret experiment-lab workflow summary
- Uses setup/compare/tune/finalize/save in regression workflow.
- Uses time-aware split options where feasible.
- Retains finalized model only if business ranking improves.

## Artifacts produced
- `artifacts/leaderboard_retail_forecasting.csv`
- `artifacts/backtest_summary.csv`
- `artifacts/ops_signals.csv`
- optional PyCaret artifact prefix in `artifacts/`

## Deployment and monitoring notes
- Optional Streamlit operations dashboard in `dashboard/app.py`.
- Monitor sMAPE/MAE drift, holiday-period errors, and spike detection precision.
- Retrain cadence can follow monthly schedule plus drift-trigger policy.

## Helper scripts
- `scripts/setup_env.sh`
- `scripts/download_data.sh`
- `scripts/run_notebook.sh`
- `scripts/run_dashboard.sh`
