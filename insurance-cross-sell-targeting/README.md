# insurance-cross-sell-targeting

## Project goal
Build an insurance cross-sell propensity and campaign targeting system that supports budget-aware targeting decisions.

## Dataset
- Competition: Binary Classification of Insurance Cross Selling
- Link: https://www.kaggle.com/competitions/playground-series-s4e7

Download commands:
```bash
mkdir -p data/raw/insurance_cross_sell
kaggle competitions download -c playground-series-s4e7 -p data/raw/insurance_cross_sell
unzip data/raw/insurance_cross_sell/playground-series-s4e7.zip -d data/raw/insurance_cross_sell
```

## Setup (uv + Python 3.12.10)

```bash
git clone https://github.com/pypi-ahmad/insurance-cross-sell-targeting.git
cd insurance-cross-sell-targeting
```

```bash
cd insurance-cross-sell-targeting
uv venv --python 3.12.10 .venv
source .venv/bin/activate
uv sync
```

## Activate environment
```bash
cd insurance-cross-sell-targeting
source .venv/bin/activate
```

## Run notebook
```bash
jupyter lab insurance_cross_sell_targeting.ipynb
```

## Model-selection policy
- LazyPredict discovery first (after preprocessing and realistic split).
- Only top 3 eligible families move to manual engineering.
- Eligibility prioritizes PR-AUC, ROC-AUC, and precision@k/recall policy quality.

## FLAML optimization workflow summary
- Uses explicit time budget for budget-aware campaign model search.
- Evaluates whether FLAML challenger improves top-k targeting utility.
- Compares operational rank-quality, not metric-only leaderboard wins.

## PyCaret experiment-lab workflow summary
- Uses setup/compare/tune/calibrate/finalize workflow.
- Checks if calibration improves targeting policy stability.
- Keeps finalized artifact only if business targeting improves.

## Artifacts produced
- `artifacts/leaderboard_insurance_cross_sell.csv`
- `artifacts/target_list.csv`
- `artifacts/lift_gain_table.csv`

## Deployment and monitoring notes
- Batch targeting policy outputs `target`, `hold`, `low_priority` segments.
- Monitor PR-AUC drift, precision@k drift, campaign conversion uplift, and policy stability.
- Retrain when ranking quality drops beyond campaign tolerance.

## Helper scripts
- `scripts/setup_env.sh`
- `scripts/download_data.sh`
- `scripts/run_notebook.sh`
