# marketplace-pricing-intelligence

## Project goal
Build a marketplace pricing intelligence system using text + structured features, with seller-facing price suggestions and deployable inference.

## Dataset
- Competition: Mercari Price Suggestion Challenge
- Link: https://www.kaggle.com/competitions/mercari-price-suggestion-challenge

Download commands:
```bash
mkdir -p data/raw/mercari
kaggle competitions download -c mercari-price-suggestion-challenge -p data/raw/mercari
unzip data/raw/mercari/mercari-price-suggestion-challenge.zip -d data/raw/mercari
```

## Setup (uv + Python 3.12.10)

```bash
git clone https://github.com/pypi-ahmad/marketplace-pricing-intelligence.git
cd marketplace-pricing-intelligence
```

```bash
cd marketplace-pricing-intelligence
uv venv --python 3.12.10 .venv
source .venv/bin/activate
uv sync
```

## Activate environment
```bash
cd marketplace-pricing-intelligence
source .venv/bin/activate
```

## Run notebook
```bash
jupyter lab marketplace_pricing_intelligence.ipynb
```

## Model-selection policy
- LazyPredict discovery runs first on a manageable reduced representation.
- Only top 3 eligible families from LazyPredict move into manual engineering.
- Eligibility prioritizes RMSLE, MAE, and operational latency/model practicality.

## FLAML optimization workflow summary
- Uses explicit `time_budget` and budget-aware search over candidate regressors.
- Evaluates discovered model under fixed budget and compares speed/accuracy tradeoff.
- Decides production-worthiness by business usefulness, not metric alone.

## PyCaret experiment-lab workflow summary
- Uses setup/compare/tune/finalize/save workflow.
- Evaluates whether PyCaret model improves calibration-like pricing behavior and deployability.
- Keeps finalized artifact only if leaderboard and business checks support it.

## Artifacts produced
- `artifacts/leaderboard_marketplace_pricing.csv`
- `artifacts/category_error_analysis.csv`
- `artifacts/rerun_seed_stability.csv`
- `artifacts/pricing_model.joblib`
- `artifacts/pricing_preprocessor.joblib`
- `artifacts/pricing_meta.json`

## Deployment and monitoring notes
- FastAPI endpoint in `app/main.py` provides seller-facing suggested price and interval.
- Monitor category mix drift, price distribution drift, RMSLE drift, and residual spread drift.
- Retrain on schedule or when category drift exceeds threshold.

## Helper scripts
- `scripts/setup_env.sh`
- `scripts/download_data.sh`
- `scripts/run_notebook.sh`
- `scripts/run_api.sh`
