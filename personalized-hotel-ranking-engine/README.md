# personalized-hotel-ranking-engine

## Project goal
Build an end-to-end personalized hotel recommendation system that scores candidates and returns top-5 ranked hotel clusters for each search context.

## Dataset
- Competition: Expedia Hotel Recommendations
- Link: https://www.kaggle.com/competitions/expedia-hotel-recommendations

Download commands:
```bash
mkdir -p data/raw/expedia
kaggle competitions download -c expedia-hotel-recommendations -p data/raw/expedia
unzip data/raw/expedia/expedia-hotel-recommendations.zip -d data/raw/expedia
```

## Setup (uv + Python 3.12.10)

```bash
git clone https://github.com/pypi-ahmad/personalized-hotel-ranking-engine.git
cd personalized-hotel-ranking-engine
```

```bash
cd personalized-hotel-ranking-engine
uv venv --python 3.12.10 .venv
source .venv/bin/activate
uv sync
```

## Activate environment
```bash
cd personalized-hotel-ranking-engine
source .venv/bin/activate
```

## Run notebook
```bash
jupyter lab personalized_hotel_ranking_engine.ipynb
```

## Model-selection policy
- Ranking is reformulated into supervised candidate scoring.
- LazyPredict runs first on engineered candidate table.
- Only top 3 eligible families from LazyPredict move into manual engineering.
- Eligibility uses MAP@5, HitRate@5, and latency.

## FLAML optimization workflow summary
- Uses explicit `time_budget` for candidate-scoring optimization.
- Validates whether FLAML improves ranking quality, not just classification objective.
- Reviews best estimator, best config, and production tradeoffs.

## PyCaret experiment-lab workflow summary
- Uses setup/compare/tune/finalize workflow for candidate scoring.
- Final model evaluated with ranking metrics, not classification score alone.
- Finalized artifact retained only if top-k business utility improves.

## Artifacts produced
- `artifacts/leaderboard_expedia_ranking.csv`
- `artifacts/top5_sample_recommendations.csv`
- `artifacts/ranking_model.joblib`
- `artifacts/ranking_preprocessor.joblib`

## Deployment and monitoring notes
- FastAPI endpoint in `app/main.py` serves top-5 recommendations from candidate payload.
- Monitor MAP@5, HitRate@5, candidate-coverage drift, and cold-start segment performance.
- Retrain when ranking quality drops materially on recent booking windows.

## Helper scripts
- `scripts/setup_env.sh`
- `scripts/download_data.sh`
- `scripts/run_notebook.sh`
- `scripts/run_api.sh`
