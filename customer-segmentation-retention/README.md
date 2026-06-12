# Customer Segmentation and Retention Analytics

## Project Goal
Build an interview-ready, business-aligned retention analytics workflow where customer segmentation and churn modeling are connected to intervention decisions under budget constraints.

The notebook implements four serious modeling labs in one end-to-end flow:
1. LazyPredict Discovery Lab
2. Manual Engineering Lab
3. FLAML Optimization Lab
4. PyCaret Experiment Lab

## Setup and Dataset

```bash
git clone https://github.com/pypi-ahmad/customer-segmentation-retention.git
cd customer-segmentation-retention
```

### Dataset
- Primary dataset: Online Retail (Kaggle)
  - https://www.kaggle.com/datasets/vijayuv/onlineretail
- Optional reference dataset: Telco Churn (Kaggle)
  - https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### Download commands
```bash
uv run kaggle datasets download -d vijayuv/onlineretail -p data/raw/retail --unzip
uv run kaggle datasets download -d blastchar/telco-customer-churn -p data/raw/telco --unzip
```

### Environment and run
```bash
cd customer-segmentation-retention
uv sync
source .venv/bin/activate
jupyter notebook customer_segmentation_retention.ipynb
```

## Current Workflow
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

## LazyPredict -> Top 3 Manual Model Rule
- LazyPredict is used for model-family discovery after feature matrix and validation split are fixed.
- Candidate families are filtered for manual feasibility and minimum quality.
- Only the top 3 eligible families are promoted to manual implementation.
- Manual models are not chosen arbitrarily.

## FLAML Optimization Workflow
- FLAML runs as a dedicated lab with explicit `time_budget`.
- Uses the project’s primary metric focus (average precision / PR-oriented objective).
- Reports searched estimator list, best estimator, best config, and budget context.
- Evaluated with the same validation discipline and business-threshold logic used elsewhere.

## PyCaret Experiment Workflow
- Uses `ClassificationExperiment` orchestration (`fit` setup-equivalent, `compare_models`, `tune_model`, `calibrate_model`, `finalize_model`, `save_model`).
- Captures compare leaderboard, tuning outcomes, calibration output, finalized artifact path.
- Produces a deployable PyCaret model candidate with holdout evaluation.

## Final Leaderboard Logic
Unified leaderboard combines:
- Top LazyPredict results
- Manual top-3 implementations
- Best FLAML result
- Finalized PyCaret result
- Baseline reference

Ranking emphasizes:
- primary: PR-AUC
- secondary: recall at business threshold
- tertiary: precision at business threshold
- calibration: Brier score (lower better)

Output files in `artifacts/`:
- `leaderboard_segmentation.csv`
- `leaderboard_churn.csv`
- `leaderboard_unified.csv`

## Deployment and Monitoring Path
- Deployment contract is exported as `artifacts/deployment_spec.json`.
- Recommended production flow:
  1. Scheduled feature generation
  2. Batch/online churn scoring
  3. Action policy mapping (`retention_offer`, `premium_upsell`, `monitor`, `low_priority`)
  4. CRM activation and KPI tracking
- Monitoring includes drift, calibration health, segment mix shift, and business KPI triggers with retraining actions.

## Code Structure
- `src/features.py`: cleaning, leakage checks, customer feature engineering, LTV proxy, data dictionary helper
- `src/modeling.py`: segmentation utilities, all four churn labs, threshold economics, calibration/error analysis, unified ranking, seed stability
- `customer_segmentation_retention.ipynb`: main end-to-end project notebook
