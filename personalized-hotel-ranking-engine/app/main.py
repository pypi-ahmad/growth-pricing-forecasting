from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "ranking_model.joblib"
PREPROCESSOR_PATH = ARTIFACT_DIR / "ranking_preprocessor.joblib"

app = FastAPI(title="Hotel Ranking API", version="0.1.0")


class RankRequest(BaseModel):
    candidates: List[Dict[str, Any]] = Field(..., description="Candidate feature rows for one or more search ids")
    top_k: int = 5


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend_top5")
def recommend_top5(request: RankRequest):
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise HTTPException(status_code=404, detail="Model artifacts not found. Train notebook first.")

    if not request.candidates:
        raise HTTPException(status_code=400, detail="No candidates provided")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    df = pd.DataFrame(request.candidates)
    if "srch_id" not in df.columns or "hotel_cluster" not in df.columns:
        raise HTTPException(status_code=400, detail="candidates must include srch_id and hotel_cluster")

    feature_df = df.drop(columns=[c for c in ["label", "date_time"] if c in df.columns], errors="ignore")
    x = preprocessor.transform(feature_df.drop(columns=["srch_id"], errors="ignore"))
    scores = model.predict_proba(x)[:, 1]

    out = df[["srch_id", "hotel_cluster"]].copy()
    out["score"] = scores

    recs = []
    for srch_id, g in out.groupby("srch_id"):
        top = g.sort_values("score", ascending=False).head(request.top_k)
        recs.append(
            {
                "srch_id": int(srch_id),
                "top_k": top["hotel_cluster"].astype(int).tolist(),
                "scores": [float(x) for x in top["score"].tolist()],
            }
        )

    return {"recommendations": recs}
