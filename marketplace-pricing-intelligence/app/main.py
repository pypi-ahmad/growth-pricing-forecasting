from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "pricing_model.joblib"
PREPROCESSOR_PATH = ARTIFACT_DIR / "pricing_preprocessor.joblib"
META_PATH = ARTIFACT_DIR / "pricing_meta.json"

app = FastAPI(title="Marketplace Pricing API", version="0.1.0")


class PriceRequest(BaseModel):
    name: str = Field(...)
    item_description: str = Field(...)
    category_name: str = Field(...)
    brand_name: str = Field("unknown_brand")
    shipping: int = Field(0)
    item_condition_id: int = Field(1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/price")
def price(request: PriceRequest):
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise HTTPException(status_code=404, detail="Model artifacts not found. Train notebook first.")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    payload = pd.DataFrame([
        {
            "name": request.name,
            "item_description": request.item_description,
            "category_name": request.category_name,
            "brand_name": request.brand_name,
            "shipping": request.shipping,
            "item_condition_id": request.item_condition_id,
        }
    ])

    x = preprocessor.transform(payload)
    pred_log = float(model.predict(x)[0])
    pred_price = float(np.expm1(pred_log))

    p50, p90 = 5.0, 12.0
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        p50 = float(meta.get("abs_error_p50", p50))
        p90 = float(meta.get("abs_error_p90", p90))

    return {
        "suggested_price": pred_price,
        "price_interval_p50": [max(0.0, pred_price - p50), pred_price + p50],
        "price_interval_p90": [max(0.0, pred_price - p90), pred_price + p90],
    }
