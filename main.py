import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rf_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

# --- Load model + metadata once at startup ---
model = joblib.load(MODEL_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

FEATURE_ORDER = metadata["feature_order"]

app = FastAPI(
    title="VoltCast — Electricity Demand Forecasting API",
    description="Predicts daily electricity demand (Victoria, Australia) from weather and calendar features.",
    version="1.0.0",
)

# Allow the frontend (served from anywhere) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DemandRequest(BaseModel):
    min_temperature: float = Field(..., example=15.0, description="Minimum daily temperature (°C)")
    max_temperature: float = Field(..., example=28.0, description="Maximum daily temperature (°C)")
    solar_exposure: float = Field(..., example=20.0, description="Solar exposure (MJ/m²)")
    rainfall: float = Field(..., example=0.0, description="Rainfall (mm)")
    school_day: int = Field(..., ge=0, le=1, example=1, description="1 if school day, else 0")
    holiday: int = Field(..., ge=0, le=1, example=0, description="1 if public holiday, else 0")
    month: int = Field(..., ge=1, le=12, example=6, description="Month (1-12)")
    day_of_week: int = Field(..., ge=0, le=6, example=2, description="Day of week (0=Mon ... 6=Sun)")
    is_weekend: int = Field(..., ge=0, le=1, example=0, description="1 if Saturday/Sunday, else 0")


class DemandResponse(BaseModel):
    predicted_demand: float
    unit: str = "MW (aggregate daily demand)"


@app.get("/")
def root():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api")
def api_info():
    return {
        "message": "VoltCast Electricity Demand Forecasting API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=DemandResponse)
def predict(payload: DemandRequest):
    try:
        data = payload.dict()
        row = pd.DataFrame([[data[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
        prediction = model.predict(row)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return DemandResponse(predicted_demand=round(float(prediction), 2))
