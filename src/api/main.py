from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from data_loader import load_data
from feature import build_customer_features

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Train the model first.")

    artifact = joblib.load(MODEL_PATH)

    customers = load_data(ROOT / "src/data/customers.csv")
    orders = load_data(ROOT / "src/data/orders.csv")

    features = build_customer_features(
        customers.copy(),
        orders.copy(),
        artifact["snapshot_date"]
    )

    state["artifact"] = artifact
    state["features"] = features.set_index("customer_id")

    yield

    state.clear()


app = FastAPI(
    title="Customer Intelligence Platform",
    description="Customer churn prediction prototype",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "service": "Customer Intelligence Platform",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": "artifact" in state,
        "model_name": state["artifact"]["model_name"],
        "customers_loaded": len(state["features"])
    }


@app.get("/predict/{customer_id}")
def predict(customer_id: int):
    artifact = state["artifact"]
    features = state["features"]

    if customer_id not in features.index:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found"
        )

    customer = features.loc[[customer_id]]

    probability = float(
        artifact["model"].predict_proba(
            customer[artifact["feature_columns"]]
        )[0, 1]
    )

    threshold = float(artifact["threshold"])

    return {
        "customer_id": customer_id,
        "churn_probability": round(probability, 4),
        "predicted_churn": int(probability >= threshold),
        "threshold": round(threshold, 4),
        "model_name": artifact["model_name"],
        "snapshot_date": artifact["snapshot_date"],
        "horizon_days": artifact["horizon_days"]
    }
