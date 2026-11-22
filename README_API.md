# Customer Intelligence Platform - API

## Overview

FastAPI service for serving customer churn predictions
using a trained scikit-learn model and synthetic customer data.

## Install dependencies

From the project root:

    python -m pip install -r requirements.txt

## Train the model

    python src/train_model.py

## Start the API

    python -m uvicorn api.main:app --app-dir src --host 127.0.0.1 --port 8000

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Service information |
| GET | /health | Model and service health |
| GET | /predict/{customer_id} | Customer churn prediction |
| GET | /docs | Interactive Swagger documentation |

## Example prediction

GET /predict/1001

    {
        "customer_id": 1001,
        "churn_probability": 0.2855,
        "predicted_churn": 0,
        "threshold": 0.5092,
        "model_name": "logistic_regression",
        "snapshot_date": "2026-07-01",
        "horizon_days": 30
    }

## Run automated tests

Windows PowerShell:

    $env:PYTHONPATH = "$PWD\src"
    python -m pytest tests/test_api.py -v

Train the model before running tests.

## Limitations

- Uses synthetic customer and order data.
- Predictions are limited to existing customer IDs.
- Model scores are not calibrated real-world churn probabilities.
- Model performance is insufficient for production decisions.
- Authentication and authorization are not yet implemented.
- Intended for learning and demonstration purposes.
