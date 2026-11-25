# Customer Intelligence Platform

An end-to-end machine learning engineering project that predicts
customer churn using historical customer and transaction behavior.

The project demonstrates the complete ML lifecycle, from data
generation and feature engineering to authenticated serverless
inference on AWS.

## Project Status

Version 1.0 - Implemented, deployed, and validated on AWS.

## Architecture

    Synthetic Customer and Transaction Data
                    |
                    v
    Data Preparation and Feature Engineering
                    |
                    v
    Model Training and Evaluation
                    |
                    v
    Persisted Machine Learning Model
                    |
                    v
    FastAPI Inference Service
                    |
                    v
    Docker Container
                    |
                    v
    Amazon ECR
                    |
                    v
    AWS Lambda
                    |
                    v
    IAM-Authenticated HTTPS API

GitHub provides source code version control.
Amazon CloudWatch provides execution logs and monitoring.

## Problem Statement

Predict whether a customer will make a purchase during the
30 days following a prediction snapshot.

- Snapshot date: July 1, 2026
- Churn = 1: No purchase during the following 30 days
- Churn = 0: At least one purchase during the following 30 days

Only information available on or before the snapshot date
is used as model input to prevent future data leakage.

## Dataset

The project uses synthetic customer and transaction data.

| Dataset | Records |
|---|---|
| Customers | 1000 |
| Orders | 9736 |

## Feature Engineering

Customer-level features include:

- Total orders
- Total spend
- Average order value
- Recency of last purchase
- Customer tenure
- Previous purchase indicator

## Machine Learning

The training pipeline compares Logistic Regression and
Random Forest classifiers.

Logistic Regression was selected as the final model.

The pipeline includes:

- Snapshot-based churn label generation
- Missing-value handling
- Train, validation, and test datasets
- Five-fold cross-validation
- Model comparison
- Classification threshold selection
- Model persistence
- Reusable inference pipeline

## Model Evaluation

| Metric | Result |
|---|---|
| Selected model | Logistic Regression |
| Test ROC-AUC | Approximately 0.556 |
| Test Average Precision | Approximately 0.241 |
| Classification threshold | 0.5092 |

The model demonstrates the technical ML lifecycle but has
limited predictive power. It is not intended for real-world
customer retention decisions without further validation.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | Service information |
| GET | /health | Application and model health |
| GET | /predict/{customer_id} | Customer churn prediction |
| GET | /docs | Interactive API documentation |

## Example Prediction

    {
      "customer_id": 1001,
      "churn_probability": 0.2855,
      "predicted_churn": 0,
      "threshold": 0.5092,
      "model_name": "logistic_regression",
      "snapshot_date": "2026-07-01",
      "horizon_days": 30
    }

## Technology Stack

| Layer | Technology |
|---|---|
| Programming | Python 3.13 |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn |
| Model persistence | Joblib |
| REST API | FastAPI, Uvicorn |
| Testing | Pytest, HTTPX |
| Lambda integration | Mangum |
| Containerization | Docker |
| Container registry | Amazon ECR |
| Cloud compute | AWS Lambda |
| Authentication | AWS IAM, Signature Version 4 |
| Monitoring | Amazon CloudWatch |
| Version control | Git, GitHub |

## Repository Structure

    customer-intelligence-platform/
    |-- src/
    |   |-- data/
    |   |-- api/
    |   |   |-- main.py
    |   |   |-- lambda_handler.py
    |   |-- generate_data.py
    |   |-- data_loader.py
    |   |-- feature.py
    |   |-- train_model.py
    |   |-- predict.py
    |-- tests/
    |   |-- test_api.py
    |-- models/
    |-- Dockerfile
    |-- Dockerfile.lambda
    |-- requirements.txt
    |-- requirements-aws.txt
    |-- test_lambda_url.py
    |-- README.md
    |-- README_API.md
    |-- README_DOCKER.md
    |-- README_ML_UPDATE.md
    |-- README_AWS.md

Generated model artifacts are excluded from version control.

## Local Setup

Create a Python 3.13 virtual environment:

    python -m venv .venv

Install dependencies:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Train the model:

    .\.venv\Scripts\python.exe src/train_model.py

Start the API:

    .\.venv\Scripts\python.exe -m uvicorn api.main:app --app-dir src --host 127.0.0.1 --port 8000

Open http://localhost:8000/docs for interactive documentation.

## Automated Testing

    $env:PYTHONPATH = "$PWD\src"
    .\.venv\Scripts\python.exe -m pytest tests/ -v

Five API regression tests passed during project validation.

## Docker

Build the standard API container:

    docker build -t customer-intelligence-api:1.0 .

Run the container:

    docker run --rm -p 127.0.0.1:8000:8000 customer-intelligence-api:1.0

A separate Lambda-compatible Dockerfile is provided for
AWS deployment.

## AWS Deployment

The application is deployed to AWS Lambda using a private
Amazon ECR container image.

The HTTPS Function URL requires IAM authentication.

Cloud validation confirmed:

- Unauthorized requests return HTTP 403.
- Authenticated health checks return HTTP 200.
- Authenticated prediction requests return HTTP 200.
- CloudWatch records execution logs.

## Additional Documentation

- [Machine Learning Pipeline](README_ML_UPDATE.md)
- [FastAPI Service](README_API.md)
- [Docker Containerization](README_DOCKER.md)
- [AWS Deployment](README_AWS.md)

## Future Improvements

- Real-world customer datasets
- Improved predictive performance
- Automated feature and model pipelines
- Model versioning and monitoring
- Cold-start and inference optimization
- CI/CD and infrastructure as code
- Customer intelligence dashboards
- Generative AI capabilities

## Project Scope

This is a personal educational project demonstrating applied
machine learning, analytics engineering, API development,
containerization, and cloud deployment.

It is a functional prototype, not a production-ready
customer intelligence system.
