# Customer Intelligence Platform

An end-to-end machine learning project for predicting customer churn from historical customer and transaction behavior.

The project is designed to demonstrate the complete ML lifecycle: data preparation, feature engineering, supervised learning, model evaluation, API-based inference, and eventual cloud deployment.

## Problem Statement

Given customer behavior available at a specific point in time, predict whether a customer will make a purchase during the following 30 days.

For the current implementation:

- Snapshot date: July 1, 2026
- `churn = 1`: customer makes no purchase during the next 30 days
- `churn = 0`: customer makes at least one purchase during the next 30 days

Only information available on or before the snapshot date is used as model input to prevent data leakage.

## Current Features

The customer-level modeling dataset currently includes:

- Total orders
- Total spend
- Average order value
- Recency of last purchase
- Customer tenure
- Whether the customer has previously placed an order

Features are constructed from historical transactions as of the prediction snapshot.

## Machine Learning

The current baseline model is Logistic Regression.

The dataset is divided into:

- 70% training
- 15% validation
- 15% testing

The validation dataset is used for model and classification-threshold decisions, while the test dataset is kept separate for final evaluation.

Because churn is an imbalanced classification problem, evaluation focuses on metrics including:

- Precision
- Recall
- F1 score
- Confusion matrix

## Current Status

Completed:

- Synthetic customer and transaction data generation
- Snapshot-based feature engineering
- Future 30-day churn label generation
- Data leakage prevention
- Missing-value handling
- Train / validation / test splitting
- Logistic Regression baseline
- Classification threshold analysis
- Initial model evaluation

In Progress:

- ROC-AUC and PR-AUC evaluation
- Model improvement and comparison
- Feature analysis

Planned:

- Additional ML models
- Reusable training pipeline
- Model persistence
- FastAPI inference service
- Docker containerization
- AWS deployment
- GenAI / customer intelligence capabilities

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Git / GitHub

## Project Goal

The goal is to evolve this repository from an analytical ML experiment into a production-style customer intelligence service that can generate customer-level predictions and expose them through an API.
