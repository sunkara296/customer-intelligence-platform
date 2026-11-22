# Docker Deployment

## Overview

This project packages the Customer Intelligence Platform
as a Docker image running Python 3.13 and FastAPI.

The Docker build installs project dependencies and trains
the churn prediction model from synthetic CSV datasets.

## Prerequisites

- Docker Desktop with a running Docker engine

## Build

From the project root:

    docker build -t customer-intelligence-api:1.0 .

## Run

    docker run --rm -d --name customer-intelligence-api -p 127.0.0.1:8000:8000 customer-intelligence-api:1.0

## Verify

    docker ps

    Invoke-RestMethod http://127.0.0.1:8000/health

    Invoke-RestMethod http://127.0.0.1:8000/predict/1001

Swagger UI:

    http://127.0.0.1:8000/docs

## Logs

    docker logs customer-intelligence-api

## Stop

    docker stop customer-intelligence-api

## Notes

- The trained model is generated during image build.
- Synthetic data is included for demonstration.
- Model performance is not sufficient for production decisions.
- The service has no authentication or authorization.
- The recommended run command binds to localhost only.
