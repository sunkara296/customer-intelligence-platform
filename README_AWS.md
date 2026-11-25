# AWS Serverless Deployment

## Overview

The Customer Intelligence Platform is deployed as a containerized
machine learning inference service on AWS Lambda using FastAPI
and Mangum.

## Architecture

Python ML Pipeline -> FastAPI -> Docker -> Amazon ECR -> AWS Lambda

AWS IAM secures the HTTPS endpoint.
Amazon CloudWatch provides execution logs.
GitHub provides source code version control.

## AWS Configuration

| Component | Configuration |
|---|---|
| Region | us-east-1 |
| ECR repository | customer-intelligence-api |
| Lambda function | customer-intelligence-api |
| Container image tag | lambda-1.0 |
| Python runtime | 3.13 |
| Architecture | x86_64 |
| Memory | 1024 MB |
| Timeout | 30 seconds |
| Ephemeral storage | 512 MB |
| Function URL authentication | AWS_IAM |
| Invoke mode | Buffered |
| CloudWatch retention | 7 days |
| Provisioned concurrency | Disabled |
| Reserved concurrency | Not configured |
| Monthly budget alert | $5 |

## Build the Lambda Container

Run from the repository root:

    docker buildx build --platform linux/amd64 --provenance=false --sbom=false --load -f Dockerfile.lambda -t customer-intelligence-api:lambda-1.0 .

The build installs dependencies, copies application source code,
trains the model, and packages the trained model inside the image.

## Deployment Process

1. Authenticate AWS CLI using an authorized IAM identity.
2. Create a private ECR repository.
3. Authenticate Docker with ECR.
4. Tag and push the Lambda-compatible image.
5. Create a Lambda function using the ECR image.
6. Configure memory and timeout.
7. Create an IAM-authenticated Function URL.
8. Grant function-scoped invocation permissions.
9. Configure CloudWatch log retention.
10. Validate authenticated and unauthenticated requests.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Service information |
| GET | /health | Application and model health |
| GET | /predict/{customer_id} | Churn prediction |
| GET | /docs | Interactive API documentation |

Function URL requests require AWS Signature Version 4.

## Authenticated Integration Testing

Install the local testing dependency:

    .\.venv\Scripts\python.exe -m pip install -r requirements-aws.txt

Authenticate using AWS CLI:

    aws login --profile customer-intelligence-dev --remote

Configure environment variables:

    $env:AWS_PROFILE = "customer-intelligence-dev"
    $env:AWS_REGION = "us-east-1"
    $env:FUNCTION_URL = "<YOUR_LAMBDA_FUNCTION_URL>"

Run the integration test:

    .\.venv\Scripts\python.exe test_lambda_url.py

The script obtains temporary AWS credentials through AWS CLI
and signs HTTPS requests using Signature Version 4.

The IAM identity requires lambda:InvokeFunctionUrl and
lambda:InvokeFunction permissions for the target function.

## Deployment Validation

| Test | Result |
|---|---|
| Unauthenticated request | HTTP 403 |
| Authenticated health check | HTTP 200 |
| Authenticated prediction | HTTP 200 |
| Customers loaded | 1000 |
| Example customer ID | 1001 |
| Churn probability | 0.2855 |
| Classification threshold | 0.5092 |
| Predicted churn | 0 |

## Observed Performance

| Metric | Observed value |
|---|---|
| Initial health-check duration | 22859.94 ms |
| Subsequent prediction duration | 272.87 ms |
| Prediction billed duration | 273 ms |
| Maximum prediction memory used | 262 MB |
| Configured memory | 1024 MB |

These are individual test measurements, not production benchmarks.

## Security

- Private ECR repository.
- IAM-authenticated HTTPS endpoint.
- Temporary AWS CLI credentials.
- Function-scoped invocation permissions.
- Anonymous requests rejected.
- No hardcoded credentials or deployment URLs.
- Local environment files excluded from Git.

## Cost Management

- On-demand serverless execution.
- No provisioned concurrency.
- No continuously running application server.
- Seven-day CloudWatch log retention.
- Monthly budget alerts configured at $5.

Budget alerts are not a hard spending limit.
ECR storage, Lambda execution, and logging may incur charges.

Reserved concurrency was not configured because of the
current AWS account concurrency quota.

## Known Limitations

- Synthetic training data.
- Limited model predictive performance.
- Significant initial cold-start latency.
- Initialization timeout observed before successful execution.
- Model and customer features loaded during each invocation.
- Dependency versions are not fully pinned.
- Manual deployment without automated CI/CD.
- No production-scale load testing.

This project is an educational end-to-end machine learning
engineering prototype, not a production-ready system.
