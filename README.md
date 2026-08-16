# End-to-End Customer Churn MLOps Platform

An end-to-end machine learning platform for training, tracking, testing, and serving a customer churn prediction model.

This project demonstrates how a machine learning model can move from experimentation into a reproducible inference service with experiment tracking, automated testing, containerization, and continuous integration.

# Project Overview

The system trains machine learning models to predict customer churn using customer account and service information.

The project includes:

- Data preprocessing and feature engineering
- Logistic Regression and Random Forest model training
- MLflow experiment tracking
- Model artifact persistence
- FastAPI inference service
- Request validation with Pydantic
- Automated API testing with pytest
- Docker containerization
- GitHub Actions CI
- Automated Docker build validation

# Architecture

Customer Churn Dataset
        |
        v
Data Processing
        |
        v
Feature Engineering
        |
        v
Model Training
        |
        v
MLflow Experiment Tracking
        |
        v
Best Model Artifact
        |
        v
FastAPI Prediction Service
        |
        v
Docker Container
        |
        v
GitHub Actions CI

# Technology Stack

Language: Python

Machine Learning: scikit-learn

Data Processing: pandas

Experiment Tracking: MLflow

API: FastAPI

Validation: Pydantic

Testing: pytest

Model Serialization: joblib

Containerization: Docker

Continuous Integration: GitHub Actions

# Project Structure

end-to-end-mlops-platform/

api/
    main.py

configs/
    config.yaml

data/
    raw/

src/
    features/
    models/
        train.py
        predict.py
    utils/

tests/
    test_api.py

artifacts/
    best_model.joblib

.github/
    workflows/
        ci.yml

Dockerfile

.dockerignore

requirements.txt

README.md

# Model Training

The training pipeline reads configuration values from configs/config.yaml and prepares the customer churn dataset for model development.

The project evaluates multiple classification approaches including:

- Logistic Regression
- Random Forest

The trained model artifact is stored so the inference API can load the model independently from the training process.

# MLflow Experiment Tracking

MLflow is used to track machine learning experiments and compare training runs.

To start MLflow:

mlflow ui

Open in the browser:

http://127.0.0.1:5000

The MLflow interface allows model experiments and different training runs to be inspected and compared.

# Prediction API

The trained machine learning model is exposed through a FastAPI application.

Available endpoints:

GET /

Returns basic information about the API.

GET /health

Checks whether the API is healthy and whether the model artifact is available.

POST /predict

Accepts customer information and generates a churn prediction.

Example prediction request:

{
  "tenure": 12,
  "monthly_charges": 70.0,
  "total_charges": 840.0,
  "contract_type": "Month-to-month",
  "payment_method": "Electronic check",
  "internet_service": "Fiber optic",
  "support_calls": 2
}

The endpoint returns the churn prediction and prediction probability.

# Running the API Locally

Install dependencies:

pip install -r requirements.txt

Start the FastAPI application:

uvicorn api.main:app --reload

Open the API documentation:

http://127.0.0.1:8000/docs

FastAPI provides an interactive Swagger interface for testing the endpoints.

# Automated Testing

The API is tested using pytest and FastAPI's test client.

Run the tests with:

python -m pytest -q

The tests verify:

- Health endpoint availability
- Successful prediction requests
- Expected HTTP status codes
- Prediction response structure

# Docker

The prediction API is containerized using Docker.

Build the Docker image:

docker build -t churn-mlops-api .

Run the container:

docker run -p 8000:8000 churn-mlops-api

Open:

http://localhost:8000/docs

The application and its dependencies can therefore run inside a reproducible container environment.

# Continuous Integration

GitHub Actions is used for continuous integration.

The CI workflow automatically:

1. Sets up Python
2. Installs project dependencies
3. Runs the pytest test suite
4. Builds the Docker image

This helps ensure that new changes do not break the application, tests, or Docker build.

# Key Engineering Features

- End-to-end machine learning pipeline
- Configuration-driven model training
- MLflow experiment tracking
- Persisted model artifacts
- Reusable prediction logic
- REST API using FastAPI
- Pydantic request validation
- Automated pytest testing
- Docker containerization
- GitHub Actions continuous integration
- Automated Docker build validation

# Future Improvements

Potential future improvements include:

- Cloud deployment
- Model registry
- Automated model retraining
- Data drift monitoring
- Prediction drift monitoring
- Production observability
- Managed model artifact storage

# Author

Srilakshmi Dasari

End-to-end MLOps engineering project demonstrating model development, experiment tracking, API serving, automated testing, Docker containerization, and continuous integration.