from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import ChurnPredictor


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Inference API for the end-to-end MLOps platform.",
)

predictor = ChurnPredictor()


class CustomerFeatures(BaseModel):
    tenure: float = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)
    contract_type: str
    payment_method: str
    internet_service: str
    support_calls: int = Field(ge=0)


@app.get("/")
def root():
    return {"message": "Customer Churn MLOps API"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_available": predictor.model_path.exists(),
    }


@app.post("/predict")
def predict(payload: CustomerFeatures):
    try:
        return predictor.predict(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
