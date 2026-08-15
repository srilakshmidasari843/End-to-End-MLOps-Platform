from pathlib import Path
import joblib
import pandas as pd

from src.features.build_features import ALL_FEATURES
from src.utils.config import load_config


class ChurnPredictor:
    def __init__(self):
        config = load_config()
        self.model_path = Path(config["artifacts"]["model_path"])
        self.model = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run: python -m src.models.train"
            )
        self.model = joblib.load(self.model_path)

    def predict(self, payload: dict) -> dict:
        if self.model is None:
            self.load()

        row = pd.DataFrame([{k: payload[k] for k in ALL_FEATURES}])
        prediction = int(self.model.predict(row)[0])
        probability = float(self.model.predict_proba(row)[0, 1])

        return {
            "prediction": prediction,
            "label": "churn" if prediction == 1 else "no_churn",
            "probability": round(probability, 4),
        }

if __name__ == "__main__":
    predictor = ChurnPredictor()

    sample = {
        "tenure": 12,
        "monthly_charges": 70.5,
        "total_charges": 850.0,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "paperless_billing": "Yes",
        "support_calls": 3,
    }

    print(predictor.predict(sample))