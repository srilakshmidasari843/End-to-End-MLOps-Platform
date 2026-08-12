from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.ingest import load_data
from src.data.validate import validate_data
from src.features.build_features import ALL_FEATURES, build_preprocessor
from src.utils.config import load_config


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def get_models(random_state: int):
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=random_state,
            class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        ),
    }


def main():
    config = load_config()
    data_path = Path(config["data"]["raw_path"])

    df = load_data(data_path)
    validate_data(df)

    target = config["training"]["target"]
    random_state = config["training"]["random_state"]
    test_size = config["training"]["test_size"]

    X = df[ALL_FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    best_name = None
    best_score = float("-inf")
    best_pipeline = None
    leaderboard = []

    for model_name, model in get_models(random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )

        with mlflow.start_run(run_name=model_name):
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            metrics = evaluate(y_test, predictions, probabilities)

            mlflow.log_param("model_name", model_name)
            mlflow.log_metrics(metrics)
            # mlflow.sklearn.log_model(pipeline, artifact_path="model")

            leaderboard.append({"model": model_name, **metrics})

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_name = model_name
                best_pipeline = pipeline

    artifact_path = Path(config["artifacts"]["model_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, artifact_path)

    leaderboard_df = pd.DataFrame(leaderboard).sort_values(
        "roc_auc", ascending=False
    )
    leaderboard_path = artifact_path.parent / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)

    print("\nModel leaderboard:")
    print(leaderboard_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Best ROC-AUC: {best_score:.4f}")
    print(f"Saved model to: {artifact_path}")


if __name__ == "__main__":
    main()
