import pandas as pd

REQUIRED_COLUMNS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "payment_method",
    "internet_service",
    "support_calls",
    "churn",
]

def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def validate_values(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")

    for col in ["tenure", "monthly_charges", "total_charges", "support_calls"]:
        if (df[col].dropna() < 0).any():
            raise ValueError(f"Negative values detected in {col}.")

    invalid_target = set(df["churn"].dropna().unique()) - {0, 1}
    if invalid_target:
        raise ValueError(f"Invalid churn labels: {invalid_target}")

def validate_data(df: pd.DataFrame) -> None:
    validate_schema(df)
    validate_values(df)
