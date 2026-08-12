import pandas as pd
import pytest

from src.data.validate import validate_data


def valid_df():
    return pd.DataFrame(
        {
            "tenure": [1, 10],
            "monthly_charges": [50.0, 70.0],
            "total_charges": [50.0, 700.0],
            "contract_type": ["Month-to-month", "One year"],
            "payment_method": ["Electronic check", "Credit card"],
            "internet_service": ["Fiber optic", "DSL"],
            "support_calls": [1, 0],
            "churn": [1, 0],
        }
    )


def test_valid_data_passes():
    validate_data(valid_df())


def test_missing_column_fails():
    df = valid_df().drop(columns=["tenure"])
    with pytest.raises(ValueError):
        validate_data(df)


def test_negative_value_fails():
    df = valid_df()
    df.loc[0, "tenure"] = -1
    with pytest.raises(ValueError):
        validate_data(df)
