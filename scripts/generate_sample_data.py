from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)
    n = 2500

    tenure = rng.integers(0, 73, n)
    monthly = np.clip(rng.normal(70, 25, n), 18, 130)
    total = np.maximum(0, tenure * monthly + rng.normal(0, 180, n))
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n,
        p=[0.56, 0.24, 0.20],
    )
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card",
        ],
        n,
    )
    internet = rng.choice(
        ["DSL", "Fiber optic", "No"],
        n,
        p=[0.36, 0.48, 0.16],
    )
    support_calls = rng.poisson(1.4, n)

    logit = (
        -1.2
        - 0.035 * tenure
        + 0.015 * (monthly - 65)
        + 0.45 * (contract == "Month-to-month")
        + 0.55 * (internet == "Fiber optic")
        + 0.30 * (payment == "Electronic check")
        + 0.18 * support_calls
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly.round(2),
            "total_charges": total.round(2),
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": internet,
            "support_calls": support_calls,
            "churn": churn,
        }
    )

    output = Path("data/raw/customer_churn.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Created {output} with {len(df)} rows.")
    print(df.head())


if __name__ == "__main__":
    main()
