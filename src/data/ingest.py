from pathlib import Path
import pandas as pd

def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run: python scripts/generate_sample_data.py"
        )
    return pd.read_csv(path)
