from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config(path: str = "configs/config.yaml") -> dict:
    config_path = ROOT / path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
