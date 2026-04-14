import pickle
from pathlib import Path

import nltk


def ensure_nltk_data():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def save_model(model, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load_model(model_path: str):
    with Path(model_path).open("rb") as f:
        return pickle.load(f)

