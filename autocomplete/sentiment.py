from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from autocomplete.preprocess import normalize_text, tokenize


REQUIRED_COLUMNS = {"id", "text", "sentiment_label", "notes"}


def _load_labeled_rows(csv_path: Path) -> Tuple[list[str], list[str]]:
    dataframe = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    texts = dataframe["text"].fillna("").astype(str)
    labels = dataframe["sentiment_label"].fillna("").astype(str).str.strip().str.lower()

    valid_mask = (labels != "") & (texts.str.strip() != "")
    filtered_texts = texts[valid_mask].map(normalize_text).tolist()
    filtered_labels = labels[valid_mask].tolist()

    if not filtered_texts:
        raise ValueError("No labeled rows found. Fill sentiment_label for at least one row.")

    return filtered_texts, filtered_labels


def train_sentiment_model(csv_path: str, model_out_path: str, seed: int = 42) -> Dict[str, Any]:
    texts, labels = _load_labeled_rows(Path(csv_path))

    if len(set(labels)) < 2:
        raise ValueError("Need at least 2 sentiment classes to train a classifier.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=seed,
            stratify=None,
        )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenize,
                    token_pattern=None,
                    lowercase=False,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    classes = [str(label) for label in model.named_steps["classifier"].classes_]
    matrix = confusion_matrix(y_test, y_pred, labels=classes)

    output_path = Path(model_out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    return {
        "model_path": str(output_path),
        "labeled_rows": len(texts),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "labels": classes,
        "confusion_matrix": matrix.tolist(),
    }


def load_sentiment_model(model_path: str) -> Pipeline:
    return joblib.load(model_path)


def predict_sentiment(
    text: str,
    model: Optional[Pipeline] = None,
    model_path: str = "models/sentiment.pkl",
) -> Tuple[str, Dict[str, float]]:
    loaded_model = model if model is not None else load_sentiment_model(model_path)
    normalized_text = normalize_text(text)

    prediction = str(loaded_model.predict([normalized_text])[0])
    scores: Dict[str, float] = {}

    classifier = loaded_model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        probabilities = loaded_model.predict_proba([normalized_text])[0]
        classes = [str(label) for label in classifier.classes_]
        scores = {label: float(prob) for label, prob in zip(classes, probabilities)}

    return prediction, scores
