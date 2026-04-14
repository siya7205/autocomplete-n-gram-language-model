from pathlib import Path

import pandas as pd
import streamlit as st

from autocomplete.predict import (
    DEFAULT_DATA_PATH,
    rerank_with_sentiment_model,
    train_model,
)
from autocomplete.preprocess import tokenize
from autocomplete.sentiment import load_sentiment_model
from language_model import get_suggestions


@st.cache_resource(show_spinner=False)
def load_language_model(data_path: str):
    return train_model(data_path=Path(data_path))


@st.cache_resource(show_spinner=False)
def load_sentiment_model_cached(model_path: str):
    return load_sentiment_model(model_path)


def get_base_suggestions(prefix_text: str, top_k: int, data_path: str):
    vocabulary, n_gram_counts_list = load_language_model(data_path)
    tokens = tokenize(prefix_text)
    suggestions = get_suggestions(tokens, n_gram_counts_list, vocabulary, k=1.0)
    sorted_suggestions = sorted(suggestions, key=lambda row: row[1], reverse=True)
    return sorted_suggestions[:top_k]


def main() -> None:
    st.set_page_config(page_title="Sentiment-aware Autocomplete Demo", layout="centered")
    st.title("Sentiment-aware Autocomplete Demo")
    st.caption("Phase 5 optional Streamlit demo UI")

    prefix_text = st.text_input("Prefix text", value="I want to")
    top_k = st.slider("Top-k suggestions", min_value=1, max_value=20, value=5)
    sentiment = st.selectbox("Target sentiment", options=["off", "positive", "negative", "neutral"], index=0)
    sentiment_weight = st.slider(
        "Sentiment weight",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        disabled=sentiment == "off",
    )
    data_path = str(DEFAULT_DATA_PATH)
    sentiment_model_path = "models/sentiment.pkl"

    if not prefix_text.strip():
        st.info("Type a prefix to view suggestions.")
        return

    try:
        with st.spinner("Loading language model and generating suggestions..."):
            base_suggestions = get_base_suggestions(prefix_text=prefix_text, top_k=top_k, data_path=data_path)
    except FileNotFoundError as exc:
        st.error(f"Language model corpus file not found: {exc}")
        return
    except Exception as exc:  # pragma: no cover - UI safety
        st.error(f"Could not generate suggestions: {exc}")
        return

    if sentiment == "off":
        rows = [
            {
                "suggestion": word,
                "lm_score": float(lm_score),
                "sentiment_score": "N/A",
                "final_score": "N/A",
            }
            for word, lm_score in base_suggestions
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return

    model_path = Path(sentiment_model_path)
    if not model_path.exists():
        st.warning(
            "Sentiment model not found. Train one first with: "
            f'python -m autocomplete.train_sentiment --csv <labeled_csv> --out "{sentiment_model_path}"'
        )
        rows = [
            {
                "suggestion": word,
                "lm_score": float(lm_score),
                "sentiment_score": "N/A",
                "final_score": "N/A",
            }
            for word, lm_score in base_suggestions
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return

    try:
        sentiment_model = load_sentiment_model_cached(str(model_path))
        reranked, used_neutral_fallback, model_labels = rerank_with_sentiment_model(
            prefix_text=prefix_text,
            suggestions=base_suggestions,
            target_sentiment=sentiment,
            model=sentiment_model,
            sentiment_weight=sentiment_weight,
        )
    except Exception as exc:  # pragma: no cover - UI safety
        st.error(f"Could not rerank with sentiment: {exc}")
        return

    if used_neutral_fallback:
        st.info(
            f'neutral is unavailable in this sentiment model (labels: {model_labels}); reranking is disabled '
            "for this request."
        )

    rows = [
        {
            "suggestion": row["word"],
            "lm_score": row["lm_score"],
            "sentiment_score": row["sentiment_score"],
            "final_score": row["final_score"],
        }
        for row in reranked
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
