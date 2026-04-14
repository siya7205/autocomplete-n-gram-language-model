import argparse
from pathlib import Path

from autocomplete.datasets import load_train_test_split
from autocomplete.preprocess import tokenize
from autocomplete.sentiment import load_sentiment_model, predict_sentiment
from data_preprocessing import preprocess_data
from language_model import count_n_grams, get_suggestions


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "en_US.twitter.txt"
RANK_WIDTH = 6
WORD_WIDTH = 18
LM_SCORE_WIDTH = 12
SENTIMENT_WIDTH = 14
FINAL_WIDTH = 14


def train_model(data_path: Path, train_fraction: float = 0.8, minimum_freq: int = 2):
    """Train N-gram count tables from a text dataset.

    Args:
        data_path: Path to newline-delimited text data.
        train_fraction: Fraction of tokenized sentences used for training.
        minimum_freq: Minimum token count kept in closed vocabulary.

    Returns:
        A tuple of (vocabulary, n_gram_counts_list).
    """
    train_data, test_data = load_train_test_split(
        data_path=data_path,
        train_fraction=train_fraction,
    )

    train_data_processed, _, vocabulary = preprocess_data(
        train_data, test_data, minimum_freq
    )

    n_gram_counts_list = []
    for n in range(1, 5):
        n_gram_counts_list.append(count_n_grams(train_data_processed, n))

    return vocabulary, n_gram_counts_list


def predict_next_words(text: str, top_k: int, data_path: Path, k_smoothing: float = 1.0):
    """Predict next-word suggestions for input text.

    Note:
        This Phase 0 baseline retrains in-memory from `data_path` on each call
        to keep usage simple and behavior aligned with existing scripts.

    Args:
        text: Input prefix text.
        top_k: Number of suggestions to return.
        data_path: Dataset path used to train before prediction.
        k_smoothing: Add-k smoothing value.

    Returns:
        List of (word, probability) tuples sorted by descending probability.
    """
    vocabulary, n_gram_counts_list = train_model(data_path=data_path)

    tokens = tokenize(text)
    suggestions = get_suggestions(tokens, n_gram_counts_list, vocabulary, k_smoothing)
    sorted_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
    return sorted_suggestions[:top_k]


def rerank_with_sentiment(
    prefix_text: str,
    suggestions,
    target_sentiment: str,
    sentiment_model_path: str,
    sentiment_weight: float,
):
    model = load_sentiment_model(sentiment_model_path)
    return rerank_with_sentiment_model(
        prefix_text=prefix_text,
        suggestions=suggestions,
        target_sentiment=target_sentiment,
        model=model,
        sentiment_weight=sentiment_weight,
    )


def rerank_with_sentiment_model(
    prefix_text: str,
    suggestions,
    target_sentiment: str,
    model,
    sentiment_weight: float,
):
    classifier = model.named_steps.get("classifier")
    if classifier is None:
        raise ValueError(
            "Sentiment model is missing the expected 'classifier' pipeline step. "
            "Re-train with: python -m autocomplete.train_sentiment --csv <labeled_csv> --out <model_path>."
        )
    model_labels = {str(label).lower() for label in getattr(classifier, "classes_", [])}

    neutral_fallback = False
    if target_sentiment not in model_labels:
        if target_sentiment == "neutral":
            neutral_fallback = True
        else:
            raise ValueError(
                f'Requested sentiment "{target_sentiment}" is not available in the trained model labels: '
                f"{sorted(model_labels)}. Train with that label or use --sentiment off."
            )

    scored = []
    for word, lm_score in suggestions:
        candidate_text = f"{prefix_text} {word}".strip()
        _, sentiment_scores = predict_sentiment(text=candidate_text, model=model)
        if neutral_fallback:
            target_score = 0.0
            final_score = float(lm_score)
        elif sentiment_scores:
            target_score = float(sentiment_scores.get(target_sentiment, 0.0))
            final_score = float(lm_score) + sentiment_weight * target_score
        else:
            target_score = 0.0
            final_score = float(lm_score)

        scored.append(
            {
                "word": word,
                "lm_score": float(lm_score),
                "sentiment_score": float(target_score),
                "final_score": float(final_score),
            }
        )

    ranked = sorted(scored, key=lambda row: row["final_score"], reverse=True)
    return ranked, neutral_fallback, sorted(model_labels)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict top-k next-word suggestions using the N-gram model.")
    parser.add_argument("--text", required=True, help="Input text prefix to autocomplete.")
    parser.add_argument(
        "--top-k",
        "--topk",
        dest="top_k",
        type=int,
        default=5,
        help="Number of suggestions to print.",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Path to dataset file used for training before prediction.",
    )
    parser.add_argument(
        "--sentiment",
        choices=["off", "positive", "negative", "neutral"],
        default="off",
        help="Target sentiment for reranking suggestions. Use 'off' to keep baseline ranking.",
    )
    parser.add_argument(
        "--sentiment-model",
        default="models/sentiment.pkl",
        help="Path to trained sentiment model artifact used for reranking.",
    )
    parser.add_argument(
        "--sentiment-weight",
        "--lambda",
        dest="sentiment_weight",
        type=float,
        default=1.0,
        help="Weight for sentiment score in reranking: final = lm_score + weight * sentiment_score.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    suggestions = predict_next_words(
        text=args.text,
        top_k=args.top_k,
        data_path=Path(args.data),
    )

    if args.sentiment == "off":
        print(f'Input: "{args.text}"')
        print(f"Top {args.top_k} suggestions:")
        for rank, (word, probability) in enumerate(suggestions, start=1):
            print(f"{rank}. {word}\t{probability:.6f}")
        return

    model_path = Path(args.sentiment_model)
    if not model_path.exists():
        raise FileNotFoundError(
            f'Sentiment model not found at "{model_path}". '
            "Train it first with: python -m autocomplete.train_sentiment --csv <labeled_csv> --out "
            f'"{model_path}"'
        )

    reranked, used_neutral_fallback, model_labels = rerank_with_sentiment(
        prefix_text=args.text,
        suggestions=suggestions,
        target_sentiment=args.sentiment,
        sentiment_model_path=str(model_path),
        sentiment_weight=args.sentiment_weight,
    )

    print(f'Input: "{args.text}"')
    print(
        f"Top {args.top_k} suggestions (sentiment={args.sentiment}, weight={args.sentiment_weight:.3f}):"
    )
    if used_neutral_fallback:
        print(
            f'Note: model labels are {model_labels}; "neutral" is unavailable, so reranking is disabled '
            "for this request."
        )
    print(
        f"{'Rank':<{RANK_WIDTH}}"
        f"{'Word':<{WORD_WIDTH}}"
        f"{'LM score':>{LM_SCORE_WIDTH}}"
        f"{'Sentiment':>{SENTIMENT_WIDTH}}"
        f"{'Final':>{FINAL_WIDTH}}"
    )
    for rank, row in enumerate(reranked, start=1):
        print(
            f"{rank:<{RANK_WIDTH}}"
            f"{row['word']:<{WORD_WIDTH}}"
            f"{row['lm_score']:>{LM_SCORE_WIDTH}.6f}"
            f"{row['sentiment_score']:>{SENTIMENT_WIDTH}.6f}"
            f"{row['final_score']:>{FINAL_WIDTH}.6f}"
        )


if __name__ == "__main__":
    main()
