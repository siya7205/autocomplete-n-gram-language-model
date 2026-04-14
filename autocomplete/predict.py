import argparse
import random
from pathlib import Path

import nltk

from data_preprocessing import get_tokenized_data, preprocess_data
from language_model import count_n_grams, get_suggestions


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "en_US.twitter.txt"
RANDOM_SEED = 87


def _ensure_nltk_tokenizer() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def train_model(data_path: Path, train_fraction: float = 0.8, minimum_freq: int = 2):
    """Train N-gram count tables from a text dataset.

    Args:
        data_path: Path to newline-delimited text data.
        train_fraction: Fraction of tokenized sentences used for training.
        minimum_freq: Minimum token count kept in closed vocabulary.

    Returns:
        A tuple of (vocabulary, n_gram_counts_list).
    """
    with open(data_path, "r", encoding="utf-8") as dataset_file:
        data = dataset_file.read()

    tokenized_data = get_tokenized_data(data)
    random.seed(RANDOM_SEED)
    random.shuffle(tokenized_data)

    train_size = int(len(tokenized_data) * train_fraction)
    train_data = tokenized_data[:train_size]

    train_data_processed, _, vocabulary = preprocess_data(
        train_data, tokenized_data[train_size:], minimum_freq
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
    _ensure_nltk_tokenizer()
    vocabulary, n_gram_counts_list = train_model(data_path=data_path)

    tokens = nltk.word_tokenize(text.lower().strip())
    suggestions = get_suggestions(tokens, n_gram_counts_list, vocabulary, k_smoothing)
    sorted_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
    return sorted_suggestions[:top_k]


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

    print(f'Input: "{args.text}"')
    print(f"Top {args.top_k} suggestions:")
    for rank, (word, probability) in enumerate(suggestions, start=1):
        print(f"{rank}. {word}\t{probability:.6f}")


if __name__ == "__main__":
    main()
