import argparse
import random
from pathlib import Path

import nltk

from data_preprocessing import get_tokenized_data, preprocess_data
from language_model import count_n_grams, get_suggestions


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "en_US.twitter.txt"


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


def train_model(data_path: Path, train_split: float = 0.8, minimum_freq: int = 2):
    with open(data_path, "r", encoding="utf-8") as dataset_file:
        data = dataset_file.read()

    tokenized_data = get_tokenized_data(data)
    random.seed(87)
    random.shuffle(tokenized_data)

    train_size = int(len(tokenized_data) * train_split)
    train_data = tokenized_data[:train_size]
    test_data = tokenized_data[train_size:]

    train_data_processed, _, vocabulary = preprocess_data(train_data, test_data, minimum_freq)

    n_gram_counts_list = []
    for n in range(1, 5):
        n_gram_counts_list.append(count_n_grams(train_data_processed, n))

    return vocabulary, n_gram_counts_list


def predict_next_words(text: str, topk: int, data_path: Path, k_smoothing: float = 1.0):
    _ensure_nltk_tokenizer()
    vocabulary, n_gram_counts_list = train_model(data_path=data_path)

    tokens = nltk.word_tokenize(text.lower().strip())
    suggestions = get_suggestions(tokens, n_gram_counts_list, vocabulary, k_smoothing)
    sorted_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
    return sorted_suggestions[:topk]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict top-k next-word suggestions using the N-gram model.")
    parser.add_argument("--text", required=True, help="Input text prefix to autocomplete.")
    parser.add_argument("--topk", type=int, default=5, help="Number of suggestions to print.")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Path to dataset file used for training before prediction.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.topk < 1:
        raise ValueError("--topk must be at least 1.")

    suggestions = predict_next_words(
        text=args.text,
        topk=args.topk,
        data_path=Path(args.data),
    )

    print(f'Input: "{args.text}"')
    print(f"Top {args.topk} suggestions:")
    for rank, (word, probability) in enumerate(suggestions, start=1):
        print(f"{rank}. {word}\t{probability:.6f}")


if __name__ == "__main__":
    main()

