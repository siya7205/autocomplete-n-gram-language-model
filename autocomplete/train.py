import argparse
import random
from pathlib import Path

from data_preprocessing import get_tokenized_data, preprocess_data
from language_model import count_n_grams

from .common import ensure_nltk_data, save_model


def train_baseline_model(
    dataset_path: str,
    model_path: str,
    minimum_freq: int = 2,
    max_n: int = 4,
    train_split: float = 0.8,
    seed: int = 87,
):
    ensure_nltk_data()
    data = Path(dataset_path).read_text(encoding="utf-8")
    tokenized_data = get_tokenized_data(data)

    if len(tokenized_data) < 2:
        raise ValueError("Dataset must contain at least 2 lines/sentences.")

    random.seed(seed)
    random.shuffle(tokenized_data)

    split_index = int(len(tokenized_data) * train_split)
    if split_index <= 0 or split_index >= len(tokenized_data):
        raise ValueError("train_split produced an empty train or test set.")

    train_data = tokenized_data[:split_index]
    test_data = tokenized_data[split_index:]
    train_processed, test_processed, vocabulary = preprocess_data(
        train_data, test_data, minimum_freq
    )

    n_gram_counts_list = [
        count_n_grams(train_processed, n) for n in range(1, max_n + 1)
    ]
    model = {
        "dataset_path": dataset_path,
        "minimum_freq": minimum_freq,
        "max_n": max_n,
        "train_split": train_split,
        "seed": seed,
        "vocabulary": vocabulary,
        "n_gram_counts_list": n_gram_counts_list,
        "train_size": len(train_processed),
        "test_size": len(test_processed),
    }
    save_model(model, model_path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train baseline n-gram model.")
    parser.add_argument(
        "--dataset",
        default="data/disney.txt",
        help="Path to input text file (one sentence per line).",
    )
    parser.add_argument(
        "--model-path",
        default="models/ngram_baseline.pkl",
        help="Path to save trained model.",
    )
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=4)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=87)
    args = parser.parse_args()

    model = train_baseline_model(
        dataset_path=args.dataset,
        model_path=args.model_path,
        minimum_freq=args.min_freq,
        max_n=args.max_n,
        train_split=args.train_split,
        seed=args.seed,
    )
    print(f"Saved model to: {args.model_path}")
    print(
        f"Train docs: {model['train_size']}, Test docs: {model['test_size']}, "
        f"Vocab size: {len(model['vocabulary'])}"
    )


if __name__ == "__main__":
    main()

