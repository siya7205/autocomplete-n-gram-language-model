import argparse
from pathlib import Path

import nltk

from language_model import estimate_probabilities

from .common import ensure_nltk_data, load_model


def top_k_suggestions(text: str, model: dict, topk: int = 5, smoothing_k: float = 1.0):
    tokens = nltk.word_tokenize(text.lower())
    counts = model["n_gram_counts_list"]
    vocabulary = model["vocabulary"]

    if len(counts) < 2:
        raise ValueError("Model must contain at least unigram and bigram counts.")

    n_gram_counts = counts[-2]
    n_plus1_gram_counts = counts[-1]
    n = len(next(iter(n_gram_counts)))
    previous_n_gram = tokens[-n:]

    probabilities = estimate_probabilities(
        previous_n_gram,
        n_gram_counts,
        n_plus1_gram_counts,
        vocabulary,
        k=smoothing_k,
    )

    ranked = [
        (word, prob)
        for word, prob in probabilities.items()
        if word not in {"<e>", "<unk>", "<s>"}
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:topk]


def main():
    parser = argparse.ArgumentParser(description="Predict top-k autocomplete suggestions.")
    parser.add_argument("--text", required=True, help="Input text prompt.")
    parser.add_argument("--topk", type=int, default=5, help="Number of suggestions to return.")
    parser.add_argument(
        "--model-path",
        default="models/ngram_baseline.pkl",
        help="Path to trained model file.",
    )
    parser.add_argument("--k", type=float, default=1.0, help="Add-k smoothing value.")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {args.model_path}. Run `python -m autocomplete.train` first."
        )

    ensure_nltk_data()
    model = load_model(args.model_path)
    suggestions = top_k_suggestions(args.text, model, topk=args.topk, smoothing_k=args.k)

    print(f"Input: {args.text}")
    print("Top suggestions:")
    for idx, (word, prob) in enumerate(suggestions, start=1):
        print(f"{idx}. {word} ({prob:.6f})")


if __name__ == "__main__":
    main()

