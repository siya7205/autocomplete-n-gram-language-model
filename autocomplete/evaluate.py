import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from autocomplete.datasets import load_tokenized_sentences, split_train_test
from autocomplete.preprocess import tokenize
from autocomplete.sentiment import load_sentiment_model, predict_sentiment
from data_preprocessing import preprocess_data
from language_model import count_n_grams, get_suggestions


DEFAULT_CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "en_US.twitter.txt"
DEFAULT_SENTIMENT_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "sentiment_labeled.csv"
DEFAULT_SENTIMENT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sentiment.pkl"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "metrics.json"
SUPPORTED_TARGET_SENTIMENTS = ("positive", "negative", "neutral")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate autocomplete quality (top-k hit rate) and sentiment alignment."
    )
    parser.add_argument(
        "--corpus",
        default=str(DEFAULT_CORPUS_PATH),
        help="Path to newline-delimited corpus used to train/evaluate the language model.",
    )
    parser.add_argument(
        "--sentiment-csv",
        default=str(DEFAULT_SENTIMENT_CSV_PATH),
        help="Path to labeled sentiment CSV containing at least a text column.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of next-word suggestions used for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits and sampling.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for JSON metrics.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=500,
        help="Maximum number of evaluation examples for each metric group.",
    )
    parser.add_argument(
        "--sentiment-model",
        default=str(DEFAULT_SENTIMENT_MODEL_PATH),
        help="Path to trained sentiment model artifact.",
    )
    parser.add_argument(
        "--sentiment-weight",
        "--lambda",
        dest="sentiment_weight",
        type=float,
        default=1.0,
        help="Weight used during sentiment reranking.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of corpus rows used for LM training (rest is held out for evaluation).",
    )
    parser.add_argument(
        "--minimum-freq",
        type=int,
        default=2,
        help="Minimum token frequency kept in LM vocabulary.",
    )
    return parser


def _ensure_inputs(corpus_path: Path, sentiment_csv_path: Path, sentiment_model_path: Path) -> None:
    if not corpus_path.exists():
        raise FileNotFoundError(f'Corpus file not found at "{corpus_path}".')
    if not sentiment_csv_path.exists():
        raise FileNotFoundError(f'Sentiment CSV file not found at "{sentiment_csv_path}".')
    if not sentiment_model_path.exists():
        raise FileNotFoundError(
            f'Sentiment model not found at "{sentiment_model_path}". '
            "Train it first with: python -m autocomplete.train_sentiment --csv <labeled_csv> --out "
            f'"{sentiment_model_path}"'
        )


def _choose_prefix_and_true_next(tokens: Sequence[str], rng: random.Random) -> Tuple[List[str], str] | None:
    if len(tokens) < 2:
        return None
    split_index = rng.randint(1, len(tokens) - 1)
    return list(tokens[:split_index]), str(tokens[split_index])


def _train_language_model(
    corpus_path: Path,
    seed: int,
    train_fraction: float,
    minimum_freq: int,
) -> Tuple[List[List[str]], List[List[str]], List[str], List[Dict[Tuple[str, ...], int]]]:
    tokenized_sentences = load_tokenized_sentences(corpus_path)
    train_data, test_data = split_train_test(
        tokenized_sentences=tokenized_sentences,
        train_fraction=train_fraction,
        shuffle=True,
        seed=seed,
    )
    train_processed, test_processed, vocabulary = preprocess_data(train_data, test_data, minimum_freq)
    n_gram_counts_list = [count_n_grams(train_processed, n) for n in range(1, 5)]
    return train_processed, test_processed, vocabulary, n_gram_counts_list


def _top_k_words(
    prefix_tokens: Sequence[str],
    n_gram_counts_list: Sequence[Dict[Tuple[str, ...], int]],
    vocabulary: Sequence[str],
    top_k: int,
) -> List[Tuple[str, float]]:
    suggestions = get_suggestions(list(prefix_tokens), list(n_gram_counts_list), list(vocabulary), k=1.0)
    sorted_suggestions = sorted(suggestions, key=lambda item: item[1], reverse=True)
    return [(str(word), float(score)) for word, score in sorted_suggestions[:top_k]]


def _evaluate_top_k_hit_rate(
    test_data: Sequence[Sequence[str]],
    n_gram_counts_list: Sequence[Dict[Tuple[str, ...], int]],
    vocabulary: Sequence[str],
    top_k: int,
    max_examples: int,
    seed: int,
) -> Dict[str, float | int]:
    rng = random.Random(seed)
    hits = 0
    examples_used = 0

    for sentence in test_data:
        if examples_used >= max_examples:
            break
        sampled = _choose_prefix_and_true_next(sentence, rng)
        if sampled is None:
            continue
        prefix_tokens, true_next_word = sampled
        predicted_words = [word for word, _ in _top_k_words(prefix_tokens, n_gram_counts_list, vocabulary, top_k)]
        if true_next_word in predicted_words:
            hits += 1
        examples_used += 1

    hit_rate = float(hits / examples_used) if examples_used else 0.0
    return {
        "top_k_hit_rate": hit_rate,
        "hits": hits,
        "examples_used": examples_used,
    }


def _load_sentiment_texts(sentiment_csv_path: Path) -> List[str]:
    dataframe = pd.read_csv(sentiment_csv_path)
    required_columns = {"id", "text", "sentiment_label", "notes"}
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"Sentiment CSV is missing required columns: {sorted(missing)}")
    texts = dataframe["text"].fillna("").astype(str).str.strip()
    texts = texts[texts != ""]
    return texts.tolist()


def _collect_prefix_tokens(texts: Sequence[str], max_examples: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)

    prefixes: List[List[str]] = []
    for idx in indices:
        if len(prefixes) >= max_examples:
            break
        tokens = tokenize(texts[idx])
        sampled = _choose_prefix_and_true_next(tokens, rng)
        if sampled is None:
            continue
        prefix_tokens, _ = sampled
        prefixes.append(prefix_tokens)
    return prefixes


def _rerank_suggestions(
    prefix_tokens: Sequence[str],
    suggestions: Sequence[Tuple[str, float]],
    target_sentiment: str,
    model,
    sentiment_weight: float,
) -> List[Tuple[str, float, float, float]]:
    prefix_text = " ".join(prefix_tokens).strip()
    reranked: List[Tuple[str, float, float, float]] = []

    for word, lm_score in suggestions:
        candidate_text = f"{prefix_text} {word}".strip()
        _, sentiment_scores = predict_sentiment(text=candidate_text, model=model)
        target_score = float(sentiment_scores.get(target_sentiment, 0.0))
        final_score = float(lm_score) + sentiment_weight * target_score
        reranked.append((word, float(lm_score), target_score, final_score))

    reranked.sort(key=lambda item: item[3], reverse=True)
    return reranked


def _evaluate_sentiment_alignment(
    prefix_tokens_list: Sequence[Sequence[str]],
    n_gram_counts_list: Sequence[Dict[Tuple[str, ...], int]],
    vocabulary: Sequence[str],
    model,
    top_k: int,
    sentiment_weight: float,
) -> Dict[str, Dict[str, float | int | bool]]:
    classifier = model.named_steps.get("classifier")
    if classifier is None:
        raise ValueError(
            "Sentiment model is missing the expected 'classifier' pipeline step. "
            "Re-train with: python -m autocomplete.train_sentiment --csv <labeled_csv> --out <model_path>."
        )

    model_labels = {str(label).lower() for label in getattr(classifier, "classes_", [])}
    target_sentiments = [label for label in SUPPORTED_TARGET_SENTIMENTS if label in model_labels]
    if not target_sentiments:
        raise ValueError(
            "Sentiment model does not contain any supported labels from "
            f"{list(SUPPORTED_TARGET_SENTIMENTS)}; found {sorted(model_labels)}."
        )

    alignment: Dict[str, Dict[str, float | int | bool]] = {}
    for target_sentiment in SUPPORTED_TARGET_SENTIMENTS:
        if target_sentiment not in target_sentiments:
            alignment[target_sentiment] = {
                "supported": False,
                "alignment_rate": 0.0,
                "aligned_suggestions": 0,
                "total_suggestions": 0,
                "prefixes_evaluated": 0,
            }
            continue

        aligned = 0
        total = 0
        prefixes_evaluated = 0
        for prefix_tokens in prefix_tokens_list:
            suggestions = _top_k_words(prefix_tokens, n_gram_counts_list, vocabulary, top_k)
            if not suggestions:
                continue
            reranked = _rerank_suggestions(
                prefix_tokens=prefix_tokens,
                suggestions=suggestions,
                target_sentiment=target_sentiment,
                model=model,
                sentiment_weight=sentiment_weight,
            )[:top_k]
            for word, _, _, _ in reranked:
                candidate_text = f"{' '.join(prefix_tokens).strip()} {word}".strip()
                predicted_label, _ = predict_sentiment(text=candidate_text, model=model)
                if str(predicted_label).lower() == target_sentiment:
                    aligned += 1
                total += 1
            prefixes_evaluated += 1

        alignment_rate = float(aligned / total) if total else 0.0
        alignment[target_sentiment] = {
            "supported": True,
            "alignment_rate": alignment_rate,
            "aligned_suggestions": aligned,
            "total_suggestions": total,
            "prefixes_evaluated": prefixes_evaluated,
        }

    return alignment


def evaluate(
    corpus_path: Path,
    sentiment_csv_path: Path,
    sentiment_model_path: Path,
    top_k: int,
    seed: int,
    out_path: Path,
    max_examples: int,
    sentiment_weight: float,
    train_fraction: float,
    minimum_freq: int,
) -> Dict[str, object]:
    _ensure_inputs(corpus_path, sentiment_csv_path, sentiment_model_path)

    _, test_data, vocabulary, n_gram_counts_list = _train_language_model(
        corpus_path=corpus_path,
        seed=seed,
        train_fraction=train_fraction,
        minimum_freq=minimum_freq,
    )

    top_k_metrics = _evaluate_top_k_hit_rate(
        test_data=test_data,
        n_gram_counts_list=n_gram_counts_list,
        vocabulary=vocabulary,
        top_k=top_k,
        max_examples=max_examples,
        seed=seed,
    )

    sentiment_texts = _load_sentiment_texts(sentiment_csv_path)
    prefix_tokens_list = _collect_prefix_tokens(sentiment_texts, max_examples=max_examples, seed=seed)
    model = load_sentiment_model(str(sentiment_model_path))

    sentiment_alignment = _evaluate_sentiment_alignment(
        prefix_tokens_list=prefix_tokens_list,
        n_gram_counts_list=n_gram_counts_list,
        vocabulary=vocabulary,
        model=model,
        top_k=top_k,
        sentiment_weight=sentiment_weight,
    )

    metrics: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "corpus": str(corpus_path),
            "sentiment_csv": str(sentiment_csv_path),
            "sentiment_model": str(sentiment_model_path),
            "top_k": top_k,
            "seed": seed,
            "max_examples": max_examples,
            "sentiment_weight": sentiment_weight,
            "train_fraction": train_fraction,
            "minimum_freq": minimum_freq,
        },
        "autocomplete": top_k_metrics,
        "sentiment_alignment": {
            "method": "A suggestion is aligned when classifier predicted label equals target sentiment.",
            "per_sentiment": sentiment_alignment,
            "prefixes_sampled": len(prefix_tokens_list),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _print_summary(metrics: Dict[str, object]) -> None:
    autocomplete = metrics["autocomplete"]
    sentiment_alignment = metrics["sentiment_alignment"]
    print("Evaluation summary")
    print("==================")
    print(
        "Autocomplete top-k hit rate: "
        f"{autocomplete['top_k_hit_rate']:.4f} "
        f"({autocomplete['hits']}/{autocomplete['examples_used']})"
    )
    print("Sentiment alignment:")
    for sentiment, stats in sentiment_alignment["per_sentiment"].items():
        if not stats["supported"]:
            print(f"- {sentiment:<8} unsupported by model")
            continue
        print(
            f"- {sentiment:<8} {stats['alignment_rate']:.4f} "
            f"({stats['aligned_suggestions']}/{stats['total_suggestions']})"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if args.max_examples < 1:
        raise ValueError("--max-examples must be at least 1.")
    if not 0 < args.train_fraction <= 1:
        raise ValueError("--train-fraction must be greater than 0 and at most 1.")
    if args.minimum_freq < 1:
        raise ValueError("--minimum-freq must be at least 1.")

    metrics = evaluate(
        corpus_path=Path(args.corpus),
        sentiment_csv_path=Path(args.sentiment_csv),
        sentiment_model_path=Path(args.sentiment_model),
        top_k=args.top_k,
        seed=args.seed,
        out_path=Path(args.out),
        max_examples=args.max_examples,
        sentiment_weight=args.sentiment_weight,
        train_fraction=args.train_fraction,
        minimum_freq=args.minimum_freq,
    )

    _print_summary(metrics)
    print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
