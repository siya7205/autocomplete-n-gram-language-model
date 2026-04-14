import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocomplete.datasets import load_corpus_rows
from autocomplete.preprocess import tokenize


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "sentiment_labeled_weak.csv"
DEFAULT_MIN_PER_CLASS = 50
DEFAULT_MAX_ROWS = 1000

POS_WORDS = {
    "amazing",
    "awesome",
    "best",
    "beautiful",
    "brilliant",
    "calm",
    "delight",
    "enjoy",
    "excellent",
    "fair",
    "fantastic",
    "friend",
    "gentle",
    "good",
    "grace",
    "great",
    "happy",
    "heaven",
    "joy",
    "joyful",
    "kind",
    "like",
    "love",
    "lovely",
    "nice",
    "outstanding",
    "perfect",
    "pleasant",
    "positive",
    "peace",
    "praise",
    "smile",
    "strong",
    "success",
    "sweet",
    "terrific",
    "virtue",
    "wonderful",
}

NEG_WORDS = {
    "alas",
    "angry",
    "awful",
    "bad",
    "banish",
    "banished",
    "betray",
    "blood",
    "boring",
    "broken",
    "bloody",
    "cold",
    "curse",
    "cruel",
    "cry",
    "damage",
    "dark",
    "dead",
    "death",
    "died",
    "die",
    "disappoint",
    "disaster",
    "doom",
    "dread",
    "enemy",
    "evil",
    "fail",
    "fear",
    "foul",
    "grief",
    "hate",
    "hell",
    "horrible",
    "hurt",
    "kill",
    "mad",
    "murder",
    "loss",
    "negative",
    "pain",
    "poor",
    "problem",
    "rough",
    "sad",
    "scared",
    "sin",
    "sorrow",
    "stress",
    "terrible",
    "toxic",
    "tyrant",
    "ugly",
    "villain",
    "violent",
    "war",
    "wars",
    "weak",
    "weep",
    "woe",
    "worse",
    "worst",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate weakly supervised sentiment labels from a corpus text file or an existing worksheet CSV."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--corpus",
        help="Path to newline-delimited source corpus text file.",
    )
    source.add_argument(
        "--worksheet",
        help="Path to worksheet CSV with at least a text column.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to output weak-labeled CSV.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Maximum number of rows sampled from corpus input (ignored for worksheet input).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=DEFAULT_MIN_PER_CLASS,
        help="Minimum labeled examples required for each class (positive and negative).",
    )
    return parser


def _sample_rows(rows: Sequence[str], max_rows: int, seed: int) -> List[str]:
    cleaned = [row.strip() for row in rows if str(row).strip()]
    if max_rows < 1:
        raise ValueError("--max-rows must be at least 1.")
    if len(cleaned) <= max_rows:
        return cleaned
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(cleaned)), k=max_rows))
    return [cleaned[index] for index in selected_indices]


def _rows_from_corpus(corpus_path: Path, max_rows: int, seed: int) -> List[Dict[str, str]]:
    if not corpus_path.exists():
        raise FileNotFoundError(f'Corpus file not found at "{corpus_path}".')
    sampled_rows = _sample_rows(load_corpus_rows(corpus_path), max_rows=max_rows, seed=seed)
    return [{"id": str(idx), "text": text} for idx, text in enumerate(sampled_rows, start=1)]


def _rows_from_worksheet(worksheet_path: Path) -> List[Dict[str, str]]:
    if not worksheet_path.exists():
        raise FileNotFoundError(f'Worksheet CSV not found at "{worksheet_path}".')
    dataframe = pd.read_csv(worksheet_path)
    if "text" not in dataframe.columns:
        raise ValueError('Worksheet CSV must include a "text" column.')
    texts = dataframe["text"].fillna("").astype(str).str.strip()
    ids = dataframe["id"] if "id" in dataframe.columns else range(1, len(dataframe) + 1)

    rows: List[Dict[str, str]] = []
    for row_id, text in zip(ids, texts):
        if not text:
            continue
        rows.append({"id": str(row_id), "text": text})
    return rows


def _label_text(text: str, pos_words: Iterable[str], neg_words: Iterable[str]) -> tuple[str, str]:
    tokens = [token.lower() for token in tokenize(text)]
    pos_set = set(pos_words)
    neg_set = set(neg_words)
    pos_count = sum(1 for token in tokens if token in pos_set)
    neg_count = sum(1 for token in tokens if token in neg_set)

    if pos_count > neg_count:
        label = "positive"
    elif neg_count > pos_count:
        label = "negative"
    else:
        label = ""
    notes = f"weak_lexicon;pos_count={pos_count};neg_count={neg_count}"
    return label, notes


def generate_weak_labels(
    out_path: Path,
    seed: int = 42,
    corpus_path: Path | None = None,
    worksheet_path: Path | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    pos_words: Iterable[str] = POS_WORDS,
    neg_words: Iterable[str] = NEG_WORDS,
) -> Dict[str, int | str]:
    if (corpus_path is None) == (worksheet_path is None):
        raise ValueError("Provide exactly one source: corpus_path or worksheet_path.")
    if min_per_class < 1:
        raise ValueError("--min-per-class must be at least 1.")

    if corpus_path is not None:
        source_rows = _rows_from_corpus(corpus_path, max_rows=max_rows, seed=seed)
        source_hint = "increase --max-rows or expand POS_WORDS/NEG_WORDS lexicons"
    else:
        source_rows = _rows_from_worksheet(worksheet_path)  # type: ignore[arg-type]
        source_hint = "use a larger worksheet/corpus or expand POS_WORDS/NEG_WORDS lexicons"

    output_rows: List[Dict[str, str]] = []
    for row in source_rows:
        label, notes = _label_text(row["text"], pos_words=pos_words, neg_words=neg_words)
        if not label:
            continue
        output_rows.append(
            {
                "id": row["id"],
                "text": row["text"],
                "sentiment_label": label,
                "notes": notes,
            }
        )

    label_counts = Counter([row["sentiment_label"] for row in output_rows])
    pos_count = int(label_counts.get("positive", 0))
    neg_count = int(label_counts.get("negative", 0))
    if pos_count < min_per_class or neg_count < min_per_class:
        raise ValueError(
            "Weak labeling produced too few examples per class: "
            f"positive={pos_count}, negative={neg_count}, required>={min_per_class}. "
            f"Try to {source_hint}."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["id", "text", "sentiment_label", "notes"])
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "output_path": str(out_path),
        "input_rows": len(source_rows),
        "labeled_rows": len(output_rows),
        "positive_rows": pos_count,
        "negative_rows": neg_count,
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    metrics = generate_weak_labels(
        corpus_path=Path(args.corpus) if args.corpus else None,
        worksheet_path=Path(args.worksheet) if args.worksheet else None,
        out_path=Path(args.out),
        max_rows=args.max_rows,
        seed=args.seed,
        min_per_class=args.min_per_class,
    )
    print(f"Weak labels saved: {metrics['output_path']}")
    print(f"Input rows: {metrics['input_rows']}")
    print(f"Labeled rows: {metrics['labeled_rows']}")
    print(f"Positive rows: {metrics['positive_rows']}")
    print(f"Negative rows: {metrics['negative_rows']}")


if __name__ == "__main__":
    main()
    "wicked",
    "threat",
    "torment",
    "sick",
    "sickness",
    "rage",
    "grave",
    "despair",
    "danger",
