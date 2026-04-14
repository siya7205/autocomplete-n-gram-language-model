import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List

from autocomplete.datasets import DEFAULT_RANDOM_SEED, load_corpus_rows
from autocomplete.preprocess import tokenize


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "en_US.twitter.txt"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "sentiment_labeling_worksheet.csv"


def _build_candidates(rows: List[str], min_tokens: int, dedupe: bool) -> List[str]:
    candidates = []
    seen = set()
    for row in rows:
        if len(tokenize(row)) < min_tokens:
            continue
        if dedupe:
            if row in seen:
                continue
            seen.add(row)
        candidates.append(row)
    return candidates


def generate_labeling_worksheet(
    data_path: Path,
    output_path: Path,
    sample_size: int = 300,
    seed: int = DEFAULT_RANDOM_SEED,
    min_tokens: int = 3,
    dedupe: bool = True,
) -> int:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")

    rows = load_corpus_rows(data_path)
    candidates = _build_candidates(rows, min_tokens=min_tokens, dedupe=dedupe)
    if not candidates:
        raise ValueError("No candidate rows found after filtering.")

    count = min(sample_size, len(candidates))
    rng = random.Random(seed)
    selected_rows = rng.sample(candidates, k=count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["id", "text", "sentiment_label", "notes"])
        writer.writeheader()
        for idx, text in enumerate(selected_rows, start=1):
            writer.writerow(
                {
                    "id": idx,
                    "text": text,
                    "sentiment_label": "",
                    "notes": "",
                }
            )

    return count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a sentiment labeling worksheet CSV by sampling rows from an existing corpus."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Path to source text corpus (newline-delimited).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to output labeling worksheet CSV.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of rows to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=3,
        help="Minimum token count required for a row to be included.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate rows instead of deduplicating.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    written = generate_labeling_worksheet(
        data_path=Path(args.data),
        output_path=Path(args.output),
        sample_size=args.sample_size,
        seed=args.seed,
        min_tokens=args.min_tokens,
        dedupe=not args.keep_duplicates,
    )
    print(f"Wrote {written} rows to {args.output}")


if __name__ == "__main__":
    main()
