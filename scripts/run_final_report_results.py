import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocomplete.evaluate import (
    _collect_prefix_tokens,
    _evaluate_sentiment_alignment,
    _evaluate_top_k_hit_rate,
    _load_sentiment_texts,
    _train_language_model,
)
from autocomplete.sentiment import load_sentiment_model, train_sentiment_model
from scripts.weak_label_sentiment import generate_weak_labels


TOP_K_SWEEP_DEFAULT = [1, 3, 5]
SENTIMENT_WEIGHT_SWEEP_DEFAULT = [0.0, 0.5, 1.0, 2.0]
SUPPORTED_SENTIMENTS = ("positive", "negative", "neutral")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run weak-labeling (optional), train sentiment model, and generate final report artifacts "
            "(sweep tables, plots, markdown snippet)."
        )
    )
    parser.add_argument("--corpus", required=True, help="Path to corpus text file.")
    parser.add_argument(
        "--sentiment-csv",
        default="data/sentiment_labeled_weak.csv",
        help="Path to labeled sentiment CSV used for training/evaluation.",
    )
    parser.add_argument(
        "--model",
        default="models/sentiment_weak.pkl",
        help="Path to trained sentiment model (.pkl) output.",
    )
    parser.add_argument("--outdir", default="results/final", help="Output directory for report artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=500,
        help="Maximum number of examples used in each metric evaluation.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of corpus used for LM training.",
    )
    parser.add_argument(
        "--minimum-freq",
        type=int,
        default=2,
        help="Minimum token frequency kept in LM vocabulary.",
    )
    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=TOP_K_SWEEP_DEFAULT,
        help="Sweep values for top_k (default: 1 3 5).",
    )
    parser.add_argument(
        "--sentiment-weight-values",
        type=float,
        nargs="+",
        default=SENTIMENT_WEIGHT_SWEEP_DEFAULT,
        help="Sweep values for sentiment_weight (default: 0.0 0.5 1.0 2.0).",
    )
    parser.add_argument(
        "--run-weak-labeling",
        action="store_true",
        help="Generate weak labels before training and evaluation.",
    )
    parser.add_argument(
        "--worksheet",
        help="Optional worksheet CSV source for weak labeling (if omitted, weak labels are generated from --corpus).",
    )
    parser.add_argument(
        "--weak-max-rows",
        type=int,
        default=1000,
        help="Maximum rows sampled when weak labeling from corpus.",
    )
    parser.add_argument(
        "--weak-min-per-class",
        type=int,
        default=50,
        help="Minimum weak-labeled rows required for each class (positive/negative).",
    )
    return parser


def _make_summary_rows(
    test_data,
    n_gram_counts_list,
    vocabulary,
    prefix_tokens_list,
    sentiment_model,
    top_k_values: List[int],
    sentiment_weight_values: List[float],
    seed: int,
    max_examples: int,
) -> List[Dict[str, Optional[Union[float, int]]]]:
    rows: List[Dict[str, Optional[Union[float, int]]]] = []

    for top_k in top_k_values:
        for sentiment_weight in sentiment_weight_values:
            top_k_metrics = _evaluate_top_k_hit_rate(
                test_data=test_data,
                n_gram_counts_list=n_gram_counts_list,
                vocabulary=vocabulary,
                top_k=top_k,
                max_examples=max_examples,
                seed=seed,
            )
            alignment = _evaluate_sentiment_alignment(
                prefix_tokens_list=prefix_tokens_list,
                n_gram_counts_list=n_gram_counts_list,
                vocabulary=vocabulary,
                model=sentiment_model,
                top_k=top_k,
                sentiment_weight=sentiment_weight,
            )

            row: Dict[str, Optional[Union[float, int]]] = {
                "top_k": top_k,
                "sentiment_weight": sentiment_weight,
                "top_k_hit_rate": float(top_k_metrics["top_k_hit_rate"]),
                "examples_used": int(top_k_metrics["examples_used"]),
            }
            for sentiment in SUPPORTED_SENTIMENTS:
                sentiment_stats = alignment.get(sentiment, {})
                if sentiment_stats.get("supported"):
                    row[f"alignment_{sentiment}"] = float(sentiment_stats["alignment_rate"])
                else:
                    row[f"alignment_{sentiment}"] = None
            rows.append(row)
    return rows


def _save_tables(summary_df: pd.DataFrame, outdir: Path) -> None:
    summary_csv_path = outdir / "summary.csv"
    summary_json_path = outdir / "summary.json"
    summary_df.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )


def _plot_topk_hit_rate(summary_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sentiment_weight_values = sorted(summary_df["sentiment_weight"].unique().tolist())
    top_k_values = sorted(summary_df["top_k"].unique().tolist())
    for sentiment_weight in sentiment_weight_values:
        subset = summary_df[summary_df["sentiment_weight"] == sentiment_weight].sort_values("top_k")
        if subset.empty:
            continue
        plt.plot(
            subset["top_k"],
            subset["top_k_hit_rate"],
            marker="o",
            label=f"sentiment_weight={sentiment_weight}",
        )
    plt.title("Top-K Hit Rate vs Top-K")
    plt.xlabel("top_k")
    plt.ylabel("top-k hit rate")
    plt.xticks(top_k_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "topk_hit_rate.png", dpi=150)
    plt.close()


def _plot_sentiment_alignment(summary_df: pd.DataFrame, outdir: Path) -> None:
    alignment_col_map = {
        "positive": "alignment_positive",
        "negative": "alignment_negative",
        "neutral": "alignment_neutral",
    }
    alignment_value_cols = [col for col in alignment_col_map.values() if col in summary_df.columns]
    grouped = summary_df.groupby("sentiment_weight", as_index=False)[alignment_value_cols].mean()

    plt.figure(figsize=(8, 5))
    for sentiment in ("positive", "negative", "neutral"):
        col = alignment_col_map[sentiment]
        if col not in grouped.columns or grouped[col].isna().all():
            continue
        plt.plot(grouped["sentiment_weight"], grouped[col], marker="o", label=sentiment)
    plt.title("Sentiment Alignment Rate vs Sentiment Weight")
    plt.xlabel("sentiment_weight")
    plt.ylabel("alignment rate")
    plt.xticks(sorted(grouped["sentiment_weight"].tolist()))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "sentiment_alignment.png", dpi=150)
    plt.close()


def _best_row_by_metric(summary_df: pd.DataFrame, metric: str) -> pd.Series:
    idx = summary_df[metric].astype(float).idxmax()
    return summary_df.loc[idx]


def _dataframe_to_markdown_table(dataframe: pd.DataFrame) -> str:
    headers = [str(column) for column in dataframe.columns]
    separator = ["---" for _ in headers]
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(separator)} |",
    ]

    for _, row in dataframe.iterrows():
        values: List[str] = []
        for column in dataframe.columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append(f"| {' | '.join(values)} |")
    return "\n".join(lines)


def _build_report_snippet(
    summary_df: pd.DataFrame,
    corpus_path: Path,
    sentiment_csv_path: Path,
    sentiment_model_path: Path,
    seed: int,
) -> str:
    alignment_metric_cols = ["alignment_positive", "alignment_negative"]
    if "alignment_neutral" in summary_df.columns and not summary_df["alignment_neutral"].isna().all():
        alignment_metric_cols.append("alignment_neutral")

    alignment_score = summary_df[alignment_metric_cols].mean(axis=1, skipna=True)
    summary_with_alignment = summary_df.assign(alignment_mean=alignment_score)

    best_hit = _best_row_by_metric(summary_df, "top_k_hit_rate")
    best_alignment = _best_row_by_metric(summary_with_alignment, "alignment_mean")

    table_df = summary_df.copy()
    float_cols = [c for c in table_df.columns if c.startswith("alignment_") or c == "top_k_hit_rate"]
    for col in float_cols:
        table_df[col] = table_df[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
    table_markdown = _dataframe_to_markdown_table(table_df)

    return "\n".join(
        [
            "# Final Report Results Snippet",
            "",
            "## Dataset and model inputs",
            f"- Corpus: `{corpus_path}`",
            f"- Sentiment CSV: `{sentiment_csv_path}`",
            f"- Sentiment model: `{sentiment_model_path}`",
            f"- Seed: `{seed}`",
            "",
            "## Best observed configurations",
            (
                f"- Best top-k hit rate: **{best_hit['top_k_hit_rate']:.4f}** "
                f"(top_k={int(best_hit['top_k'])}, sentiment_weight={best_hit['sentiment_weight']})"
            ),
            (
                f"- Best sentiment alignment (mean over available labels): "
                f"**{best_alignment['alignment_mean']:.4f}** "
                f"(top_k={int(best_alignment['top_k'])}, sentiment_weight={best_alignment['sentiment_weight']})"
            ),
            "",
            "## Summary table",
            table_markdown,
            "",
            "_Note: weak labels are heuristic and indicative, not ground truth._",
            "",
        ]
    )


def _maybe_generate_weak_labels(args: argparse.Namespace, sentiment_csv_path: Path) -> None:
    if not args.run_weak_labeling:
        return
    weak_metrics = generate_weak_labels(
        corpus_path=None if args.worksheet else Path(args.corpus),
        worksheet_path=Path(args.worksheet) if args.worksheet else None,
        out_path=sentiment_csv_path,
        max_rows=args.weak_max_rows,
        seed=args.seed,
        min_per_class=args.weak_min_per_class,
    )
    print(
        "Weak labeling complete: "
        f"{weak_metrics['labeled_rows']} labeled rows "
        f"(positive={weak_metrics['positive_rows']}, negative={weak_metrics['negative_rows']})."
    )


def run(args: argparse.Namespace) -> int:
    if args.max_examples < 1:
        raise ValueError("--max-examples must be at least 1.")
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be greater than 0 and less than 1.")
    if args.minimum_freq < 1:
        raise ValueError("--minimum-freq must be at least 1.")
    if not args.top_k_values or any(value < 1 for value in args.top_k_values):
        raise ValueError("--top-k-values must contain one or more integers >= 1.")
    if not args.sentiment_weight_values:
        raise ValueError("--sentiment-weight-values must contain one or more values.")

    corpus_path = Path(args.corpus)
    sentiment_csv_path = Path(args.sentiment_csv)
    sentiment_model_path = Path(args.model)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(f'Corpus file not found at "{corpus_path}".')

    _maybe_generate_weak_labels(args, sentiment_csv_path)
    if not sentiment_csv_path.exists():
        raise FileNotFoundError(
            f'Sentiment CSV not found at "{sentiment_csv_path}". '
            "Either provide an existing file via --sentiment-csv or set --run-weak-labeling."
        )

    train_metrics = train_sentiment_model(
        csv_path=str(sentiment_csv_path),
        model_out_path=str(sentiment_model_path),
        seed=args.seed,
    )
    print(
        "Sentiment model trained: "
        f"{sentiment_model_path} "
        f"(rows={train_metrics['labeled_rows']}, accuracy={train_metrics['accuracy']:.4f})."
    )

    _, test_data, vocabulary, n_gram_counts_list = _train_language_model(
        corpus_path=corpus_path,
        seed=args.seed,
        train_fraction=args.train_fraction,
        minimum_freq=args.minimum_freq,
    )
    if not test_data:
        raise ValueError(
            "No held-out corpus examples available for evaluation. "
            "Use a larger corpus or lower --train-fraction."
        )

    sentiment_texts = _load_sentiment_texts(sentiment_csv_path)
    prefix_tokens_list = _collect_prefix_tokens(sentiment_texts, max_examples=args.max_examples, seed=args.seed)
    if not prefix_tokens_list:
        raise ValueError(
            "No valid sentiment-evaluation prefixes were produced from the sentiment CSV. "
            "Ensure there are non-empty text rows with at least 2 tokens."
        )
    sentiment_model = load_sentiment_model(str(sentiment_model_path))

    rows = _make_summary_rows(
        test_data=test_data,
        n_gram_counts_list=n_gram_counts_list,
        vocabulary=vocabulary,
        prefix_tokens_list=prefix_tokens_list,
        sentiment_model=sentiment_model,
        top_k_values=args.top_k_values,
        sentiment_weight_values=args.sentiment_weight_values,
        seed=args.seed,
        max_examples=args.max_examples,
    )
    summary_df = pd.DataFrame(rows).sort_values(["top_k", "sentiment_weight"]).reset_index(drop=True)

    _save_tables(summary_df, outdir)
    _plot_topk_hit_rate(summary_df, outdir)
    _plot_sentiment_alignment(summary_df, outdir)

    report_snippet = _build_report_snippet(
        summary_df=summary_df,
        corpus_path=corpus_path,
        sentiment_csv_path=sentiment_csv_path,
        sentiment_model_path=sentiment_model_path,
        seed=args.seed,
    )
    (outdir / "REPORT_SNIPPET.md").write_text(report_snippet, encoding="utf-8")

    print(f"Saved final report artifacts to {outdir}")
    print(f"- {outdir / 'summary.csv'}")
    print(f"- {outdir / 'summary.json'}")
    print(f"- {outdir / 'topk_hit_rate.png'}")
    print(f"- {outdir / 'sentiment_alignment.png'}")
    print(f"- {outdir / 'REPORT_SNIPPET.md'}")
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
