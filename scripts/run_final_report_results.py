import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autocomplete.evaluate import (
    _collect_prefix_tokens,
    _ensure_inputs,
    _evaluate_sentiment_alignment,
    _evaluate_top_k_hit_rate,
    _load_sentiment_texts,
    _train_language_model,
)
from autocomplete.sentiment import load_sentiment_model


TOP_K_SWEEP_DEFAULT = [1, 3, 5]
SENTIMENT_WEIGHT_SWEEP_DEFAULT = [0.0, 0.5, 1.0, 2.0]
SUPPORTED_SENTIMENTS = ("positive", "negative", "neutral")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the final report sweep and save summary tables, plots, and markdown snippet."
    )
    parser.add_argument("--corpus", required=True, help="Path to corpus text file.")
    parser.add_argument("--sentiment-csv", required=True, help="Path to labeled sentiment CSV.")
    parser.add_argument("--model", default="models/sentiment.pkl", help="Path to trained sentiment model (.pkl).")
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
) -> List[Dict[str, float | int | None]]:
    rows: List[Dict[str, float | int | None]] = []

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

            row: Dict[str, float | int | None] = {
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
    plt.title("Top-k Hit Rate vs top_k")
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
    plt.title("Sentiment Alignment Rate vs sentiment_weight")
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


def _build_report_snippet(
    summary_df: pd.DataFrame,
    corpus_path: Path,
    sentiment_csv_path: Path,
    sentiment_model_path: Path,
    generated_at: str,
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
    table_markdown = table_df.to_markdown(index=False)

    return "\n".join(
        [
            "# Final Report Results Snippet",
            "",
            "## Dataset and model inputs",
            f"- Corpus: `{corpus_path}`",
            f"- Sentiment CSV: `{sentiment_csv_path}`",
            f"- Sentiment model: `{sentiment_model_path}`",
            f"- Generated at (UTC): `{generated_at}`",
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
        ]
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

    _ensure_inputs(corpus_path, sentiment_csv_path, sentiment_model_path)

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

    generated_at = datetime.now(timezone.utc).isoformat()
    report_snippet = _build_report_snippet(
        summary_df=summary_df,
        corpus_path=corpus_path,
        sentiment_csv_path=sentiment_csv_path,
        sentiment_model_path=sentiment_model_path,
        generated_at=generated_at,
    )
    (outdir / "REPORT_SNIPPET.md").write_text(report_snippet, encoding="utf-8")

    print(f"[{generated_at}] Saved final report artifacts to {outdir}")
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
