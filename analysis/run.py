#!/usr/bin/env python3
"""
analysis/run.py — Multi-dataset analysis runner for the autocomplete n-gram language model.

Usage
-----
Run with built-in default datasets (stored in data/):
    python -m analysis.run

Run with explicit datasets (name:path pairs):
    python -m analysis.run \\
        --dataset Twitter:data/en_US.twitter.txt \\
        --dataset Disney:data/disney.txt

Run with a single external dataset:
    python -m analysis.run --dataset MyCorpus:/path/to/corpus.txt

Expected dataset format: plain-text file, one sentence per line.

Outputs
-------
    reports/REPORT.md               — human-readable Markdown report
    reports/artifacts/metrics.csv   — per-dataset metrics table
    reports/artifacts/metrics.json  — same metrics in JSON format
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter

import nltk
import numpy as np
import pandas as pd

# Ensure NLTK punkt tokeniser is available without requiring internet
nltk.data.path.insert(0, ".")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# Add repo root to path so the existing modules are importable whether the
# script is called as `python analysis/run.py` or `python -m analysis.run`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_preprocessing import get_tokenized_data, preprocess_data  # noqa: E402
from language_model import (  # noqa: E402
    count_n_grams,
    estimate_probability,
    estimate_probabilities,
    calculate_perplexity,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 87
TRAIN_FRAC = 0.70
VALID_FRAC = 0.15
# TEST_FRAC  = 1 - TRAIN_FRAC - VALID_FRAC  (implicit)
MINIMUM_FREQ = 2
K_SMOOTH = 1.0
MAX_N = 4           # highest n-gram order
TOP_K = 10          # top-k n-grams to report
MAX_EVAL_SENTS = 500  # cap sentences used for perplexity / autocomplete eval
MAX_QUAL_EXAMPLES = 5  # qualitative examples per dataset

DEFAULT_DATASETS = {
    "Twitter": "data/en_US.twitter.txt",
    "Shakespeare": "data/merchant.txt",
    "Disney": "data/disney.txt",
    "OnCampus": "data/oncampus_no_numbers.txt",
}

REPORT_DIR = "reports"
ARTIFACTS_DIR = os.path.join(REPORT_DIR, "artifacts")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_and_split(path: str, seed: int = RANDOM_SEED):
    """Load a dataset file and split into train / validation / test lists of
    tokenised sentences.

    Parameters
    ----------
    path:
        Path to a plain-text file with one sentence per line.
    seed:
        Random seed used for reproducible shuffling.

    Returns
    -------
    train_raw, valid_raw, test_raw : list[list[str]]
        Raw (pre-preprocessing) tokenised sentence lists.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = fh.read()
    tokenized = get_tokenized_data(data)
    random.seed(seed)
    random.shuffle(tokenized)
    n = len(tokenized)
    train_end = int(n * TRAIN_FRAC)
    valid_end = train_end + int(n * VALID_FRAC)
    return tokenized[:train_end], tokenized[train_end:valid_end], tokenized[valid_end:]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def compute_dataset_stats(train_raw, valid_raw, test_raw, vocabulary):
    """Return basic corpus size and vocabulary statistics.

    Parameters
    ----------
    train_raw, valid_raw, test_raw:
        Raw tokenised splits (before OOV replacement).
    vocabulary:
        Closed-vocabulary list built from training data.

    Returns
    -------
    dict with keys: n_docs_train, n_docs_valid, n_docs_test,
        n_tokens_train, n_tokens_test, vocab_size, oov_rate_test
    """
    def _flatten(split):
        return [tok for sent in split for tok in sent]

    train_tokens = _flatten(train_raw)
    test_tokens = _flatten(test_raw)
    vocab_set = set(vocabulary)
    oov_count = sum(1 for t in test_tokens if t not in vocab_set)
    oov_rate = oov_count / len(test_tokens) if test_tokens else 0.0

    return {
        "n_docs_train": len(train_raw),
        "n_docs_valid": len(valid_raw),
        "n_docs_test": len(test_raw),
        "n_tokens_train": len(train_tokens),
        "n_tokens_valid": len(_flatten(valid_raw)),
        "n_tokens_test": len(test_tokens),
        "vocab_size": len(vocabulary),
        "oov_rate_test": round(oov_rate, 4),
    }


def compute_top_ngrams(train_processed, n: int, top_k: int = TOP_K):
    """Return the *top_k* most common n-grams from processed training data.

    Parameters
    ----------
    train_processed:
        OOV-replaced, tokenised training sentences.
    n:
        N-gram order (1, 2, or 3).
    top_k:
        Number of top n-grams to return.

    Returns
    -------
    list of ((token, ...), count) tuples.
    """
    counter: Counter = Counter()
    for sent in train_processed:
        tokens = tuple(sent)
        for i in range(len(tokens) - n + 1):
            counter[tokens[i : i + n]] += 1
    return counter.most_common(top_k)


def compute_model_stats(ngram_counts_list):
    """Return the number of unique n-gram types for each order.

    Parameters
    ----------
    ngram_counts_list:
        List of n-gram count dicts, ordered from 1-gram to N-gram.

    Returns
    -------
    dict mapping n-gram order (int) to number of unique types (int).
    """
    return {n + 1: len(counts) for n, counts in enumerate(ngram_counts_list)}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _safe_perplexity(sentence, ngram_counts, n_plus1_gram_counts, vocab_size):
    """Compute perplexity, returning NaN on failure."""
    try:
        return calculate_perplexity(
            sentence, ngram_counts, n_plus1_gram_counts, vocab_size, k=K_SMOOTH
        )
    except Exception:
        return float("nan")


def compute_perplexity_stats(test_processed, ngram_counts_list, vocab_size: int):
    """Compute mean perplexity for each n-gram order on the test split.

    Sentences shorter than n are skipped.

    Parameters
    ----------
    test_processed:
        OOV-replaced test sentences.
    ngram_counts_list:
        List of n-gram count dicts (1-gram … N-gram).
    vocab_size:
        Size of the closed vocabulary (including <unk> and <e>).

    Returns
    -------
    dict mapping n-gram order to mean perplexity.
    """
    sentences = test_processed[:MAX_EVAL_SENTS]
    result = {}
    for n in range(1, len(ngram_counts_list)):
        ngram_counts = ngram_counts_list[n - 1]
        n_plus1_counts = ngram_counts_list[n]
        perps = [
            _safe_perplexity(s, ngram_counts, n_plus1_counts, vocab_size)
            for s in sentences
            if len(s) >= n
        ]
        finite = [p for p in perps if math.isfinite(p)]
        result[n] = round(sum(finite) / len(finite), 2) if finite else float("nan")
    return result


def _top5_suggestions(previous_tokens, ngram_counts, n_plus1_counts, vocabulary):
    """Return top-5 (word, prob) pairs for the next word prediction.

    Parameters
    ----------
    previous_tokens:
        Sequence of preceding tokens (list[str]).
    ngram_counts, n_plus1_counts:
        N-gram and (N+1)-gram count dicts for one model order.
    vocabulary:
        Closed-vocabulary list (without <e> / <unk>, those are added inside
        estimate_probabilities).

    Returns
    -------
    list of up to 5 (word, probability) tuples, sorted by descending probability.
    """
    n = len(next(iter(ngram_counts.keys())))
    prev = previous_tokens[-n:]
    probs = estimate_probabilities(prev, ngram_counts, n_plus1_counts, vocabulary, k=K_SMOOTH)
    # Exclude special tokens from autocomplete suggestions
    probs.pop("<e>", None)
    probs.pop("<unk>", None)
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return ranked[:5]


def compute_autocomplete_accuracy(test_processed, ngram_counts_list, vocabulary):
    """Measure Top-1 / Top-5 accuracy and mean log-prob of the true next word.

    For each sentence with at least 2 tokens, every consecutive (prefix, word)
    pair is treated as one prediction instance.  The highest-order model is
    used for prediction.

    Parameters
    ----------
    test_processed:
        OOV-replaced test sentences (at most MAX_EVAL_SENTS used).
    ngram_counts_list:
        Ordered list of n-gram count dicts (1-gram first).
    vocabulary:
        Closed-vocabulary list.

    Returns
    -------
    dict with keys: top1_accuracy, top5_accuracy, mean_log_prob, n_instances
    """
    # Use the highest-order model available (last pair in the list)
    n_order = len(ngram_counts_list) - 1  # index of n-gram counts
    ngram_counts = ngram_counts_list[n_order - 1]
    n_plus1_counts = ngram_counts_list[n_order]
    vocab_size = len(vocabulary) + 2  # +<e> +<unk>

    sentences = test_processed[:MAX_EVAL_SENTS]
    top1_hits = 0
    top5_hits = 0
    log_probs = []
    n_instances = 0

    for sent in sentences:
        if len(sent) < 2:
            continue
        for i in range(1, len(sent)):
            prefix = sent[:i]
            true_word = sent[i]
            try:
                top5 = _top5_suggestions(prefix, ngram_counts, n_plus1_counts, vocabulary)
            except Exception:
                continue
            words_top5 = [w for w, _ in top5]
            if words_top5 and words_top5[0] == true_word:
                top1_hits += 1
            if true_word in words_top5:
                top5_hits += 1
            # log-prob of true word
            n = len(next(iter(ngram_counts.keys())))
            prev = tuple(prefix[-n:])
            p = estimate_probability(
                true_word, prev, ngram_counts, n_plus1_counts, vocab_size, k=K_SMOOTH
            )
            if p > 0:
                log_probs.append(math.log(p))
            n_instances += 1

    if n_instances == 0:
        return {"top1_accuracy": 0.0, "top5_accuracy": 0.0,
                "mean_log_prob": float("nan"), "n_instances": 0}

    return {
        "top1_accuracy": round(top1_hits / n_instances, 4),
        "top5_accuracy": round(top5_hits / n_instances, 4),
        "mean_log_prob": round(sum(log_probs) / len(log_probs), 4) if log_probs else float("nan"),
        "n_instances": n_instances,
    }


def collect_qualitative_examples(test_processed, ngram_counts_list, vocabulary,
                                  n_examples: int = MAX_QUAL_EXAMPLES):
    """Collect qualitative good / bad prediction examples.

    A *good* example is one where the model's top-1 prediction matches the
    true next word.  A *bad* example is one where it does not.

    Parameters
    ----------
    test_processed:
        OOV-replaced test sentences.
    ngram_counts_list:
        Ordered list of n-gram count dicts.
    vocabulary:
        Closed-vocabulary list.
    n_examples:
        How many good / bad examples to collect (each).

    Returns
    -------
    dict with keys 'good' and 'bad', each containing a list of dicts:
        {prefix, true_word, predicted_word, probability}
    """
    n_order = len(ngram_counts_list) - 1
    ngram_counts = ngram_counts_list[n_order - 1]
    n_plus1_counts = ngram_counts_list[n_order]

    good_examples = []
    bad_examples = []

    for sent in test_processed:
        if len(sent) < 3:
            continue
        mid = len(sent) // 2
        prefix = sent[:mid]
        true_word = sent[mid]
        try:
            top5 = _top5_suggestions(prefix, ngram_counts, n_plus1_counts, vocabulary)
        except Exception:
            continue
        if not top5:
            continue
        pred_word, prob = top5[0]
        entry = {
            "prefix": " ".join(prefix[-4:]),  # show last 4 tokens for brevity
            "true_word": true_word,
            "predicted_word": pred_word,
            "probability": round(prob, 6),
        }
        if pred_word == true_word and len(good_examples) < n_examples:
            good_examples.append(entry)
        elif pred_word != true_word and len(bad_examples) < n_examples:
            bad_examples.append(entry)
        if len(good_examples) >= n_examples and len(bad_examples) >= n_examples:
            break

    return {"good": good_examples, "bad": bad_examples}


# ---------------------------------------------------------------------------
# Report generation helpers
# ---------------------------------------------------------------------------


def _ngram_label(n: int) -> str:
    labels = {1: "Unigram", 2: "Bigram", 3: "Trigram", 4: "4-gram"}
    return labels.get(n, f"{n}-gram")


def _format_ngram(tup) -> str:
    return " ".join(tup)


def _build_top_ngram_table(top_ngrams, n: int) -> str:
    label = _ngram_label(n)
    lines = [f"**Top-{TOP_K} {label}s**", "", "| Rank | N-gram | Count |",
             "|------|--------|-------|"]
    for rank, (ngram, count) in enumerate(top_ngrams, 1):
        lines.append(f"| {rank} | `{_format_ngram(ngram)}` | {count} |")
    return "\n".join(lines)


def _perplexity_table(perp_stats: dict) -> str:
    lines = ["| N-gram Order | Mean Perplexity |", "|--------------|-----------------|"]
    for n, perp in sorted(perp_stats.items()):
        val = f"{perp:.2f}" if math.isfinite(perp) else "N/A"
        lines.append(f"| {_ngram_label(n)} | {val} |")
    return "\n".join(lines)


def _qual_table(examples: list) -> str:
    if not examples:
        return "_No examples found._"
    lines = ["| Prefix (last 4 tokens) | True word | Predicted | Probability |",
             "|------------------------|-----------|-----------|-------------|"]
    for ex in examples:
        lines.append(
            f"| `{ex['prefix']}` | **{ex['true_word']}** | {ex['predicted_word']} | {ex['probability']:.6f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------


def save_artifacts(all_metrics: list):
    """Save per-dataset metrics to CSV and JSON.

    Parameters
    ----------
    all_metrics:
        List of flat metric dicts, one per dataset.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Flatten perplexity dicts into top-level keys
    rows = []
    for m in all_metrics:
        row = {k: v for k, v in m.items() if k != "perplexity_by_order"}
        perp = m.get("perplexity_by_order", {})
        for n, val in perp.items():
            row[f"perplexity_{_ngram_label(n)}"] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(ARTIFACTS_DIR, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    json_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2, default=str)
    print(f"  Saved {json_path}")


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

_REPORT_HEADER = """\
# N-gram Autocomplete Language Model — Multi-Dataset Analysis Report

**Generated:** {date}

---

## Project Overview

This project implements an **N-gram language model** for word-autocomplete.  
The pipeline (defined in `data_preprocessing.py` and `language_model.py`):

1. **Tokenisation** — sentences are split on newlines, lowercased, and tokenised
   with NLTK's `word_tokenize`.
2. **Vocabulary building** — a *closed vocabulary* is derived from training data
   by keeping only words that appear at least `minimum_freq` (default: 2) times.
   All other words are replaced by `<unk>`.
3. **N-gram counting** — `count_n_grams(data, n)` prepends `n` start tokens
   `<s>` and one end token `<e>` to every sentence, then slides a window of
   size `n` to accumulate counts.
4. **Probability estimation** — add-*k* (Laplace) smoothing with default `k=1`:

   ```
   P(w | context) = (C(context, w) + k) / (C(context) + k * |V|)
   ```

   where `|V|` includes `<e>` and `<unk>`.
5. **Word suggestion** — the word with the highest smoothed probability given
   the last *n−1* tokens is returned as the top-1 suggestion.
6. **Perplexity** — geometric-mean inverse probability over a held-out sentence:

   ```
   PP(W) = [ prod_t 1/P(w_t | context_t) ]^(1/N)
   ```

No back-off or interpolation is applied; each n-gram order is a standalone
model.  The analysis runner (this script) uses the **{max_n}-gram model**
for autocomplete evaluation (a ({max_n}+1)-gram count table is also built
to allow perplexity computation for the {max_n}-gram model).

---

## How to Run

### Install dependencies
```bash
pip install nltk numpy pandas matplotlib
```

### Run on default datasets (Twitter, Shakespeare/Merchant, Disney, OnCampus)
```bash
python -m analysis.run
```

### Run on custom datasets (name:path pairs)
```bash
python -m analysis.run \\
    --dataset Twitter:data/en_US.twitter.txt \\
    --dataset Disney:data/disney.txt
```

### Run on an external dataset
```bash
python -m analysis.run --dataset MyCorpus:/path/to/corpus.txt
```

**Expected format:** plain-text file, one sentence per line, UTF-8 encoding.

Artefacts are written to:
- `reports/artifacts/metrics.csv`
- `reports/artifacts/metrics.json`

---

## Experimental Settings

| Parameter | Value |
|-----------|-------|
| Random seed | {seed} |
| Train / Valid / Test split | {train_pct}% / {valid_pct}% / {test_pct}% |
| Minimum word frequency (vocabulary) | {min_freq} |
| Add-*k* smoothing constant | {k_smooth} |
| N-gram orders evaluated | 1 – {max_n} |
| Max sentences for perplexity / autocomplete eval | {max_eval} |

---
"""

_DATASET_SECTION = """\
## Dataset: {name}

**File:** `{path}`

### Size & Vocabulary

| Metric | Value |
|--------|-------|
| Training sentences | {n_docs_train:,} |
| Validation sentences | {n_docs_valid:,} |
| Test sentences | {n_docs_test:,} |
| Training tokens | {n_tokens_train:,} |
| Test tokens | {n_tokens_test:,} |
| Vocabulary size | {vocab_size:,} |
| OOV rate (test) | {oov_rate_test:.2%} |

### N-gram Type Counts (Sparsity)

| N-gram Order | Unique Types |
|--------------|-------------|
{ngram_type_rows}

### Top N-grams

{top_unigram_table}

{top_bigram_table}

{top_trigram_table}

### Perplexity on Test Split ({max_eval} sentences max)

{perplexity_table}

### Autocomplete Accuracy (using {max_n}-gram model, {max_eval} sentences max)

| Metric | Value |
|--------|-------|
| Top-1 accuracy | {top1_accuracy:.2%} |
| Top-5 accuracy | {top5_accuracy:.2%} |
| Mean log-probability of true word | {mean_log_prob:.4f} |
| Number of prediction instances | {n_instances:,} |

### Qualitative Examples

**Good predictions** (model top-1 = true next word):

{good_table}

**Bad predictions** (model top-1 ≠ true next word):

{bad_table}

---
"""

_COMPARISON_SECTION = """\
## Cross-Dataset Comparison

### Summary Table

| Dataset | Vocab | OOV% | Unigram PP | Bigram PP | Trigram PP | {max_n}-gram PP | Top-1 Acc | Top-5 Acc |
|---------|-------|------|------------|-----------|------------|-----------------|-----------|-----------|
{comparison_rows}

### Discussion

**Why results differ across datasets:**

- **Vocabulary size and domain** — Larger, more diverse corpora (e.g., Twitter)
  produce larger vocabularies and higher perplexity because the model must
  spread probability mass over many more words.
- **OOV rate** — Corpora with richer vocabulary relative to the training set
  size will have higher OOV rates on the test split, which raises perplexity
  because unseen words receive only the smoothed floor probability.
- **Sentence length and structure** — Short, informal sentences (Twitter) have
  fewer high-order n-gram matches, limiting the benefit of higher-order models.
  Structured narrative text (Shakespeare) shows more consistent phrasing,
  allowing higher-order n-grams to be more informative.
- **Data sparsity** — Small corpora (Disney) have very few training sentences,
  leading to sparse n-gram tables.  With add-k smoothing this flattens
  probability distributions and can raise perplexity.
- **Top-1 / Top-5 accuracy** — Datasets with highly repetitive phrasing will
  yield higher autocomplete accuracy because the most-frequent next word is
  often the true next word.

**Limitations:**

- Only add-k (Laplace) smoothing is used; no back-off or Kneser-Ney smoothing.
- No neural or sub-word models are evaluated.
- Perplexity is computed per-token with start/end markers, not per-word, so
  values are not directly comparable to standard benchmarks.
- The autocomplete evaluation uses the closed vocabulary, so OOV test words can
  never be the predicted word even if the model assigns them probability via
  the `<unk>` token.

**Next steps:**

1. Implement Kneser-Ney or Witten-Bell smoothing for better generalisation.
2. Add interpolation across n-gram orders.
3. Evaluate on held-out splits from the *same* domain used for training.
4. Explore neural language models (LSTM, Transformer) as baselines.
5. Add a web-based demo interface.
"""


def generate_report(dataset_results: list) -> str:
    """Build the full Markdown report string.

    Parameters
    ----------
    dataset_results:
        List of per-dataset result dicts produced by ``analyse_dataset``.

    Returns
    -------
    str — complete Markdown report.
    """
    import datetime
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    test_pct = round((1 - TRAIN_FRAC - VALID_FRAC) * 100)

    header = _REPORT_HEADER.format(
        date=date_str,
        seed=RANDOM_SEED,
        train_pct=int(TRAIN_FRAC * 100),
        valid_pct=int(VALID_FRAC * 100),
        test_pct=test_pct,
        min_freq=MINIMUM_FREQ,
        k_smooth=K_SMOOTH,
        max_n=MAX_N,
        max_eval=MAX_EVAL_SENTS,
    )

    sections = [header]

    for res in dataset_results:
        # N-gram type rows
        ngram_type_rows = "\n".join(
            f"| {_ngram_label(n)} | {count:,} |"
            for n, count in sorted(res["model_stats"].items())
        )

        section = _DATASET_SECTION.format(
            name=res["name"],
            path=res["path"],
            n_docs_train=res["dataset_stats"]["n_docs_train"],
            n_docs_valid=res["dataset_stats"]["n_docs_valid"],
            n_docs_test=res["dataset_stats"]["n_docs_test"],
            n_tokens_train=res["dataset_stats"]["n_tokens_train"],
            n_tokens_test=res["dataset_stats"]["n_tokens_test"],
            vocab_size=res["dataset_stats"]["vocab_size"],
            oov_rate_test=res["dataset_stats"]["oov_rate_test"],
            ngram_type_rows=ngram_type_rows,
            top_unigram_table=_build_top_ngram_table(res["top_unigrams"], 1),
            top_bigram_table=_build_top_ngram_table(res["top_bigrams"], 2),
            top_trigram_table=_build_top_ngram_table(res["top_trigrams"], 3),
            perplexity_table=_perplexity_table(res["perplexity_by_order"]),
            max_eval=MAX_EVAL_SENTS,
            max_n=MAX_N,
            top1_accuracy=res["autocomplete"]["top1_accuracy"],
            top5_accuracy=res["autocomplete"]["top5_accuracy"],
            mean_log_prob=res["autocomplete"]["mean_log_prob"],
            n_instances=res["autocomplete"]["n_instances"],
            good_table=_qual_table(res["qualitative"]["good"]),
            bad_table=_qual_table(res["qualitative"]["bad"]),
        )
        sections.append(section)

    # Comparison table rows
    comparison_rows = []
    for res in dataset_results:
        perp = res["perplexity_by_order"]
        row = (
            f"| {res['name']} "
            f"| {res['dataset_stats']['vocab_size']:,} "
            f"| {res['dataset_stats']['oov_rate_test']:.1%} "
            f"| {perp.get(1, float('nan')):.1f} "
            f"| {perp.get(2, float('nan')):.1f} "
            f"| {perp.get(3, float('nan')):.1f} "
            f"| {perp.get(MAX_N, float('nan')):.1f} "
            f"| {res['autocomplete']['top1_accuracy']:.2%} "
            f"| {res['autocomplete']['top5_accuracy']:.2%} |"
        )
        comparison_rows.append(row)

    comparison = _COMPARISON_SECTION.format(
        max_n=MAX_N,
        comparison_rows="\n".join(comparison_rows),
    )
    sections.append(comparison)

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def analyse_dataset(name: str, path: str) -> dict:
    """Run the full analysis pipeline for one dataset.

    Parameters
    ----------
    name:
        Human-readable dataset name.
    path:
        Path to the dataset file.

    Returns
    -------
    dict containing all computed metrics and examples.
    """
    print(f"\n[{name}] Loading and splitting data …")
    train_raw, valid_raw, test_raw = load_and_split(path)

    print(f"[{name}] Preprocessing …")
    train_processed, test_processed, vocabulary = preprocess_data(
        train_raw, test_raw, MINIMUM_FREQ
    )
    # Also preprocess valid split (using same vocabulary) for completeness
    from data_preprocessing import replace_oov_words_by_unk
    valid_processed = replace_oov_words_by_unk(valid_raw, vocabulary)

    print(f"[{name}] Computing dataset statistics …")
    dataset_stats = compute_dataset_stats(train_raw, valid_raw, test_raw, vocabulary)

    print(f"[{name}] Building n-gram models (1 … {MAX_N}) …")
    ngram_counts_list = [count_n_grams(train_processed, n) for n in range(1, MAX_N + 2)]
    # MAX_N+2 so we have (MAX_N+1)-gram counts for perplexity of MAX_N-gram model

    print(f"[{name}] Computing top n-grams …")
    top_unigrams = compute_top_ngrams(train_processed, 1)
    top_bigrams = compute_top_ngrams(train_processed, 2)
    top_trigrams = compute_top_ngrams(train_processed, 3)

    model_stats = compute_model_stats(ngram_counts_list)

    print(f"[{name}] Computing perplexity on test split …")
    vocab_size = len(vocabulary) + 2  # +<e> +<unk>
    perplexity_stats = compute_perplexity_stats(
        test_processed, ngram_counts_list, vocab_size
    )

    print(f"[{name}] Computing autocomplete accuracy …")
    # Use n-gram orders 1…MAX_N for autocomplete (need MAX_N+1 counts available)
    autocomplete_counts = ngram_counts_list[: MAX_N + 1]
    autocomplete_stats = compute_autocomplete_accuracy(
        test_processed, autocomplete_counts, vocabulary
    )

    print(f"[{name}] Collecting qualitative examples …")
    qual_examples = collect_qualitative_examples(
        test_processed, autocomplete_counts, vocabulary
    )

    return {
        "name": name,
        "path": path,
        "dataset_stats": dataset_stats,
        "model_stats": {k: v for k, v in model_stats.items() if k <= MAX_N + 1},
        "top_unigrams": top_unigrams,
        "top_bigrams": top_bigrams,
        "top_trigrams": top_trigrams,
        "perplexity_by_order": perplexity_stats,
        "autocomplete": autocomplete_stats,
        "qualitative": qual_examples,
        # flat metrics for CSV/JSON
        **dataset_stats,
        "top1_accuracy": autocomplete_stats["top1_accuracy"],
        "top5_accuracy": autocomplete_stats["top5_accuracy"],
        "mean_log_prob": autocomplete_stats["mean_log_prob"],
        "n_instances": autocomplete_stats["n_instances"],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-dataset analysis runner for the autocomplete n-gram model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        metavar="NAME:PATH",
        action="append",
        dest="datasets",
        help=(
            "Dataset to analyse in NAME:PATH format. "
            "Can be specified multiple times. "
            "If omitted, the four built-in datasets are used."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build dataset dict
    if args.datasets:
        datasets = {}
        for spec in args.datasets:
            if ":" not in spec:
                print(f"ERROR: --dataset argument must be in NAME:PATH format, got: {spec!r}")
                sys.exit(1)
            name, path = spec.split(":", 1)
            datasets[name] = path
    else:
        datasets = DEFAULT_DATASETS

    # Resolve paths relative to repo root
    os.chdir(_REPO_ROOT)

    # Validate paths
    missing = [p for p in datasets.values() if not os.path.isfile(p)]
    if missing:
        print("ERROR: The following dataset files were not found:")
        for p in missing:
            print(f"  {p}")
        print("Please check the paths and try again.")
        sys.exit(1)

    print("=" * 60)
    print("N-gram Autocomplete — Multi-Dataset Analysis Runner")
    print("=" * 60)
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Output dir: {REPORT_DIR}/")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_results = []
    for name, path in datasets.items():
        result = analyse_dataset(name, path)
        all_results.append(result)

    print("\nSaving artifacts …")
    # Build flat metrics list for CSV/JSON (exclude large objects)
    flat_metrics = []
    for res in all_results:
        m = {
            "dataset": res["name"],
            "path": res["path"],
            **res["dataset_stats"],
            **{f"ngram_types_{_ngram_label(n)}": v
               for n, v in res["model_stats"].items()},
            **{f"perplexity_{_ngram_label(n)}": v
               for n, v in res["perplexity_by_order"].items()},
            "top1_accuracy": res["autocomplete"]["top1_accuracy"],
            "top5_accuracy": res["autocomplete"]["top5_accuracy"],
            "mean_log_prob": res["autocomplete"]["mean_log_prob"],
            "n_instances": res["autocomplete"]["n_instances"],
        }
        flat_metrics.append(m)

    save_artifacts(flat_metrics)

    print("Generating report …")
    report_md = generate_report(all_results)
    report_path = os.path.join(REPORT_DIR, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_md)
    print(f"  Saved {report_path}")

    print("\nDone!")
    print(f"  Report : {report_path}")
    print(f"  Metrics: {ARTIFACTS_DIR}/metrics.csv")
    print(f"           {ARTIFACTS_DIR}/metrics.json")


if __name__ == "__main__":
    main()
