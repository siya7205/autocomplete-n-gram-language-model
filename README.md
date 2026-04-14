# Autocomplete N-gram Language Model

A sentiment-aware next-word autocomplete system built on N-gram language models (orders 1–4) with Laplace smoothing.  
The pipeline covers corpus ingestion → N-gram training → weak-label sentiment annotation → classifier training → sentiment-aware reranking → evaluation sweep → optional Streamlit demo.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Quick-Start — 5 Steps](#5-quick-start--5-steps)
6. [Step-by-Step How to Run](#6-step-by-step-how-to-run)
   - [Step 0: Baseline interactive demo](#step-0-baseline-interactive-demo)
   - [Step 1: Predict next words from the CLI](#step-1-predict-next-words-from-the-cli)
   - [Step 2: Generate a sentiment-labeling worksheet](#step-2-generate-a-sentiment-labeling-worksheet)
   - [Step 3: Train a sentiment classifier](#step-3-train-a-sentiment-classifier)
   - [Step 3b: Weak-label alternative (no manual labeling)](#step-3b-weak-label-alternative-no-manual-labeling)
   - [Step 4: Sentiment-aware autocomplete reranking](#step-4-sentiment-aware-autocomplete-reranking)
   - [Step 5: Run the full evaluation sweep](#step-5-run-the-full-evaluation-sweep)
   - [Step 6: Generate the final report artifacts](#step-6-generate-the-final-report-artifacts)
   - [Step 7 (optional): Launch the Streamlit demo UI](#step-7-optional-launch-the-streamlit-demo-ui)
7. [CLI Reference](#7-cli-reference)
8. [Output Files](#8-output-files)
9. [Results Summary](#9-results-summary)
10. [Background — How the Model Works](#10-background--how-the-model-works)
11. [References](#11-references)

---

## 1  Project Overview

| Capability | Details |
|---|---|
| **Language model** | N-gram (1-gram through 4-gram) with Laplace smoothing |
| **Corpora** | Twitter, Shakespeare, Disney, on-campus, Merchant of Venice (all in `data/`) |
| **Sentiment** | Logistic regression on TF-IDF features; trained on manually- or weakly-labeled data |
| **Reranking** | `final_score = lm_score + λ × P_sentiment(target \| context)` |
| **Evaluation** | Top-K hit rate + sentiment alignment, swept over `top_k ∈ {1,3,5}` and `λ ∈ {0.0,0.5,1.0,2.0}` |
| **Demo** | Streamlit web app (`app.py`) |

---

## 2  Repository Layout

```
.
├── autocomplete/               # Core Python package
│   ├── datasets.py             # Corpus loading helpers
│   ├── evaluate.py             # Phase 4 evaluator (top-k hit rate + sentiment alignment)
│   ├── generate_labeling_csv.py# Worksheet CSV generator for manual labeling
│   ├── predict.py              # predict CLI (baseline + sentiment reranking)
│   ├── preprocess.py           # Tokenization and text normalization
│   ├── sentiment.py            # Sentiment model training and inference
│   ├── sentiment_predict.py    # Sentiment prediction CLI
│   └── train_sentiment.py      # Sentiment training CLI
├── scripts/
│   ├── weak_label_sentiment.py # Auto-label corpus with keyword lexicon
│   └── run_final_report_results.py  # One-command sweep + report generator
├── analysis/
│   └── run.py                  # Cross-corpus analysis and perplexity plots
├── data/                       # Plain-text corpora + generated CSVs
├── models/                     # Saved sentiment model artifacts (.pkl)
├── results/                    # Generated metrics, plots, and reports
├── app.py                      # Streamlit demo UI
├── language_model.py           # Core N-gram model (get_suggestions, perplexity)
├── data_preprocessing.py       # Vocabulary building, <unk> replacement
├── main.py                     # Baseline interactive demo (Twitter corpus)
├── main_gram.py                # Baseline demo (Disney corpus)
├── main_per.py                 # Perplexity demo
├── main_multi_dataset.py       # Multi-corpus comparison demo
└── requirements.txt
```

---

## 3  Prerequisites

| Requirement | Version |
|---|---|
| Python | **3.10 or later** (uses PEP 604 union types) |
| pip | any recent version |

All Python dependencies are listed in `requirements.txt`:

```
nltk
numpy
pandas
scikit-learn
joblib
matplotlib
streamlit
```

---

## 4  Installation

```bash
# 1. Clone the repo
git clone https://github.com/siya7205/autocomplete-n-gram-language-model.git
cd autocomplete-n-gram-language-model

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5  Quick-Start — 5 Steps

If you just want to go from zero to a working, evaluated system:

```bash
# 1. Install
pip install -r requirements.txt

# 2. Predict next words (baseline, no sentiment)
python -m autocomplete.predict --text "to be or not" --top-k 5

# 3. Auto-generate weak sentiment labels from Shakespeare corpus
python scripts/weak_label_sentiment.py \
  --corpus data/Shakespeare.txt \
  --out data/sentiment_labeled_weak.csv \
  --max-rows 1000 --seed 42

# 4. Train the sentiment classifier
python -m autocomplete.train_sentiment \
  --csv data/sentiment_labeled_weak.csv \
  --out models/sentiment_weak.pkl \
  --seed 42

# 5. Run the full evaluation sweep and produce all report artifacts
python scripts/run_final_report_results.py \
  --corpus data/Shakespeare.txt \
  --sentiment-csv data/sentiment_labeled_weak.csv \
  --model models/sentiment_weak.pkl \
  --outdir results/final \
  --seed 42 \
  --max-examples 500
```

All output lands in `results/final/`.

---

## 6  Step-by-Step How to Run

### Step 0: Baseline interactive demo

Run an in-memory N-gram model on a corpus and type prefixes interactively:

```bash
python main.py                   # Twitter corpus
python main_gram.py              # Disney corpus
python main_per.py               # Perplexity demo
python main_multi_dataset.py     # Compare all corpora side-by-side
```

---

### Step 1: Predict next words from the CLI

```bash
python -m autocomplete.predict --text "I want to" --top-k 5
```

Use a different corpus:

```bash
python -m autocomplete.predict --text "I want to" --top-k 5 --data data/disney.txt
```

**Example output:**

```
Input: "I want to"
Top 5 suggestions:
1. be      0.0312
2. go      0.0287
3. see     0.0241
4. have    0.0198
5. get     0.0175
```

---

### Step 2: Generate a sentiment-labeling worksheet

Creates a CSV with sampled sentences for manual labeling:

```bash
python -m autocomplete.generate_labeling_csv --sample-size 300
# Output: data/sentiment_labeling_worksheet.csv
```

Custom options:

```bash
python -m autocomplete.generate_labeling_csv \
  --data data/en_US.twitter.txt \
  --output data/sentiment_labeling_worksheet.csv \
  --sample-size 300 \
  --seed 87 \
  --min-tokens 3
```

Open the CSV and fill the `sentiment_label` column with `positive`, `negative`, or `neutral` for each row.

---

### Step 3: Train a sentiment classifier

After manually labeling the worksheet:

```bash
python -m autocomplete.train_sentiment \
  --csv data/sentiment_labeled.csv \
  --out models/sentiment.pkl \
  --seed 42
```

Prints: accuracy, weighted precision / recall / F1, and a confusion matrix.

Test a single prediction:

```bash
python -m autocomplete.sentiment_predict \
  --text "I love this beautiful day" \
  --model models/sentiment.pkl
```

---

### Step 3b: Weak-label alternative (no manual labeling)

Skip manual labeling entirely — auto-assign labels from a keyword lexicon:

```bash
python scripts/weak_label_sentiment.py \
  --corpus data/Shakespeare.txt \
  --out data/sentiment_labeled_weak.csv \
  --max-rows 1000 \
  --seed 42
```

Or apply weak labels to an existing worksheet:

```bash
python scripts/weak_label_sentiment.py \
  --worksheet data/sentiment_labeling_worksheet.csv \
  --out data/sentiment_labeled_weak.csv \
  --seed 42
```

Then train from the weak-labeled file:

```bash
python -m autocomplete.train_sentiment \
  --csv data/sentiment_labeled_weak.csv \
  --out models/sentiment_weak.pkl \
  --seed 42
```

> **Note:** Weak labels are heuristic (keyword-based). Metrics from a weak-labeled model are indicative only, not ground-truth quality.

---

### Step 4: Sentiment-aware autocomplete reranking

```bash
python -m autocomplete.predict \
  --text "I feel" \
  --top-k 5 \
  --sentiment positive \
  --sentiment-model models/sentiment_weak.pkl
```

Supported `--sentiment` values: `off` (default), `positive`, `negative`, `neutral`.

Tune the reranking strength with `--sentiment-weight` (alias `--lambda`):

```bash
python -m autocomplete.predict \
  --text "I want to" \
  --top-k 5 \
  --sentiment negative \
  --sentiment-weight 0.5
```

- Higher `--sentiment-weight` → stronger sentiment bias.
- `--sentiment off` → pure language-model order (no reranking).
- If the model file is missing, the command exits with an actionable error.

---

### Step 5: Run the full evaluation sweep

Computes **top-K hit rate** and **sentiment alignment** on held-out test examples:

```bash
python -m autocomplete.evaluate \
  --corpus data/Shakespeare.txt \
  --sentiment-csv data/sentiment_labeled_weak.csv \
  --top-k 5 \
  --seed 42 \
  --out results/metrics.json
```

Cap runtime with `--max-examples`:

```bash
python -m autocomplete.evaluate \
  --corpus data/Shakespeare.txt \
  --sentiment-csv data/sentiment_labeled_weak.csv \
  --top-k 5 \
  --seed 42 \
  --max-examples 300 \
  --out results/metrics.json
```

`results/` is created automatically if it does not exist. The JSON output includes run config, top-k hit-rate stats, per-sentiment alignment stats, and a timestamp.

---

### Step 6: Generate the final report artifacts

One command runs a 12-configuration sweep (`top_k × sentiment_weight`) and writes tables, plots, and a markdown snippet:

```bash
python scripts/run_final_report_results.py \
  --corpus data/Shakespeare.txt \
  --sentiment-csv data/sentiment_labeled_weak.csv \
  --model models/sentiment_weak.pkl \
  --outdir results/final \
  --seed 42 \
  --max-examples 500
```

Add `--run-weak-labeling` to regenerate weak labels in the same command:

```bash
python scripts/run_final_report_results.py \
  --corpus data/Shakespeare.txt \
  --sentiment-csv data/sentiment_labeled_weak.csv \
  --model models/sentiment_weak.pkl \
  --run-weak-labeling \
  --outdir results/final \
  --seed 42 \
  --max-examples 500
```

**Outputs in `results/final/`:**

| File | Contents |
|---|---|
| `summary.csv` | 12-row sweep table (CSV) |
| `summary.json` | Same table in JSON |
| `topk_hit_rate.png` | Hit rate vs `top_k` chart |
| `sentiment_alignment.png` | Alignment vs `sentiment_weight` chart |
| `REPORT_SNIPPET.md` | Auto-generated markdown summary |

A full paper-style report is at `results/final_final_report.md`.

---

### Step 7 (optional): Launch the Streamlit demo UI

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

| Control | Description |
|---|---|
| **Prefix text** | Text prefix to autocomplete |
| **Top-k suggestions** | Number of suggestions to display |
| **Target sentiment** | `off`, `positive`, `negative`, or `neutral` |
| **Sentiment weight** | Reranking strength (active when sentiment ≠ `off`) |

If the sentiment model file is missing the app shows a friendly message with the exact command to train it.

---

## 7  CLI Reference

| Command | Purpose |
|---|---|
| `python main.py` | Interactive baseline (Twitter) |
| `python main_gram.py` | Interactive baseline (Disney) |
| `python main_per.py` | Perplexity demo |
| `python main_multi_dataset.py` | Multi-corpus comparison |
| `python -m autocomplete.predict --text "..." --top-k N` | Predict next N words |
| `python -m autocomplete.generate_labeling_csv --sample-size N` | Generate labeling worksheet |
| `python -m autocomplete.train_sentiment --csv FILE --out MODEL` | Train sentiment classifier |
| `python -m autocomplete.sentiment_predict --text "..." --model MODEL` | Predict sentiment of text |
| `python -m autocomplete.evaluate --corpus FILE --sentiment-csv FILE --out FILE` | Evaluate hit rate + alignment |
| `python scripts/weak_label_sentiment.py --corpus FILE --out FILE` | Auto-generate weak sentiment labels |
| `python scripts/run_final_report_results.py ...` | Full sweep + report artifacts |
| `streamlit run app.py` | Launch demo UI |
| `python -m analysis.run` | Cross-corpus analysis plots |

---

## 8  Output Files

| Path | Created by | Contents |
|---|---|---|
| `models/sentiment.pkl` | `train_sentiment` CLI | Trained sentiment pipeline |
| `models/sentiment_weak.pkl` | `train_sentiment` CLI | Sentiment pipeline from weak labels |
| `data/sentiment_labeling_worksheet.csv` | `generate_labeling_csv` | Raw sentences to hand-label |
| `data/sentiment_labeled_weak.csv` | `weak_label_sentiment.py` | Auto-labeled sentiment CSV |
| `results/metrics.json` | `evaluate` CLI | Top-k hit rate + alignment JSON |
| `results/final/summary.csv` | `run_final_report_results.py` | Full sweep table |
| `results/final/summary.json` | `run_final_report_results.py` | Sweep table (JSON) |
| `results/final/topk_hit_rate.png` | `run_final_report_results.py` | Hit-rate chart |
| `results/final/sentiment_alignment.png` | `run_final_report_results.py` | Alignment chart |
| `results/final/REPORT_SNIPPET.md` | `run_final_report_results.py` | Short markdown summary |
| `results/final_final_report.md` | (pre-generated) | Full paper-style evaluation report |

---

## 9  Results Summary

Results from the Shakespeare corpus sweep (80 held-out examples per configuration, seed 42):

| top_k | sentiment_weight | top-k hit rate | positive alignment | negative alignment |
|---:|---:|---:|---:|---:|
| 1 | 0.0 – 2.0 | 0.1750 | 0.9500 | 0.0500 |
| 3 | 0.0 – 2.0 | **0.2500** | **0.9625** | 0.0375 |
| 5 | 0.0 – 2.0 | **0.2500** | **0.9625** | 0.0375 |

Key findings:
- Expanding from `top_k=1` to `top_k=3` improves hit rate by **+4.3 pp** (+24.6 % relative).
- `top_k=5` gives no additional gain over `top_k=3` on this corpus.
- `sentiment_weight` (0.0 → 2.0) has **no measurable effect** on hit rate or alignment — caused by majority-class bias in the weakly-labeled classifier.

See `results/final_final_report.md` for the full analysis, including Discussion and Threats to Validity sections.

---

## 10  Background — How the Model Works

### N-gram language model

An N-gram is a contiguous sequence of N words. The model approximates the probability of the next word by looking at only the N−1 preceding words:

```
P(w_n | w_1 … w_{n-1}) ≈ P(w_n | w_{n-N+1} … w_{n-1})
```

Orders 1–4 are trained and consulted together; `language_model.get_suggestions()` returns one scored candidate per adjacent-order pair.

### Laplace (additive) smoothing

Adds a constant *k* to every count to avoid zero probabilities for unseen N-grams:

```
P(w_n | context) = (count(context, w_n) + k) / (count(context) + k × |V|)
```

### Sentiment reranking

```
final_score(w) = lm_score(w) + λ × P_sentiment(target | prefix + w)
```

where `λ` is `--sentiment-weight`. Candidates are re-sorted by `final_score` descending.

---

## 11  References

1. Jurafsky, D. & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/
2. Chen, S. F. & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling. *Computer Speech & Language*, 13(4), 359–394.
