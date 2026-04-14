# Runbook — Generate `results/final/` Artifacts

Follow these steps **in order** from a clean checkout to produce the five final report files:

```
results/final/summary.csv
results/final/summary.json
results/final/topk_hit_rate.png
results/final/sentiment_alignment.png
results/final/REPORT_SNIPPET.md
```

---

## Step 0 — Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 + |
| Git | any |

All Python packages are pinned in `requirements.txt`.

---

## Step 1 — Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 2 — Verify corpus data

The corpus files are already committed under `data/`. Confirm the one you plan to use is present:

```bash
ls data/
# en_US.twitter.txt   ← recommended (largest, best accuracy)
# disney.txt
# merchant.txt
# oncampus_no_numbers.txt
```

The default corpus used in all examples below is **`data/en_US.twitter.txt`**.  
Swap in any other file with `--corpus data/<file>.txt` if you prefer.

---

## Step 3 — Generate the sentiment labeling worksheet

This creates a CSV with sampled sentences that you will manually label:

```bash
python -m autocomplete.generate_labeling_csv \
  --data data/en_US.twitter.txt \
  --output data/sentiment_labeling_worksheet.csv \
  --sample-size 300 \
  --seed 42
```

A file `data/sentiment_labeling_worksheet.csv` is created with columns:
- `text` — raw sentence from the corpus
- `sentiment_label` — **empty, you fill this in**

---

## Step 4 — Label the worksheet (manual step)

Open `data/sentiment_labeling_worksheet.csv` in Excel, Google Sheets, or any text editor.

Fill the `sentiment_label` column for every row using **exactly** one of:

| Value | Meaning |
|---|---|
| `positive` | Clearly positive / upbeat tone |
| `negative` | Clearly negative / critical tone |
| `neutral` | No strong sentiment |

Tips:
- You do **not** need all three classes — `positive` + `negative` is sufficient.
- Aim for at least **200 labeled rows** total; ~300 is recommended.
- Empty or blank rows are skipped automatically by the trainer.
- Save as plain CSV (UTF-8).

When done, **save the file as `data/sentiment_labeled.csv`** (rename or save-as from the worksheet).

---

## Step 5 — Train the sentiment classifier

```bash
mkdir -p models

python -m autocomplete.train_sentiment \
  --csv  data/sentiment_labeled.csv \
  --out  models/sentiment.pkl \
  --seed 42
```

Expected output (numbers will vary):

```
Accuracy : 0.87
Precision: 0.86
Recall   : 0.87
F1       : 0.86
Saved → models/sentiment.pkl
```

> **Troubleshooting:** If accuracy is very low (< 0.60), check that your labels are spelled correctly (`positive` / `negative` / `neutral`) and that you have at least ~50 examples per class.

Quick sanity check:

```bash
python -m autocomplete.sentiment_predict \
  --text "I absolutely love this!" \
  --model models/sentiment.pkl
# Expected: positive
```

---

## Step 6 — Run the final report script

```bash
python scripts/run_final_report_results.py \
  --corpus       data/en_US.twitter.txt \
  --sentiment-csv data/sentiment_labeled.csv \
  --model        models/sentiment.pkl \
  --outdir       results/final \
  --seed         42 \
  --max-examples 500
```

This sweeps over `top_k ∈ {1, 3, 5}` and `sentiment_weight ∈ {0.0, 0.5, 1.0, 2.0}` and writes:

| File | Description |
|---|---|
| `results/final/summary.csv` | Full sweep table (CSV) |
| `results/final/summary.json` | Same table as JSON |
| `results/final/topk_hit_rate.png` | Hit-rate vs top_k plot |
| `results/final/sentiment_alignment.png` | Alignment rate vs sentiment weight plot |
| `results/final/REPORT_SNIPPET.md` | Report-ready markdown with best settings |

Runtime is roughly **1–5 minutes** depending on corpus size and `--max-examples`.

---

## Optional flags

| Flag | Default | Effect |
|---|---|---|
| `--max-examples N` | 500 | Fewer examples → faster run, less precise metrics |
| `--train-fraction F` | 0.8 | Fraction of corpus used for LM training |
| `--minimum-freq N` | 2 | Min token frequency kept in vocabulary |
| `--top-k-values 1 3 5 10` | `1 3 5` | Custom top-k sweep values |
| `--sentiment-weight-values 0 1 2` | `0.0 0.5 1.0 2.0` | Custom sentiment-weight sweep values |

Example with all overrides:

```bash
python scripts/run_final_report_results.py \
  --corpus        data/en_US.twitter.txt \
  --sentiment-csv data/sentiment_labeled.csv \
  --model         models/sentiment.pkl \
  --outdir        results/final \
  --seed          42 \
  --max-examples  1000 \
  --top-k-values  1 3 5 10 \
  --sentiment-weight-values 0.0 0.5 1.0 2.0
```

---

## Quick reference — full command sequence

```bash
# 1  Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2  Generate labeling worksheet
python -m autocomplete.generate_labeling_csv \
  --data data/en_US.twitter.txt \
  --output data/sentiment_labeling_worksheet.csv \
  --sample-size 300 --seed 42

# 3  << MANUALLY LABEL data/sentiment_labeling_worksheet.csv >>
# << Save as data/sentiment_labeled.csv >>

# 4  Train sentiment model
mkdir -p models
python -m autocomplete.train_sentiment \
  --csv data/sentiment_labeled.csv \
  --out models/sentiment.pkl --seed 42

# 5  Generate final report artifacts
python scripts/run_final_report_results.py \
  --corpus data/en_US.twitter.txt \
  --sentiment-csv data/sentiment_labeled.csv \
  --model models/sentiment.pkl \
  --outdir results/final \
  --seed 42 --max-examples 500
```

After Step 5 completes you will find all five files under `results/final/`.
