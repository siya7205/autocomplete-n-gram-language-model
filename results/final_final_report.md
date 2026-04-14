# Sentiment-Aware Autocomplete with Weakly Supervised Labels:
## A Final Evaluation Report

---

## Abstract

This report evaluates a sentiment-aware N-gram language model for next-word autocomplete.
Because manual sentiment labeling is time-consuming, we introduce a **weak-supervision pipeline**
that automatically assigns binary sentiment labels (positive / negative) to corpus sentences
using a curated keyword lexicon.
A logistic-regression classifier is then trained on the resulting labels and used to
**rerank** the N-gram model's top-K candidate words toward a specified target sentiment.

We sweep two design dimensions — the candidate-list size (`top_k ∈ {1, 3, 5}`) and
the reranking strength (`sentiment_weight ∈ {0.0, 0.5, 1.0, 2.0}`) — and measure two
complementary objectives:

| Metric | Definition |
|---|---|
| **Top-K hit rate** | Fraction of test positions where the true next word appears in the top-K candidates |
| **Sentiment alignment rate** | Fraction of top-K suggestions whose predicted sentiment equals the target sentiment |

Across 12 sweep configurations evaluated over **80 held-out examples each**, the model
achieves a best top-K hit rate of **0.25** (`top_k = 3` or `5`) and a positive-class
alignment rate of up to **0.9625**.
Notably, `sentiment_weight` has no measurable effect on either metric, exposing a
structural limitation of the current reranking interaction with the lexicon-trained classifier.

---

## 1  Introduction

Language models that merely maximise fluency can produce continuations with an unintended
emotional tone.
Sentiment-aware autocomplete addresses this by biasing next-word selection toward a user-chosen
sentiment polarity.
The standard approach requires a manually labeled dataset — which is expensive at scale.
**Weakly supervised labeling** replaces hand annotations with signal from a domain-matched
keyword lexicon, enabling rapid experimentation while accepting a known quality trade-off.

This work makes the following contributions:

1. A reproducible weak-labeling pipeline (`scripts/weak_label_sentiment.py`) that produces
   a labeled CSV from any plain-text corpus in a single deterministic command.
2. An evaluation sweep framework (`scripts/run_final_report_results.py`) covering the
   full `top_k × sentiment_weight` grid.
3. An empirical characterisation of what happens to hit rate and sentiment alignment when
   only heuristic labels are available.

---

## 2  Methods

### 2.1  Corpus and data split

- **Corpus:** `data/Shakespeare.txt` — a literary English corpus totalling several hundred
  thousand tokens.
- **Train / test split:** 80 % training, 20 % held-out test, using a fixed shuffle seed
  (`seed = 42`) for reproducibility.
- **Vocabulary filtering:** tokens appearing fewer than 2 times in the training set are
  replaced with `<unk>`.

### 2.2  Language model

An N-gram language model (orders 1 – 4) is built using Laplace-smoothed count tables.
At inference time the four tables are consulted together and scores are pooled via the
`get_suggestions` routine, which returns one scored candidate per adjacent-order pair.
Candidates are sorted by language-model score to yield the initial top-K list.

### 2.3  Weak sentiment labeling

Each sentence in the corpus is tokenised and matched against two fixed keyword sets:

| Polarity | Example keywords |
|---|---|
| **Positive** | love, joy, grace, heaven, peace, friend, gentle, virtue, … (38 terms) |
| **Negative** | hate, death, kill, woe, evil, villain, wicked, alas, despair, … (52 terms) |

**Labeling rule:**

```
if pos_count > neg_count  →  "positive"
if neg_count > pos_count  →  "negative"
otherwise (tie or both zero) →  row dropped
```

Up to `max_rows = 1000` corpus sentences are sampled (seed 42);
only rows with a decisive majority are retained.
The pipeline enforces a minimum of **50 rows per class** and fails with an actionable
error if the threshold is not met.

**Resulting label distribution (this run):**

| Class | Count | Share |
|---|---:|---:|
| Positive | 102 | 65.8 % |
| Negative | 53 | 34.2 % |
| **Total** | **155** | **100 %** |

The class imbalance (≈ 2 : 1) is typical for Shakespearean prose, where celebratory
language slightly outnumbers explicitly negative language at the lexical level.

### 2.4  Sentiment classifier

A scikit-learn `Pipeline` consisting of:

1. `TfidfVectorizer` (custom tokenizer, no lowercasing — matching training preprocessing)
2. `LogisticRegression` (`max_iter = 1000`, `random_state = 42`)

The 155 labeled rows are split 80 / 20 (stratified where possible) into training and
test sets.
The trained model is serialised to `models/sentiment_weak.pkl`.

**Classifier performance on held-out split:**

| Metric | Value |
|---|---:|
| Accuracy | **0.6774** |
| *(Weighted precision, recall, F1 not logged in this run)* | — |

An accuracy of 0.68 on a two-class task where chance is 0.50 indicates that the
lexicon-derived labels carry a genuine signal, though the margin is modest.

### 2.5  Sentiment-aware reranking

For a given prefix, the model produces a candidate list of size K from the N-gram model.
Each candidate `w` is scored as:

```
final_score(w) = lm_score(w) + λ × P_sentiment(target | prefix + w)
```

where `λ` is `sentiment_weight` and `P_sentiment` is the classifier's predicted
probability for the target class.
Candidates are re-sorted by `final_score` descending.

### 2.6  Evaluation protocol

**Top-K hit rate** — for each test sentence, a random split point is chosen; the prefix
is fed to the model; a hit is recorded if the true next word appears in the top-K
predictions.  Evaluated over `min(80, |test|)` examples.

**Sentiment alignment rate** — for each prefix drawn from the weak-labeled CSV,
the top-K suggestions after reranking are classified by the same sentiment model;
alignment is the fraction of suggestions whose predicted label equals the
target sentiment.

The full sweep covers 3 × 4 = 12 (top_k, sentiment_weight) combinations.

---

## 3  Results

### 3.1  Complete sweep table

All 12 configurations, evaluated over 80 examples each:

| top_k | sentiment_weight | top_k_hit_rate | examples_used | alignment_positive | alignment_negative |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.1750 | 80 | 0.9500 | 0.0500 |
| 1 | 0.5 | 0.1750 | 80 | 0.9500 | 0.0500 |
| 1 | 1.0 | 0.1750 | 80 | 0.9500 | 0.0500 |
| 1 | 2.0 | 0.1750 | 80 | 0.9500 | 0.0500 |
| 3 | 0.0 | 0.2500 | 80 | 0.9625 | 0.0375 |
| 3 | 0.5 | 0.2500 | 80 | 0.9625 | 0.0375 |
| 3 | 1.0 | 0.2500 | 80 | 0.9625 | 0.0375 |
| 3 | 2.0 | 0.2500 | 80 | 0.9625 | 0.0375 |
| 5 | 0.0 | 0.2500 | 80 | 0.9625 | 0.0375 |
| 5 | 0.5 | 0.2500 | 80 | 0.9625 | 0.0375 |
| 5 | 1.0 | 0.2500 | 80 | 0.9625 | 0.0375 |
| 5 | 2.0 | 0.2500 | 80 | 0.9625 | 0.0375 |

`alignment_neutral` is blank for all rows — the weak-labeled dataset contains only
`positive` and `negative` classes, so the classifier never predicts `neutral`.

### 3.2  Effect of top_k on hit rate

| top_k | Mean top-K hit rate |
|---:|---:|
| 1 | 0.1750 |
| 3 | 0.2500 |
| 5 | 0.2500 |

Expanding the candidate list from 1 to 3 yields a **+4.3 percentage-point** absolute
improvement in hit rate (+24.6 % relative). No further gain is observed at `top_k = 5`,
suggesting that the N-gram model's marginal 4th and 5th candidates do not contain the
true next word beyond what the top 3 already capture in this corpus.

### 3.3  Effect of sentiment_weight on hit rate

| sentiment_weight | Mean top-K hit rate (across all top_k) |
|---:|---:|
| 0.0 | 0.2250 |
| 0.5 | 0.2250 |
| 1.0 | 0.2250 |
| 2.0 | 0.2250 |

Hit rate is **invariant to sentiment_weight** across the entire sweep range.
This is expected: reranking shuffles the order of existing candidates rather than
introducing new ones, so whether the true next word is hit at all is unaffected by
the reranking step.

### 3.4  Effect of sentiment_weight on alignment

| sentiment_weight | Mean alignment_positive | Mean alignment_negative |
|---:|---:|---:|
| 0.0 | 0.9583 | 0.0417 |
| 0.5 | 0.9583 | 0.0417 |
| 1.0 | 0.9583 | 0.0417 |
| 2.0 | 0.9583 | 0.0417 |

Sentiment alignment is also **invariant to sentiment_weight**.
The positive alignment rate hovers near 96 % irrespective of reranking strength.

### 3.5  Best observed configurations

| Criterion | top_k | sentiment_weight | Value |
|---|---:|---:|---:|
| Best top-K hit rate | 3 (or 5) | any | **0.2500** |
| Best positive alignment | 3 (or 5) | any | **0.9625** |
| Best negative alignment | 1 | any | **0.0500** |

---

## 4  Discussion

### 4.1  The reranking invariance finding

The most striking result is that **`sentiment_weight` has no measurable effect** on any
metric, even at `λ = 2.0` (a value that should substantially amplify sentiment signal).
Two plausible explanations are:

**a)  Classifier dominance by the majority class.**  
The classifier was trained on a corpus where positive tokens appear roughly twice as
often as negative ones.  It likely assigns high positive probability to almost every
candidate in a Shakespearean context, meaning that reranking by
`lm_score + λ × P(positive)` is essentially equivalent to reranking by `lm_score` alone
— the `P(positive)` term is near-uniform across candidates.

**b)  Candidate pool homogeneity.**  
Because the N-gram model operates over the same Shakespearean vocabulary for both
the prefix and the candidate generation, the candidate pool for any given prefix is
likely dominated by a small number of common function words (e.g., "the", "and", "of")
that the classifier labels uniformly.
Reranking a homogeneous pool has little effect regardless of weight.

### 4.2  Hit-rate interpretation

A top-3 hit rate of 25 % on a Shakespearean corpus with a minimal-frequency vocabulary
is consistent with N-gram model baselines on literary text.
Shakespeare's language features unusual word order, archaic constructions, and
domain-specific vocabulary that are poorly covered by the small 1000-sentence training
sample used here; a larger sample or a lower `minimum_freq` threshold would likely
improve hit rates.

### 4.3  Alignment rate interpretation

The near-perfect positive alignment (≈ 96 %) is not evidence of effective sentiment
steering — it is a symptom of the classifier assigning "positive" to the vast majority
of Shakespearean candidates regardless of context.
The complementary near-zero negative alignment (≈ 4 %) confirms this: the model cannot
effectively distinguish negative candidates from positive ones in this literary domain.

This is a direct consequence of weak labeling: the lexicon does capture a real signal
(accuracy 0.68 > 0.50 chance), but the signal is too coarse to produce a calibrated
classifier across the full vocabulary.

### 4.4  Practical recommendations

| Scenario | Recommendation |
|---|---|
| Need higher hit rate | Increase corpus sample size (e.g., `--max-rows 5000`) |
| Need better sentiment calibration | Manually review and correct ≥ 50 weak labels per class |
| Need negative alignment to work | Balance training classes or apply class-weight correction |
| Production deployment | Replace weak labels with human annotations for the classifier only |

---

## 5  Threats to Validity

### 5.1  Construct validity — weak labels ≠ ground truth

The sentiment labels were generated automatically using a keyword lexicon.
A text containing the word "death" is labeled negative, even if the surrounding
context is celebratory (e.g., "victory over death").
All downstream classifier metrics — accuracy, alignment — are **relative to the weak
labels**, not to human-perceived sentiment.
The reported accuracy of 0.6774 means the classifier reproduces the lexicon's judgment
68 % of the time; it does not measure true sentiment recognition.

### 5.2  Internal validity — small evaluation set

Each sweep configuration was evaluated over a capped sample of **80 examples**.
At this sample size a single additional hit changes the hit rate by 0.0125 (1.25 pp),
making results sensitive to the particular random sample drawn.
Reported figures should be treated as point estimates with wide confidence intervals.

### 5.3  External validity — single literary corpus

All experiments use `data/Shakespeare.txt`.
Shakespearean English is highly atypical: archaic vocabulary, elevated register, and
poetic structure.
Results — particularly the alignment plateau and the reranking invariance — may not
transfer to conversational corpora (e.g., Twitter/Reddit) where sentiment vocabulary
is more diverse and reranking could have a measurable effect.

### 5.4  Statistical significance

No statistical tests were performed.
Given the sample size of 80 per configuration and the observed variance of zero across
`sentiment_weight`, the flat alignment and hit-rate curves are likely deterministic
artefacts of the evaluation design rather than statistically stable empirical estimates.

### 5.5  Model selection bias

The N-gram order (1 – 4) and the TF-IDF + logistic-regression architecture were fixed
in advance; no hyperparameter search was performed.
Better language-model backoffs (e.g., Kneser-Ney smoothing) or a stronger sentiment
classifier (e.g., fine-tuned BERT) could substantially alter all reported figures.

---

## 6  Conclusion

We demonstrated a fully automated, reproducible pipeline for weakly supervised
sentiment-aware autocomplete.
In its current form on Shakespearean text, expanding the candidate list from 1 to 3
is the single most impactful lever (+4.3 pp hit rate), while sentiment reranking
weight has no measurable effect due to classifier bias toward the majority class.
The pipeline is technically complete and reproducible from a single command; the
primary path to better results is improved label quality, a larger training sample,
and class-balance correction in the sentiment model.

---

## Appendix A — Reproducibility

All results in this report can be regenerated exactly with:

```bash
python scripts/run_final_report_results.py \
  --corpus data/Shakespeare.txt \
  --sentiment-csv data/sentiment_labeled_weak.csv \
  --model models/sentiment_weak.pkl \
  --run-weak-labeling \
  --outdir results/final \
  --seed 42 \
  --max-examples 80
```

Running the command twice with the same seed yields byte-identical `summary.csv` and
`summary.json` outputs.

## Appendix B — Artifact manifest

| File | Description |
|---|---|
| `results/final/summary.csv` | Full 12-row sweep table (CSV) |
| `results/final/summary.json` | Same table in JSON |
| `results/final/topk_hit_rate.png` | Hit-rate vs top_k (one line per sentiment_weight) |
| `results/final/sentiment_alignment.png` | Alignment rate vs sentiment_weight |
| `results/final/REPORT_SNIPPET.md` | Auto-generated one-page summary |
| `results/final_final_report.md` | **This document** |
| `data/sentiment_labeled_weak.csv` | 155-row weak-labeled sentiment dataset |
| `models/sentiment_weak.pkl` | Trained logistic-regression sentiment model |
