# Final Report Results Snippet

## Dataset and model inputs
- Corpus: `data/Shakespeare.txt`
- Sentiment CSV: `data/sentiment_labeled_weak.csv`
- Sentiment model: `models/sentiment_weak.pkl`
- Seed: `42`

## Best observed configurations
- Best top-k hit rate: **0.2500** (top_k=3, sentiment_weight=0.0)
- Best sentiment alignment (mean over available labels): **0.5000** (top_k=1, sentiment_weight=0.0)

## Summary table
| top_k | sentiment_weight | top_k_hit_rate | examples_used | alignment_positive | alignment_negative | alignment_neutral |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0 | 0.1750 | 80 | 0.9500 | 0.0500 |  |
| 1 | 0.5 | 0.1750 | 80 | 0.9500 | 0.0500 |  |
| 1 | 1.0 | 0.1750 | 80 | 0.9500 | 0.0500 |  |
| 1 | 2.0 | 0.1750 | 80 | 0.9500 | 0.0500 |  |
| 3 | 0.0 | 0.2500 | 80 | 0.9625 | 0.0375 |  |
| 3 | 0.5 | 0.2500 | 80 | 0.9625 | 0.0375 |  |
| 3 | 1.0 | 0.2500 | 80 | 0.9625 | 0.0375 |  |
| 3 | 2.0 | 0.2500 | 80 | 0.9625 | 0.0375 |  |
| 5 | 0.0 | 0.2500 | 80 | 0.9625 | 0.0375 |  |
| 5 | 0.5 | 0.2500 | 80 | 0.9625 | 0.0375 |  |
| 5 | 1.0 | 0.2500 | 80 | 0.9625 | 0.0375 |  |
| 5 | 2.0 | 0.2500 | 80 | 0.9625 | 0.0375 |  |

_Note: weak labels are heuristic and indicative, not ground truth._
