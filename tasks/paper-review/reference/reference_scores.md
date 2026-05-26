# Reference Scores — Paper Review Task

Scores are on the 100-paper held-out test set using the `score = max(0, (accuracy - 0.5) / 0.5)` metric.

## HyperAgents Paper Baselines (arxiv 2603.19461)

These are from the HyperAgents paper (Meta, 2026). They use GPT-4o-based agents on a
100-paper balanced ICLR 2025 test set (different from our dev set but comparable).

| Method | Test Accuracy | Score (normalized) | Notes |
|--------|--------------|-------------------|-------|
| Initial agent (random format) | 0.00 | 0.00 | Base agent fails to produce valid output |
| Random baseline | 0.50 | 0.00 | Theoretical |
| DGM (original) | 0.51 | 0.02 | Median of 5 runs; CI 0.00–0.51 |
| DGM-custom | 0.59 | 0.18 | Median; CI 0.57–0.65 |
| AI-Scientist-v2 (static baseline) | 0.63 | 0.26 | Yamada et al. 2025; GPT-4o-based reviewer |
| DGM-HA w/o open-ended exploration | 0.00 | 0.00 | Median; CI 0.00–0.56 |
| DGM-HA w/o self-improve | 0.00 | 0.00 | Median; CI 0.00–0.13 |
| **DGM-HA (full system)** | **0.71** | **0.42** | Median; CI 0.59–0.75 |

Note: all HyperAgents results use LLM API calls inside the agent function (GPT-4o). Our benchmark
is designed for non-API approaches but the scores provide useful reference targets.

## Expected Scores for Non-LLM Approaches (estimated)

| Method | Expected Accuracy | Expected Score |
|--------|------------------|----------------|
| Always-accept (trivial) | 0.50 | 0.00 |
| TF-IDF + Logistic Regression (C=1) | 0.60–0.65 | 0.20–0.30 |
| TF-IDF + SVM | 0.62–0.67 | 0.24–0.34 |
| TF-IDF + Gradient Boosting | 0.63–0.68 | 0.26–0.36 |
| Sentence embeddings + cosine sim | 0.60–0.65 | 0.20–0.30 |
| Ensemble of above | 0.65–0.70 | 0.30–0.40 |

## Generalization Gap

On the private evaluation (NeurIPS 2023 + ICLR 2023), expect ~3–8 accuracy points of drop
from the dev score due to conference distribution shift. Methods relying heavily on ICLR-specific
vocabulary will see larger drops.
