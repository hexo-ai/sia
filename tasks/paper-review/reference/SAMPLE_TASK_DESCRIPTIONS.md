# Sample Task Descriptions — ML Paper Acceptance Prediction

These are example descriptions of the paper-review task as it should be presented to a target agent.
They emphasize that a good solution must generalize across conferences and years, not just perform
well on the development dataset.

---

## Example A

Implement `predict_acceptance(papers, **kwargs)` in `solution.py`. Each paper is a dict with
`'title'` and `'abstract'`. You also receive `train_papers` in `kwargs` — the same structure
plus `'label'` (`'accept'` or `'reject'`).

Your solution is **developed on ICLR 2024 papers**, but the final score is computed privately on
**NeurIPS 2023** and **ICLR 2023** papers you never see during development. Overfitting to ICLR-specific
patterns (venue vocabulary, formatting conventions, trending topics) will hurt generalization.

Evaluate with `python {dataset_dir}/evaluate.py solution.py`. Score = max(0, (accuracy - 0.5) / 0.5),
range 0–1, higher is better. A TF-IDF + logistic regression baseline scores ~0.30 and is the
recommended starting point.

---

## Example B

Your task is to predict ML paper acceptance by implementing `predict_acceptance(papers, **kwargs)`.
Each paper has a `'title'` and `'abstract'`. You receive labeled training examples via
`kwargs['train_papers']`.

A strong solution captures general signals of paper quality — technical clarity, novelty framing,
experimental rigor — that hold across conferences. The ICLR 2024 development set is your proxy:
the real test is whether your approach generalizes to **NeurIPS 2023** and **ICLR 2023** — different
venues and years that you never see during development.

Start with `sklearn`'s `TfidfVectorizer` + `LogisticRegression` (scores ~0.30), then iterate:
try richer features (abstract length, sentence complexity, topic modeling), different classifiers
(SVM, gradient boosting), or semantic embeddings if available.

---

## Example C

Implement a generalizable paper acceptance predictor. The key challenge: acceptance decisions at
top ML venues reflect paper quality — novelty, technical rigor, clarity, and significance. These
signals are partially encoded in the abstract: good papers often have clear problem statements,
quantified improvements, and precise technical language.

Develop on the ICLR 2024 benchmark (`python {dataset_dir}/evaluate.py solution.py`), but design
as if you will be evaluated on any top ML conference. The private evaluation runs on **NeurIPS 2023**
and **ICLR 2023** — techniques relying on ICLR-specific vocabulary, citation counts to 2024 work,
or topic trends from 2024 will underperform on these held-out conferences.

Score = max(0, (accuracy - 0.5) / 0.5). A TF-IDF + LogReg baseline scores ~0.30. Published
LLM-based systems (AI-Scientist-v2) score ~0.26 and DGM-HA scores ~0.42 on the HyperAgents benchmark.
