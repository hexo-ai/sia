# Paper Review Task

## Overview

Your task is to implement a `predict_acceptance` function that predicts whether ML research papers will be accepted at a top conference. You write and iteratively improve a Python function, evaluating it against the ICLR 2024 benchmark.

## Problem

Given the title and abstract of papers submitted to a machine learning conference, predict whether each paper was accepted or rejected. The dataset is balanced (50% accept, 50% reject).

Your function is evaluated on a held-out test split of ICLR 2024 papers. Your final private score is computed on **ICLR 2023** and **ICLR 2022** papers — years you never see during development.

## Data Format

- Input `papers`: list of dicts with keys `'title'` (str) and `'abstract'` (str)  
- kwarg `train_papers`: list of dicts with keys `'title'`, `'abstract'`, **and `'label'`** (`'accept'` or `'reject'`) — labeled training examples
- Output: list of strings — each element must be exactly `'accept'` or `'reject'`

## Function Signature

```python
def predict_acceptance(papers, **kwargs):
    # papers:       list of {'title': str, 'abstract': str}  — to predict (no labels)
    # train_papers: list of {'title': str, 'abstract': str, 'label': str}  — labeled training set
    train_papers = kwargs.get('train_papers', [])
    return ['accept'] * len(papers)
```

## Evaluation

The dataset (200 ICLR 2024 papers) is split into 100 train + 100 test using a fixed seed. Your function receives:
- `papers` = the 100 test papers (no labels)
- `train_papers` = the 100 train papers (with labels)

Score is computed on the 100 test papers:
```
accuracy     = fraction of correct predictions
score        = max(0.0, (accuracy - 0.5) / 0.5)
```
- Random baseline: accuracy = 0.50 → score = 0.00
- Good ML model: accuracy ≈ 0.65 → score ≈ 0.30
- LLM-based (AI-Scientist-v2): accuracy = 0.63 → score = 0.26  ← reported by HyperAgents paper
- LLM-based (DGM-HA): accuracy = 0.71 → score = 0.42          ← reported by HyperAgents paper
- Perfect: accuracy = 1.00 → score = 1.00

## Available Libraries

`numpy`, `scipy`, `sklearn`, `torch`, `transformers`, `nltk`, `re`, `collections`, `string`

**Do NOT** make network calls or load files outside the dataset directory. Models loaded from HuggingFace cache (already on disk) are allowed.

## Rules

1. Write `solution.py` containing your `predict_acceptance` function (all helpers must be top-level)
2. Evaluate using `python {dataset_dir}/evaluate.py solution.py`
3. After each evaluation, `results.json` is automatically written to your working directory — do not write it yourself
4. At the end of your run, your working directory **must** contain `solution.py`
5. Iterate and improve until time runs out
6. No network IO inside `predict_acceptance`
7. No closures or lambdas — all helper functions must be top-level

## Suggested Starting Point

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def predict_acceptance(papers, **kwargs):
    train_papers = kwargs.get('train_papers', [])
    if not train_papers:
        return ['accept'] * len(papers)
    train_texts  = [p['title'] + ' ' + p['abstract'] for p in train_papers]
    train_labels = [p['label'] for p in train_papers]
    test_texts   = [p['title'] + ' ' + p['abstract'] for p in papers]
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train_texts)
    X_test  = vec.transform(test_texts)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, train_labels)
    return list(clf.predict(X_test))
```

This scores ~0.60–0.65 accuracy on the dev set. **Start here and improve.**

## Generalization

**The development dataset is ICLR 2024.** Your final score is computed on ICLR 2023 and ICLR 2022 — earlier years with different topic trends and area chair compositions.

A model that overfit to 2024-specific topic trends (e.g., "diffusion models", "RLHF", "RAG") will fail on 2022 papers where those terms were absent.
Approaches that use general semantic signals (writing quality, technical depth, contribution clarity) generalize better across years.

## Required Output

Your working directory must contain at the end:
- `solution.py` — your best `predict_acceptance` implementation
- `results.json` — written automatically by `evaluate.py` after the last evaluation

`results.json` format (written automatically):
```json
{
  "accuracy": 0.65,
  "score": 0.30,
  "elapsed_seconds": 1.2,
  "error": null
}
```

## Dataset Directory Layout

```
{dataset_dir}/
├── iclr2024.json   ← development dataset (200 papers, balanced)
├── evaluate.py     ← evaluation script
└── task.md         ← this file

{working_dir}/      ← your read/write workspace (initially empty)
```

## Evaluation Script

Run `python {dataset_dir}/evaluate.py solution.py` from your working directory.

## Tips

- TF-IDF + logistic regression is a fast, strong baseline — start there
- Abstract length, sentence complexity, and vocabulary richness correlate with acceptance
- Avoid keywords that are year-specific (e.g., "diffusion model", "RLHF", "GPT-4") — they won't generalize to 2022 papers
- If using embeddings, prefer sentence-level aggregation over word-level
- Calibrate probabilities: predicting soft scores then thresholding outperforms hard classification
