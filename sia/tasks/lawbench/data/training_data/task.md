# LawBench — Criminal Charge Prediction

You are evaluated on **how many criminal charges you predict correctly** from a fixed set of Chinese court case descriptions.

## Background

Each record comes from a real Chinese criminal case. The case description (事实, *facts*) is the text submitted by the procuratorate at trial. Your task is to predict which **criminal charge** (罪名) the court ultimately convicted the defendant of.

## Data

- **train.csv** — 5,332 labeled cases for training and validation
  - `id` — unique integer ID
  - `text` — Chinese case description (平均 ≈ 400 characters)
  - `label` — criminal charge string (one of 191 valid classes)
- **test.csv** — 913 unlabeled cases to predict
  - `id`, `text` — same as above, **no label column**
- **sample_submission.csv** — correct submission format
- **classes.json** — exhaustive list of all 191 valid charge labels

## Objective

**Maximize accuracy** (correct / attempted) across all 913 test cases.

## Evaluation (required)

After generating predictions, evaluate them by comparing against the ground-truth labels in the private split. Report the resulting accuracy. This check against true solutions is required — do not stop at generating predictions.

## Constraints

- Predict **only** from the 191 labels in `classes.json`. Any prediction not in this list counts as wrong.
- Output must be a CSV named `submission.csv` with columns `id` and `label` (exact case-sensitive match).
- All 913 test IDs must appear in your submission.

## Submission Format

```
id,label
0,盗窃
1,故意伤害
2,非法占用农用地
...
```

## Baseline Context

- A zero-shot LLM approach achieves ~7% accuracy (191 classes, hard to guess).
- A strong few-shot harness (Meta-Harness paper) achieves ~45%.
- Your goal is to exceed both — a well-engineered ML agent should reach 70%+.

## Model

If you use an LLM to generate predictions, use **`claude-haiku-4-5-20251001`** as the solver. For traditional ML approaches (TF-IDF, SVM, etc.), no model restriction applies.
