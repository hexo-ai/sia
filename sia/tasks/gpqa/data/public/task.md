# GPQA-style benchmark (`diamond_questions.json`)

You are evaluated on **how many questions you answer correctly** on a fixed set of **graduate-level** multiple-choice items in the spirit of [GPQA](https://arxiv.org/abs/2311.12022): biology, chemistry, physics, and related domains at expert difficulty.

## Data

Each record in `diamond_questions.json` has:

- `id` — stable question identifier
- `domain`, `subdomain` — topic labels
- `Question` — stem (may include LaTeX-style notation or units)
- `options` — dictionary with keys `A`, `B`, `C`, `D` mapping to option text

The options are **pre-shuffled deterministically** (seeded on question text), so the correct answer is already randomly distributed across A-D. You must output which letter (`A`, `B`, `C`, or `D`) corresponds to the option you believe is correct.

## Objective

**Maximize the number of correct answers** over the full dataset. Accuracy (correct / attempted) is the primary success metric. Failed or invalid responses count against you as incorrect or errors depending on the harness.

## Output Format

Your agent must save results to `results/submission.json` (or timestamped filename) with this structure:

```json
{
  "model": "model_name",
  "dataset_config": "diamond_qna",
  "total_questions": 198,
  "errors": 0,
  "total_input_tokens": 12345,
  "total_output_tokens": 5678,
  "total_reasoning_tokens": 0,
  "total_cost_usd": 1.234,
  "timestamp": "2025-05-27T10:00:00",
  "details": [
    {
      "question_id": 1,
      "model_answer": "A",
      "model_answer_raw": "{\"answer\": \"A\"}",
      "input_tokens": 100,
      "output_tokens": 10,
      "reasoning_tokens": 0,
      "cost_usd": 0.01
    }
  ]
}
```

**Required fields in `details`:**
- `question_id` — matches the `id` from the dataset
- `model_answer` — single letter `A`, `B`, `C`, or `D`

**Optional fields** (for tracking):
- `model_answer_raw` — raw model output before parsing
- Token counts and costs (if available)

For errors, include `{"question_id": N, "error": "error message"}` instead.

## Model

Use **`models/gemini-3.1-flash-lite`** (Google Gemini) as the solver for each question. Configure generation so the model returns a **valid, parseable choice** (for example structured JSON `{"answer": "A"}` with `A`–`D` only).

## Constraints

- Answer **only** from the four given option strings for that question; do not invent a fifth option.  
- Follow the required output format exactly so automated grading can map your letter to the shuffled options. 
- All questions are independent and should not be kept in a single loop to solve all.
- Questions cannot share context with each other.