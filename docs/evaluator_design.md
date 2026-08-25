# Evaluator design checklist

SIA optimizes whatever signal your evaluator writes to `results.json`. Treat the
evaluator as the task specification: if it rewards the wrong behavior, the
self-improvement loop will reliably discover and amplify that behavior.

Use this checklist when creating or reviewing `data/public/evaluate.py` for a
custom task.

## 1. Keep the public task and private score separate

- Put inputs the target agent may read in `data/public/`.
- Put answer keys, hidden tests, and held-out examples in `data/private/`.
- Do not copy private labels, expected outputs, or hidden-case identifiers into
  `task.md`, sample submissions, prompts, reference agents, or logs.
- If the target agent runs with direct host filesystem access, remember that
  `data/private/` is a convention enforced by the harness, not a security
  boundary. Use Docker/external sandboxing when you need hard isolation.

## 2. Make `results.json` the canonical artifact

The orchestrator calls:

```bash
python data/public/evaluate.py --gen-dir runs/run_1/gen_1
```

Your script should find the generation's submission, score it, and write:

```text
runs/run_1/gen_1/results.json
```

Recommended top-level fields:

```json
{
  "accuracy": 0.82,
  "accuracy_percent": 82.0,
  "correct": 82,
  "incorrect": 18,
  "missing": 0,
  "invalid": 0,
  "total_questions": 100,
  "details": []
}
```

Top-level scalar metrics are easiest for the feedback prompt and web dashboard to
summarize. Keep verbose per-example data under `details` so it can be inspected
without hiding the primary score.

## 3. Score the behavior you actually want

Weak metrics become optimization targets. Avoid metrics that can be satisfied by
formatting or surface-level hacks unless that is truly the task.

Examples of fragile signals:

- keyword counts instead of semantic correctness
- citation counts without checking whether citations support the claim
- exact-string matching when many paraphrases should be valid
- public-test-only scores when the agent can tune directly to those cases
- rewarding verbosity, code size, or tool usage instead of final quality

Prefer task-grounded checks:

- compare against hidden labels, unit tests, or reference outputs
- include counterexamples and false-premise cases
- validate required output schemas before scoring
- penalize missing, invalid, duplicate, or unparsable submissions explicitly
- report both aggregate scores and enough details to debug failures

## 4. Include robustness slices

A single scalar can hide brittle progress. Add slice metrics that reveal whether
an improvement generalizes:

- easy vs hard examples
- common vs rare classes
- public/dev vs private/held-out split
- in-distribution vs paraphrased or perturbed inputs
- previously failed examples vs newly introduced hidden cases

Keep the primary metric stable across generations, but expose slices so the
feedback agent can distinguish real improvement from overfitting.

## 5. Be deterministic and reproducible

- Sort inputs before scoring when file order should not matter.
- Use fixed seeds for randomized evaluators.
- Include exact counts, not just percentages.
- Fail loudly for malformed ground truth; count malformed submissions as
  `invalid` instead of crashing when possible.
- Make the evaluator runnable locally without a full SIA run.

A useful smoke test is:

```bash
python data/public/evaluate.py --gen-dir /tmp/example-gen
python -m json.tool /tmp/example-gen/results.json
```

## 6. Review the evaluator as adversarial surface

Before launching a long run, ask:

- Could an agent get a high score without solving the task?
- Could it infer private answers from filenames, logs, examples, or prompts?
- Does the evaluator accidentally read the agent's old `results.json` as a new
  submission?
- Are hidden examples diverse enough to catch prompt- or format-specific hacks?
- Would a human trust the metric if the generated solution were deployed?

If the answer is unclear, add a small hidden regression case or an explicit
validation rule before running the self-improvement loop.
