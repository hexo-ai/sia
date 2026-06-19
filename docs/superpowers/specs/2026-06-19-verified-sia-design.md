# Verified-SIA Design: Best-of-N, Execution-Gated, Keep-Best Self-Improvement

**Date:** 2026-06-19
**Status:** Approved (design), pending implementation plan
**Repo:** `~/Desktop/judge-experiments-papers/sia` (gitignored fork of SIA / hexo-ai)

## Problem

SIA's self-improvement loop is unreliable on small local models served via Ollama.
In a matched 5-seed study on Kaggle spaceship-titanic (qwen3-coder:30b, gemma4,
SIA_MAX_TURNS=50, max_gen 3), only ~1/3–2/3 of runs produced a competitive pipeline,
and even successful runs sometimes **regressed** generation-over-generation
(0.813 → 0.792). Root causes:

1. The meta-agent often writes a target that is itself a nested LLM-agent; on Ollama's
   stricter OpenAI-compatible endpoint these fail (HTTP 400 "invalid message content
   type", hallucinated completion with zero execution).
2. There is **no gate**: each generation's target replaces the previous one even when
   it is worse or broken — gains are not protected.
3. The loop is inefficient: up to 50 turns are spent on targets that were doomed from
   the first line.

## Goal

**Optimize P(run produces a competitive solution)** — turn ~1/3 yield into ~near-1,
cheaply, with no regressions. This is the *reliability/yield* objective, not chasing
SOTA accuracy.

**Guiding principle (consistent with the verification-asymmetry papers):** every
"is this better / does it work" decision is made by an **executable oracle scoring a
self-held-out validation split**, never by a model's say-so. All new machinery lives
in the **harness** (orchestrator/run-setup), not in generated agent code, so
generated-code variance cannot break orchestration.

## Architecture

SIA's existing loop (in `sia/orchestrator.py::run_generation`):

```
meta-agent writes gen_1/target_agent.py
  → run target (produces submission.csv)
  → (optional) evaluate
  → feedback agent writes gen_{g+1}/target_agent.py
  → repeat to max_gen
```

Verified-SIA adds four harness-side components around this loop. Generated target code
is unchanged in kind (still a script/agent the model writes), but it must honor one
new output contract (§ Component A).

### Component A — Validation oracle (the executable judge) — *the crux*

- **Setup-time split.** In `sia/run_setup.py`, once per run, deterministically split
  the task's labeled training data: `train.csv` → `train_inner.csv` + `val.csv` using
  a fixed seed. Write `train_inner.csv` and a **label-stripped** `val_features.csv`
  into the dataset view the agent sees; keep `val.csv` (with labels) in a
  harness-only directory under the run dir (e.g. `runs/run_<id>/_oracle/val.csv`).
- **Contract addition.** The target must, in addition to writing `submission.csv` for
  the unlabeled test set, also write `val_predictions.csv` predicting the rows in
  `val_features.csv` (same output format: `PassengerId,Transported` with True/False).
  This requirement is injected via the meta/feedback prompt (`sia/prompts.py`),
  alongside the existing local-endpoint adaptation block.
- **Scorer.** A harness function `score_val(cand_dir) -> float | None` reads
  `<cand_dir>/val_predictions.csv` (e.g. `gen_<g>/cand_<k>/val_predictions.csv`),
  joins to the held-out `val.csv` labels, returns accuracy (or `None` if the file is
  missing/unparseable/empty). This is the only
  signal the gate and selector use. The private test gold is **never** read at solve
  time (no leakage; the test submission is the deliverable, scored only post-hoc).

### Component B — Best-of-N engine

- Per generation, produce up to **N** candidate targets (N configurable, default 4):
  for gen 1 by re-invoking the meta-agent; for gen > 1 by re-invoking the feedback
  agent. Candidates differ via sampling (temperature / seed varied per candidate).
- Execute each candidate (existing target-run path), then call `score_val`.
- **Early-stop:** stop sampling once a candidate's val score ≥ `EARLY_STOP_THRESHOLD`
  (default 0.78).
- **Select:** `best = argmax val_score` over candidates that produced a valid
  `val_predictions.csv`. Candidates are stored under
  `gen_<g>/cand_<k>/` so nothing is overwritten.

### Component C — Keep-best gate (the ratchet)

- Maintain an **incumbent**: `(gen, cand, val_score, test_submission_path)`, the best
  target seen so far in the run.
- After a generation's `best` is chosen, accept it as the new incumbent **only if**
  `best.val_score > incumbent.val_score` (strict improvement; ties keep the earlier
  incumbent to avoid pointless churn). Otherwise retain the incumbent.
- On rejection, the feedback agent for the next generation receives the incumbent's
  code plus the rejected candidate's failure mode / score delta, so it improves the
  incumbent rather than a worse draft.
- The run's final deliverable = **incumbent's test `submission.csv`**.

### Component D — Pre-execution triage (optional; default = deterministic linter)

- Before executing a candidate, a regex linter `lint_target(path) -> list[str]`
  flags known-fatal patterns:
  - builds a nested LLM-agent in the solve path (`openai` import + `chat.completions`
    used to *decide*, not just present),
  - no write to `submission.csv`,
  - no write to `val_predictions.csv`,
  - label column written as `0/1` instead of `True/False`.
- A candidate failing the linter is skipped (counts against N but costs no execution).
- A model-judge is a drop-in alternative behind the same interface, but the
  deterministic linter is the default (more robust, zero cost). Linter is advisory
  only: execution + `score_val` remain the final arbiter.

## Data Flow (per run)

```
run_setup:
  split labeled data → train_inner.csv, val_features.csv (agent sees these)
                     → _oracle/val.csv (harness-only labels)
  incumbent = None

for g in 1..max_gen:
  candidates = []
  for k in 1..N:
    produce gen_<g>/cand_<k>/target_agent.py   (meta if g==1 else feedback)
    issues = lint_target(...)                  # Component D
    if issues: record (k, val=None, issues); continue
    run target → cand_<k>/{submission.csv, val_predictions.csv}
    val = score_val(...)                       # Component A
    candidates.append((k, val, paths))
    if val is not None and val >= EARLY_STOP_THRESHOLD: break   # Component B
  best = argmax_val(candidates)                # Component B (None if all failed)
  if best and (incumbent is None or best.val >= incumbent.val):  # Component C
    incumbent = best
  else:
    feedback_context = incumbent + best_failure_summary
deliverable = incumbent.test_submission  (or honest "no solution" if incumbent is None)
```

## Error Handling (robustness — "don't break things")

- Candidate crash / timeout / missing `val_predictions.csv` → `val_score = None`,
  excluded from selection. Never crashes the run.
- All N candidates fail in a generation → incumbent unchanged; feedback agent receives
  aggregated failure modes.
- Zero valid candidates across the whole run → emit an honest "no competitive solution"
  result and a non-zero status; do **not** raise an unhandled exception.
- The provider/LLM-summary crash class is already fixed (meta_provider threaded through
  ContextManager). Sampler/gate/oracle live entirely in the harness, so generated-code
  variance cannot break orchestration.

## Testing

- **Unit:**
  - `score_val`: synthetic `val_predictions.csv` vs known `val.csv` → exact expected
    accuracy; missing/empty file → `None`.
  - gate: a sequence of generation scores `[0.80, 0.74, 0.82]` → incumbent path is
    `0.80, 0.80, 0.82` (never decreases).
  - `lint_target`: flags a nested-agent fixture; passes a plain-script fixture; flags
    a `0/1`-label fixture.
- **Integration:** a mock target that deterministically writes a known-accuracy
  submission + val file (parametrized by candidate index) → run 2 generations with
  N=3, assert best-of-N selects the highest val candidate and incumbent is monotonic.
- **End-to-end:** 5 seeds × {qwen3-coder:30b, gemma4} on spaceship-titanic, max_gen 3,
  N=4. Compare against the recorded default/adapted baselines.

## Success Metrics

- **Primary:** competitive-run yield (val-competitive, i.e. some candidate's val score
  > 0.5) → target ≈ 1.0 vs current 1/3 (gemma) – 2/3 (qwen).
- **Secondary:** per-run incumbent val score is **monotonic non-decreasing** (no
  regressions) by construction; verify empirically.
- **Efficiency:** wasted compute (turns spent on linter-rejected or early-stopped
  candidates avoided) reported; expect lower tokens per competitive solution than
  brute best-of-N without triage/early-stop.

## Config (new env / Config fields)

- `SIA_BEST_OF_N` (default 4) — candidates per generation.
- `SIA_EARLY_STOP_THRESHOLD` (default 0.78) — val score that stops sampling.
- `SIA_VAL_FRACTION` (default 0.2) — held-out fraction of train.csv.
- `SIA_TRIAGE` (default "lint" | "off" | "judge") — pre-execution triage mode.
- Reuses existing `SIA_LOCAL_ADAPT` (plain-script directive) — kept on.

## Scope / Non-Goals

- Not chasing higher task accuracy (separate "task accuracy" objective via candidate
  ensembling — future work).
- Initially targets spaceship-titanic (has labeled train for the split). Generalizing
  the val-split contract to GPQA-style multiple-choice tasks is future work (there the
  oracle is per-question correctness on a held-out question subset).
- No change to SIA's cloud-provider behavior; all additions gate on the local/OpenAI
  path and the harness.
```
