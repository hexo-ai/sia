# Verified-SIA: Best Practices for Running SIA on Local (Ollama) Models

This document records what actually works — and what doesn't — when running the SIA
self-improving-agent framework against local open-weight models served through
Ollama's OpenAI-compatible endpoint. The throughline, validated experimentally:

> **Bind decisions to an external executor, not to a model's own report.** A model
> certifying, revising, or aggregating a model is unreliable; an executable oracle
> (run it, score it against held-out labels) is what reliably produces a good result.

The `Verified-SIA` additions (module `sia/verified.py`, config flags below) implement
this. All numbers here are on Kaggle *spaceship-titanic* (tabular binary classification;
random ≈0.50, competent pipelines ≈0.79–0.82), scored on the genuinely private test
gold the agent never sees.

---

## 1. Setup gotchas (do these first)

### 1.1 Use a model that emits real `tool_calls`
SIA's agents need OpenAI-style structured `tool_calls`. Verify before anything else:

```bash
curl -s http://localhost:11434/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "<model>",
  "messages": [{"role":"user","content":"list /tmp with the ls tool"}],
  "tools": [{"type":"function","function":{"name":"ls","parameters":{"type":"object",
            "properties":{"path":{"type":"string"}}}}}]}' | python3 -c \
  'import sys,json; m=json.load(sys.stdin)["choices"][0]["message"]; print("OK" if m.get("tool_calls") else "NO tool_calls")'
```

- ✅ Work: `qwen3-coder:30b`, `gpt-oss:20b`, `llama3.1:8b`, `qwen3:8b`, `llama3.3`,
  `qwen2.5:7b`, **`gemma4`** (the new one).
- ❌ Don't: `qwen2.5-coder:32b` (dumps the call as JSON text in `content`),
  `gemma2:9b` / `gemma3:27b` (Ollama returns HTTP 400 "does not support tools").

### 1.2 Provider must be threaded everywhere
Ollama model names are `name:tag` (e.g. `qwen3-coder:30b`). Any pydantic-ai
`Agent(model_string)` path that does **not** receive the explicit provider object will
misparse the colon as `provider:model` and crash with
`ValueError: Unknown provider: qwen3-coder`. This bit the post-generation LLM-summary
path. **Audit every `run_agent` call site for `provider=`.**

### 1.3 Providers / profiles
Create an OpenAI-compatible provider pointing at Ollama and profiles that reference it:

```jsonc
// providers/ollama.json
{"provider_id":"ollama","client_kind":"openai",
 "base_url":"http://localhost:11434/v1","api_key_env":"OLLAMA_API_KEY"}
```
Run with `OLLAMA_API_KEY=ollama SIA_PROVIDERS_DIR=$PWD/providers SIA_PROFILES_DIR=$PWD/profiles`.

---

## 2. The dominant failure mode, and the fix

**By default the meta-agent writes a target that is itself a nested LLM-agent** (it
mirrors SIA's own structure and calls the task model to "decide"). On Ollama these
nested loops fail three ways:

1. HTTP 400 `invalid message content type` — Ollama is stricter than hosted OpenAI and
   requires tool-result message `content` to be a **string**, not a structured object.
2. Hallucinated tools (`Unknown tool: ls`).
3. **Narrating fake success** — "Achieved validation accuracy 0.7942; created
   submission.csv" while executing nothing and writing no file. *This is a verification
   artifact: the self-report is fiction; only execution reveals it.*

**Fix — direct it to write a plain script.** The `SIA_LOCAL_ADAPT` prompt block tells
the meta-agent to emit a self-contained `pandas`/`sklearn` program that solves the task
directly (no nested agent, no `chat.completions`), with correct I/O (True/False labels;
identical preprocessing for train/test/val to avoid `KeyError`). This alone lifts the
competitive-run yield from ~1/5 to ~2/3 (qwen3-coder) and from 0/3 to 1/3 (gemma4).

---

## 3. The reproducible win: best-of-N + an execution gate

Sample `N` candidate targets, keep only those that **actually run and score** on a
self-held-out validation split (labels withheld by the harness), and **keep the best**
(never regress). This is the dependable core:

- The gate **never ships a non-running or regressed solution** (broken candidates score
  `None` and are excluded). Plain "adapted" runs sometimes deliver a `0.000` broken
  submission; the gate structurally cannot.
- **Raise N to raise yield.** N=8 delivered a competitive pipeline (~0.80–0.82) in every
  run; N=4 left some runs with no valid candidate.
- The validation score is a **faithful proxy**: val ≈ private test (e.g. val 0.806 /
  test 0.779 on the same candidate), so selection generalizes — no meaningful leakage.

Run it:
```bash
SIA_BEST_OF_N=8 SIA_EARLY_STOP_THRESHOLD=0.80 \
sia run --task spaceship-titanic --target-agent-profile ollama-target \
  --meta-agent-profile ollama-meta --max_gen 1 --run_id <int> --sandbox none --no-web
```

---

## 4. What did NOT help (measured, so you don't repeat it)

### 4.1 Repair loop — poor ROI
Feeding the executable traceback back for fix-it turns *sounds* ideal (grounded
feedback). In practice the meta-agent **rewrites the whole target from scratch** with a
hint rather than patching, so a "repair" is just an expensive extra draw. In one N=8
run it rescued **0 of 4** failures and tripled wall-clock (~80 min). **Recommendation:
keep `SIA_REPAIR_RETRIES=0`** (or 1 at most); spend the budget on higher N instead.

### 4.2 Ensembling the gate-passers — a null on this task
Majority-voting the survivors did **not** beat taking the single validation-best
survivor: over 4 seeds, ensemble vs single-best = **0.807 vs 0.806** (per-seed
Δ = +0.005, 0, 0, 0). Why: best-of-8 usually leaves only ~2 survivors, and they are
near-identical `sklearn` pipelines whose errors are **correlated** — committees need
*diverse* members to help. A single favorable 4-member draw showed 0.821 vs 0.816;
only running more seeds revealed it as variance, not an effect.

> Lesson (and a clean instance of the verification-asymmetry thesis): **don't promote an
> n=1 favorable draw to a result.** The lift comes from the *selection* step (an external
> executor deciding who runs and scores), not from model-level aggregation on top of it.

### 4.3 Multi-generation feedback refinement — weak
The generation-over-generation feedback agent rarely improved a working pipeline and
sometimes regressed it (e.g. 0.813 → 0.792). Verified-SIA therefore runs best-of-N at
**generation 1 only** and gates; multi-generation refinement under the gate is future
work.

---

## 5. Known wart

`make_val_split` writes `train_inner.csv` / `val_features.csv` into the shared task
`data/public/` and leaves the full-label `train.csv` visible to the agent. A candidate
that trains on `train.csv` can leak val labels — which inflates the *gate's* val signal
but **not** the final metric (the deliverable is scored on the truly-private test gold).
The clean fix is a per-run dataset directory that excludes `train.csv`; until then,
trust the private-gold numbers, not the absolute val numbers.

---

## 6. Config reference (`sia/config.py`, all via `SIA_*` env)

| Flag | Default | Meaning |
|---|---|---|
| `SIA_BEST_OF_N` | 4 | candidates per generation (raise for yield) |
| `SIA_EARLY_STOP_THRESHOLD` | 0.78 | stop sampling once a candidate's val ≥ this |
| `SIA_VAL_FRACTION` | 0.2 | held-out fraction of `train.csv` |
| `SIA_TRIAGE` | `lint` | pre-execution linter (`lint`/`off`/`judge`) |
| `SIA_REPAIR_RETRIES` | 0 | fix-it turns per failed candidate — **leave at 0** |
| `SIA_ENSEMBLE` | false | majority-vote gate-passers — **null on correlated tasks** |
| `SIA_VAL_FLOOR` | 0.75 | min val to join the ensemble |
| `SIA_LOCAL_ADAPT` | 1 | plain-script directive (set `0` to reproduce the unadapted default) |

---

## 7. TL;DR

1. Pick a tool-calling model; thread the provider everywhere.
2. Force a **plain script**, not a nested agent.
3. Use **best-of-N + the execution gate** (keep what runs and scores on held-out val).
4. **Raise N** for yield; **don't** rely on repair or ensembling — both were measured
   nulls/negatives here.
5. Score the deliverable on truly-held-out labels, and **never promote a single lucky
   draw to a result** — run seeds.
