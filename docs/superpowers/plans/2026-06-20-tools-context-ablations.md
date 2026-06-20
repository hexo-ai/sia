# Harness-Knob Ablations: Tool Exposure & Context Engineering

**Goal:** Extend the harness-knob audit (paper Table `tab:knobaudit`) across two more of
the survey's design dimensions — **Tool Systems** and **Context Management** — using the
same discipline: vary one knob, hold the rest fixed, read the outcome naively, then
re-read with seeds + a power check before believing it.

**Why these two:** the harness-engineering survey makes specific claims we can test —
"tool overload → attention dispersion → worse decisions" (Tool Systems) and
"context rot / irrelevant retrieval degrade reasoning" (Context Management). Each is a
chance to either (a) replicate a real effect, or (b) show it's a manufacturable artifact
in a controlled, seeded design. Both outcomes are publishable audit data points.

**Setup (shared):** spaceship-titanic, qwen3-coder:30b, gen-1 best-of-4 + executable
gate (`SIA_BEST_OF_N=4 SIA_ENSEMBLE=0 SIA_REPAIR_RETRIES=0`), `SIA_MAX_TURNS=50`,
3 seeds per variant. Deliverable scored on the **private test gold** (never seen).
Selection is held fixed (executable gate), so any movement is attributable to the knob.
Runs are sequential (Ollama serializes the 30B model); **do not launch until the pool-gen
runs 91–96 finish.**

**Metrics per run:** delivered private-gold accuracy; competitive yield (k/3); and a
knob-specific diagnostic (tool-call counts for Tools; prompt-token count for Context).

**Audit per knob:** report the *naive* single-run reading (the most striking pairwise
gap) next to the *seeded* mean ± spread, and a power note (with n=3 per cell, what
discordant split would be detectable). Classify **robust** vs **artifact**. Append two
rows to `tab:knobaudit`.

---

## Experiment T — Tool exposure (Tool Systems dimension)

**Knob:** the meta-agent's toolset. The meta-agent (pydantic-ai) currently gets
`[write_file, read_file, bash]` from `_make_tools(working_dir)` in
`sia/agent_impls/pydantic_ai.py`.

**Variants (env `SIA_TOOLSET`):**
- **`minimal`** — `[write_file]` only (it must write the target; nothing else).
- **`standard`** — `[write_file, read_file, bash]` (current default).
- **`overloaded`** — standard + **6 decoy tools** that are plausible but useless for
  writing a self-contained script: `web_search(query)`, `calculator(expr)`,
  `list_directory(path)`, `http_get(url)`, `sql_query(q)`, `summarize(text)`. Each
  returns a short canned string (no side effects). This operationalizes "tool overload."

**Implementation:**
1. `sia/agent_impls/pydantic_ai.py`: change `_make_tools(working_dir)` →
   `_make_tools(working_dir, toolset="standard")`. Keep the three real tools; add the six
   decoy closures (each `def web_search(query: str) -> str: return "[no results]"`, etc.,
   with honest docstrings like "Search the web for a query"). Return:
   - `minimal` → `[write_file]`
   - `standard` → `[write_file, read_file, bash]`
   - `overloaded` → `[write_file, read_file, bash, web_search, calculator, list_directory, http_get, sql_query, summarize]`
   Read the mode from `os.environ.get("SIA_TOOLSET", "standard")` inside
   `run_agent_pydantic_ai` and pass it through. (Default `standard` preserves all
   existing tests.)
2. `tests/test_verified.py` (or a new `tests/test_toolset.py`): assert
   `len(_make_tools(d, "minimal")) == 1`, `== 3` for standard, `== 9` for overloaded;
   assert a decoy returns its canned string and writes nothing.

**Run matrix (9 runs):**
| variant | run_ids | env |
|---|---|---|
| minimal | 101,102,103 | `SIA_TOOLSET=minimal` |
| standard | (reuse 41–45 or 104,105,106) | `SIA_TOOLSET=standard` |
| overloaded | 107,108,109 | `SIA_TOOLSET=overloaded` |

(Prefer fresh `standard` runs 104–106 for a matched 3-seed cell rather than reusing 41–45,
which used a slightly different early-stop.)

**Hypotheses:**
- Survey prediction: `overloaded` < `standard` (tool-selection confusion).
- Plausible null: our meta-agent mostly calls `write_file`; decoys may be ignored →
  no effect. A *null contra the survey* is itself a finding (tool overload doesn't bite
  when the task has an obvious tool).
- Diagnostic: count decoy-tool invocations from the agent trajectory; if ~0, the null is
  mechanistic (the agent never took the bait), which is the cleaner story.

---

## Experiment C — Context engineering (Context Management dimension)

**Knob:** how much (and how relevant) context the meta-agent's prompt carries. Built in
`build_meta_prompt` (`sia/prompts.py`), which currently includes the task spec,
`sample_task_descriptions`, a `reference_section` (a full reference `target_agent.py`),
and a `sample_agent_execution` trajectory.

**Variants (env `SIA_CONTEXT`):**
- **`lean`** — strip the `reference_section` and the `sample_agent_execution` JSON; keep
  only the task spec + a one-line instruction. (Minimal context.)
- **`standard`** — full current prompt (reference + sample). Default.
- **`distractor`** — standard + an injected **irrelevant context block** (~600 tokens of
  plausible but off-topic ML lore: advice about image-augmentation, transformer LR
  schedules, RLHF, none relevant to a tabular CSV task). Operationalizes "irrelevant
  retrieval / context rot."

**Implementation:**
1. `sia/prompts.py`: in `build_meta_prompt`, read `mode = os.environ.get("SIA_CONTEXT","standard")`.
   - `lean`: set `reference_section = "(omitted)"` and replace the
     `json.dumps(task_files.sample_agent_execution, ...)` block with `"(omitted)"`.
   - `distractor`: append a constant `DISTRACTOR_BLOCK` string after the prompt body.
   Keep `standard` byte-identical to current (so the golden snapshot test is unaffected
   unless mode≠standard). Guard the snapshot: the snapshot test runs with no env set →
   `standard` → unchanged.
2. `tests/test_context_knob.py`: assert `lean` output excludes the reference seed text and
   the sample-execution JSON; `distractor` output contains the distractor marker;
   `standard` is unchanged vs the current builder.

**Run matrix (9 runs):**
| variant | run_ids | env |
|---|---|---|
| lean | 111,112,113 | `SIA_CONTEXT=lean` |
| standard | 114,115,116 | `SIA_CONTEXT=standard` |
| distractor | 117,118,119 | `SIA_CONTEXT=distractor` |

**Hypotheses:**
- Survey predictions: `standard` > `lean` (guidance helps); `distractor` < `standard`
  (irrelevant context rots).
- Plausible artifact: at n=3 with high per-candidate failure variance, any of these gaps
  could be manufactured. The diagnostic (prompt-token count) confirms the knob actually
  changed the context size.
- Note the reflexive risk: "more context helps" is exactly the kind of intuitive,
  cheap-to-show, hard-to-verify claim the paper warns about — so the power check is the
  point.

---

## Analysis & paper integration

1. Add a small analysis script `sia/knob_audit.py` that, given a {variant: [run_ids]}
   map, prints per-variant delivered accuracy (mean ± spread), yield, the naive max-gap,
   and the seeded delta — reusing the private-gold scorer from `plot_verified.py`.
2. For each knob, write the verdict (robust/artifact) with the power note.
3. Append two rows to `tab:knobaudit` in `paper/main.tex` (Tool exposure; Context
   richness/distractor), and add one sentence each to appendix obs (the harness-knob
   audit paragraph) summarizing the result.
4. If **any** knob shows a *robust* effect surviving seeds, that becomes a second
   positive harness-engineering result (beyond execution-gated selection) and is worth
   promoting; if all are artifacts, that strengthens the central thesis (only
   execution-binding survives) across **four** knobs (selection, budget, tools, context).

## Cost / sequencing
- 18 runs total (9 + 9), ~12 min each best-of-4 ⇒ ~3.5–4 h sequential.
- Run **after** pool-gen 91–96 completes. Suggested order: Tools first (clean mechanistic
  diagnostic), then Context.
- Honesty guardrails: n=3 per cell is underpowered by design — every reported gap gets a
  power note; null results are reported as findings, not omitted; the diagnostic
  (decoy-call counts / prompt tokens) must confirm the knob actually moved before any
  effect/null is interpreted.

## Scope notes (what we are NOT doing)
- Not auditing the survey's Subagent-Architecture or Safety/Isolation dimensions — out of
  scope for a tabular single-task selection harness.
- Not multi-generation (gen-1 best-of-N only), so the "raw-vs-summarized trace" context
  variant (which needs the feedback loop) is deferred; noted as future work.
