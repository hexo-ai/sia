# Verified-SIA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SIA self-improvement reliable on small local models by sampling N target candidates per generation, scoring each on a self-held-out validation split with an executable oracle, and keeping the best (never regressing).

**Architecture:** All new logic lives in one isolated, pure-where-possible module `sia/verified.py` (validation split, executable scorer, best-of-N selection, keep-best gate, deterministic linter, and an orchestration unit that takes injected `produce`/`run` callbacks so it is unit-testable with mock targets). `sia/orchestrator.py` gains a thin wiring change that supplies the real meta/feedback-agent production and subprocess target-run callbacks, gated on config. `sia/config.py` and `sia/prompts.py` get small additive changes (config fields + the `val_predictions.csv` output contract).

**Tech Stack:** Python 3.11, pytest, pandas (already a venv dep), the existing SIA harness (`sia/orchestrator.py`, `sia/layout.py`, `sia/run_setup.py`, `sia/config.py`, `sia/prompts.py`).

**Spec:** `docs/superpowers/specs/2026-06-19-verified-sia-design.md`

---

## File Structure

- **Create** `sia/verified.py` — all Verified-SIA logic:
  - `make_val_split(...)` — split labeled train into inner-train + val (features) + held-out labels.
  - `score_val(...)` — executable oracle: accuracy of a candidate's `val_predictions.csv` vs held-out labels.
  - `lint_target(...)` — deterministic pre-execution linter.
  - `select_best(...)` — best-of-N selection (argmax val, ignore failures).
  - `Incumbent`, `update_incumbent(...)` — keep-best gate (strict improvement).
  - `Candidate`, `GenerationResult`, `run_verified_generation(...)` — orchestration unit with injected callbacks.
- **Create** `tests/test_verified.py` — unit + integration tests for the above.
- **Modify** `sia/config.py` — add `BEST_OF_N`, `EARLY_STOP_THRESHOLD`, `VAL_FRACTION`, `TRIAGE_MODE` fields + env mappings.
- **Modify** `sia/prompts.py` — add the `val_predictions.csv` output contract to the local-endpoint adaptation block.
- **Modify** `sia/orchestrator.py` — wire `run_verified_generation` into the main loop, gated on `cfg.BEST_OF_N > 1`.

Note: the `sia/` repo is not yet under git. Task 0 initializes it so the TDD commit steps work (local-only; independent of the parent repo that gitignores `sia/`).

---

### Task 0: Initialize local git for the SIA repo

**Files:**
- Create: `sia/.gitignore`

- [ ] **Step 1: Initialize git and ignore run/venv artifacts**

```bash
cd /Users/jim/Desktop/judge-experiments-papers/sia
git init
printf 'runs/\n*.pyc\n__pycache__/\n.venv/\nvenv/\n*.egg-info/\n.env\n' > .gitignore
git add .gitignore
git commit -m "chore: init local git for verified-sia work"
```

Expected: a new git repo with one commit. `git status` shows a clean tree.

---

### Task 1: Config fields + env overrides

**Files:**
- Modify: `sia/config.py` (Config dataclass fields near line 29–52; env map near line 80–88)
- Test: `tests/test_verified.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verified.py
import os
from sia.config import Config


def test_config_verified_defaults():
    cfg = Config()
    assert cfg.BEST_OF_N == 4
    assert cfg.EARLY_STOP_THRESHOLD == 0.78
    assert cfg.VAL_FRACTION == 0.2
    assert cfg.TRIAGE_MODE == "lint"


def test_config_verified_env(monkeypatch):
    monkeypatch.setenv("SIA_BEST_OF_N", "3")
    monkeypatch.setenv("SIA_EARLY_STOP_THRESHOLD", "0.7")
    monkeypatch.setenv("SIA_VAL_FRACTION", "0.25")
    monkeypatch.setenv("SIA_TRIAGE", "off")
    cfg = Config.from_env()
    assert cfg.BEST_OF_N == 3
    assert cfg.EARLY_STOP_THRESHOLD == 0.7
    assert cfg.VAL_FRACTION == 0.25
    assert cfg.TRIAGE_MODE == "off"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sia && python -m pytest tests/test_verified.py -q`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'BEST_OF_N'`.

- [ ] **Step 3: Add fields and env mappings**

In `sia/config.py`, add these fields to the `Config` dataclass (after `EVAL_TIMEOUT`, before the sandbox block):

```python
    # ---- Verified-SIA (best-of-N, execution-gated, keep-best) ----
    BEST_OF_N: int = 4
    EARLY_STOP_THRESHOLD: float = 0.78
    VAL_FRACTION: float = 0.2
    TRIAGE_MODE: str = "lint"  # "lint" | "off" | "judge"
```

In the `env_map` dict (near line 80), add:

```python
            "SIA_BEST_OF_N": ("BEST_OF_N", int),
            "SIA_EARLY_STOP_THRESHOLD": ("EARLY_STOP_THRESHOLD", float),
            "SIA_VAL_FRACTION": ("VAL_FRACTION", float),
            "SIA_TRIAGE": ("TRIAGE_MODE", str),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sia && python -m pytest tests/test_verified.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd sia && git add sia/config.py tests/test_verified.py
git commit -m "feat(verified): config fields for best-of-N, gate, val split"
```

---

### Task 2: Validation split

**Files:**
- Create: `sia/verified.py`
- Test: `tests/test_verified.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_verified.py
import pandas as pd
from pathlib import Path
from sia import verified


def _write_train(p: Path, n=100):
    df = pd.DataFrame({
        "PassengerId": [f"{i:04d}_01" for i in range(n)],
        "Feat": list(range(n)),
        "Transported": [i % 2 == 0 for i in range(n)],
    })
    df.to_csv(p, index=False)
    return df


def test_make_val_split(tmp_path):
    train = tmp_path / "train.csv"
    _write_train(train, 100)
    agent_dir = tmp_path / "agent_data"
    oracle_dir = tmp_path / "_oracle"
    info = verified.make_val_split(
        train_csv=str(train), agent_data_dir=str(agent_dir),
        oracle_dir=str(oracle_dir), label_col="Transported",
        id_col="PassengerId", frac=0.2, seed=7,
    )
    inner = pd.read_csv(agent_dir / "train_inner.csv")
    val_feat = pd.read_csv(agent_dir / "val_features.csv")
    val_lab = pd.read_csv(oracle_dir / "val.csv")
    # 20% held out, disjoint, deterministic count
    assert len(val_feat) == 20
    assert len(inner) == 80
    assert "Transported" not in val_feat.columns          # labels stripped
    assert "Transported" in val_lab.columns               # labels retained for oracle
    assert set(val_feat["PassengerId"]) == set(val_lab["PassengerId"])
    assert set(inner["PassengerId"]).isdisjoint(set(val_feat["PassengerId"]))
    assert info["n_val"] == 20 and info["n_inner"] == 80
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sia && python -m pytest tests/test_verified.py::test_make_val_split -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sia.verified'`.

- [ ] **Step 3: Create `sia/verified.py` with `make_val_split`**

```python
# sia/verified.py
"""Verified-SIA: best-of-N, execution-gated, keep-best self-improvement.

All "is this better / does it work" decisions are made by an executable oracle
scoring a self-held-out validation split — never by a model's judgement.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd


def make_val_split(train_csv: str, agent_data_dir: str, oracle_dir: str,
                   label_col: str, id_col: str, frac: float = 0.2,
                   seed: int = 7) -> dict:
    """Split a labeled training CSV into inner-train + validation.

    Writes (for the agent to see): <agent_data_dir>/train_inner.csv (labeled) and
    <agent_data_dir>/val_features.csv (label column dropped).
    Writes (harness-only): <oracle_dir>/val.csv (id + label).
    Returns {"n_inner", "n_val"}.
    """
    os.makedirs(agent_data_dir, exist_ok=True)
    os.makedirs(oracle_dir, exist_ok=True)
    df = pd.read_csv(train_csv)
    val = df.sample(frac=frac, random_state=seed)
    inner = df.drop(val.index)
    inner.to_csv(os.path.join(agent_data_dir, "train_inner.csv"), index=False)
    val.drop(columns=[label_col]).to_csv(
        os.path.join(agent_data_dir, "val_features.csv"), index=False)
    val[[id_col, label_col]].to_csv(os.path.join(oracle_dir, "val.csv"), index=False)
    return {"n_inner": int(len(inner)), "n_val": int(len(val))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sia && python -m pytest tests/test_verified.py::test_make_val_split -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd sia && git add sia/verified.py tests/test_verified.py
git commit -m "feat(verified): make_val_split with harness-held labels"
```

---

### Task 3: Executable scorer (the oracle)

**Files:**
- Modify: `sia/verified.py`
- Test: `tests/test_verified.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_verified.py
def test_score_val_exact(tmp_path):
    oracle = tmp_path / "_oracle"; oracle.mkdir()
    pd.DataFrame({"PassengerId": ["a", "b", "c", "d"],
                  "Transported": [True, True, False, False]}).to_csv(oracle / "val.csv", index=False)
    cand = tmp_path / "cand_1"; cand.mkdir()
    # 3/4 correct (last one wrong)
    pd.DataFrame({"PassengerId": ["a", "b", "c", "d"],
                  "Transported": [True, True, False, True]}).to_csv(cand / "val_predictions.csv", index=False)
    acc = verified.score_val(str(cand), str(oracle / "val.csv"),
                             label_col="Transported", id_col="PassengerId")
    assert acc == 0.75


def test_score_val_missing_returns_none(tmp_path):
    oracle = tmp_path / "_oracle"; oracle.mkdir()
    pd.DataFrame({"PassengerId": ["a"], "Transported": [True]}).to_csv(oracle / "val.csv", index=False)
    cand = tmp_path / "cand_2"; cand.mkdir()  # no val_predictions.csv
    assert verified.score_val(str(cand), str(oracle / "val.csv"),
                              label_col="Transported", id_col="PassengerId") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sia && python -m pytest tests/test_verified.py -k score_val -q`
Expected: FAIL — `AttributeError: module 'sia.verified' has no attribute 'score_val'`.

- [ ] **Step 3: Add `score_val`**

```python
# add to sia/verified.py
def _norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def score_val(cand_dir: str, oracle_val_csv: str, label_col: str,
              id_col: str) -> float | None:
    """Accuracy of <cand_dir>/val_predictions.csv vs held-out oracle labels.

    Returns None if the prediction file is missing, unparseable, empty, lacks the
    required columns, or shares no ids with the oracle (i.e. an unusable candidate).
    Label comparison is case/whitespace-insensitive so "True"/"true"/" TRUE " match,
    but "0"/"1" will NOT match "true"/"false" (a wrong label format scores ~0).
    """
    pred_path = os.path.join(cand_dir, "val_predictions.csv")
    if not os.path.isfile(pred_path):
        return None
    try:
        pred = pd.read_csv(pred_path)
        gold = pd.read_csv(oracle_val_csv)
    except Exception:
        return None
    if id_col not in pred.columns or label_col not in pred.columns:
        return None
    g = dict(zip(_norm(gold[id_col]), _norm(gold[label_col])))
    p = dict(zip(_norm(pred[id_col]), _norm(pred[label_col])))
    common = set(g) & set(p)
    if not common:
        return None
    return sum(g[k] == p[k] for k in common) / len(common)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sia && python -m pytest tests/test_verified.py -k score_val -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd sia && git add sia/verified.py tests/test_verified.py
git commit -m "feat(verified): executable val scorer (oracle)"
```

---

### Task 4: Keep-best gate (strict improvement)

**Files:**
- Modify: `sia/verified.py`
- Test: `tests/test_verified.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_verified.py
def test_update_incumbent_monotonic():
    inc = None
    path = []
    for gen, score in [(1, 0.80), (2, 0.74), (3, 0.82)]:
        cand = verified.Candidate(gen=gen, k=0, val=score,
                                  target_path=f"g{gen}", submission_path=f"s{gen}")
        inc = verified.update_incumbent(inc, cand)
        path.append(inc.val)
    assert path == [0.80, 0.80, 0.82]   # never decreases; tie/worse keeps incumbent


def test_update_incumbent_ignores_none():
    inc = verified.Candidate(gen=1, k=0, val=0.5, target_path="g1", submission_path="s1")
    nxt = verified.Candidate(gen=2, k=0, val=None, target_path="g2", submission_path="s2")
    assert verified.update_incumbent(inc, nxt) is inc
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sia && python -m pytest tests/test_verified.py -k incumbent -q`
Expected: FAIL — `AttributeError: module 'sia.verified' has no attribute 'Candidate'`.

- [ ] **Step 3: Add `Candidate` and `update_incumbent`**

```python
# add to sia/verified.py
@dataclass
class Candidate:
    gen: int
    k: int
    val: float | None
    target_path: str
    submission_path: str


def update_incumbent(incumbent: Candidate | None,
                     candidate: Candidate | None) -> Candidate | None:
    """Accept candidate as the new incumbent only on strict val improvement.

    None-scored candidates (failed/unusable) never replace the incumbent. Ties keep
    the earlier incumbent (avoids pointless churn). Guarantees a monotonic
    non-decreasing incumbent val across a run.
    """
    if candidate is None or candidate.val is None:
        return incumbent
    if incumbent is None or incumbent.val is None:
        return candidate
    return candidate if candidate.val > incumbent.val else incumbent
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sia && python -m pytest tests/test_verified.py -k incumbent -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd sia && git add sia/verified.py tests/test_verified.py
git commit -m "feat(verified): keep-best gate (strict improvement)"
```

---

### Task 5: Deterministic pre-execution linter

**Files:**
- Modify: `sia/verified.py`
- Test: `tests/test_verified.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_verified.py
PLAIN = '''import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv(args.dataset_dir + "/train_inner.csv")
m = RandomForestClassifier().fit(X, y)
out = pd.DataFrame({"PassengerId": ids, "Transported": preds.astype(bool)})
out.to_csv(working_dir + "/submission.csv", index=False)
out.to_csv(working_dir + "/val_predictions.csv", index=False)
'''

NESTED = '''from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1")
resp = client.chat.completions.create(model="m", messages=msgs)
# decides everything via the model, writes submission.csv eventually
'''

LABEL01 = '''import pandas as pd
out = pd.DataFrame({"PassengerId": ids, "Transported": preds.astype(int)})
out.to_csv("submission.csv"); out.to_csv("val_predictions.csv")
'''


def test_lint_passes_plain(tmp_path):
    f = tmp_path / "t.py"; f.write_text(PLAIN)
    assert verified.lint_target(str(f)) == []


def test_lint_flags_nested_agent(tmp_path):
    f = tmp_path / "t.py"; f.write_text(NESTED)
    issues = verified.lint_target(str(f))
    assert any("nested" in i for i in issues)
    assert any("val_predictions" in i for i in issues)  # also missing val output


def test_lint_flags_int_labels(tmp_path):
    f = tmp_path / "t.py"; f.write_text(LABEL01)
    assert any("astype(int)" in i or "label" in i for i in verified.lint_target(str(f)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sia && python -m pytest tests/test_verified.py -k lint -q`
Expected: FAIL — `AttributeError: module 'sia.verified' has no attribute 'lint_target'`.

- [ ] **Step 3: Add `lint_target`**

```python
# add to sia/verified.py
def lint_target(target_path: str) -> list[str]:
    """Flag known-fatal patterns in a generated target before executing it.

    Advisory only: execution + score_val remain the final arbiter. Returns a list of
    human-readable issue strings (empty == passes).
    """
    try:
        with open(target_path, "r", encoding="utf-8", errors="ignore") as fh:
            src = fh.read()
    except OSError:
        return ["unreadable target file"]
    issues: list[str] = []
    uses_chat = "chat.completions" in src
    imports_openai = re.search(r"\bfrom openai import\b|\bimport openai\b", src) is not None
    if uses_chat and imports_openai:
        issues.append("nested LLM-agent: calls chat.completions to decide (fails on "
                      "local endpoint); emit a plain executable script instead")
    if "submission.csv" not in src:
        issues.append("does not write submission.csv")
    if "val_predictions.csv" not in src:
        issues.append("does not write val_predictions.csv (required for the val oracle)")
    if re.search(r"Transported.{0,40}astype\(int\)", src, re.S):
        issues.append("label written as int (astype(int)); Transported must be True/False")
    return issues
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sia && python -m pytest tests/test_verified.py -k lint -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd sia && git add sia/verified.py tests/test_verified.py
git commit -m "feat(verified): deterministic pre-execution linter"
```

---

### Task 6: Best-of-N selection

**Files:**
- Modify: `sia/verified.py`
- Test: `tests/test_verified.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_verified.py
def test_select_best_argmax():
    cands = [
        verified.Candidate(1, 0, 0.70, "g1c0", "s0"),
        verified.Candidate(1, 1, None, "g1c1", "s1"),
        verified.Candidate(1, 2, 0.81, "g1c2", "s2"),
    ]
    best = verified.select_best(cands)
    assert best.k == 2 and best.val == 0.81


def test_select_best_all_failed_returns_none():
    cands = [verified.Candidate(1, 0, None, "g1c0", "s0")]
    assert verified.select_best(cands) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sia && python -m pytest tests/test_verified.py -k select_best -q`
Expected: FAIL — `AttributeError: module 'sia.verified' has no attribute 'select_best'`.

- [ ] **Step 3: Add `select_best`**

```python
# add to sia/verified.py
def select_best(candidates: list[Candidate]) -> Candidate | None:
    """Return the candidate with the highest val score; None if all failed."""
    scored = [c for c in candidates if c.val is not None]
    if not scored:
        return None
    return max(scored, key=lambda c: c.val)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sia && python -m pytest tests/test_verified.py -k select_best -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd sia && git add sia/verified.py tests/test_verified.py
git commit -m "feat(verified): best-of-N selection"
```

---

### Task 7: Orchestration unit with injected callbacks

**Files:**
- Modify: `sia/verified.py`
- Test: `tests/test_verified.py`

This is the unit that loops candidates, applies triage, runs, scores, early-stops, and selects — using injected `produce_target` and `run_target` callbacks so it is fully testable with mock targets (no real model calls).

- [ ] **Step 1: Write the failing integration test**

```python
# add to tests/test_verified.py
def _mock_oracle(tmp_path):
    oracle = tmp_path / "_oracle"; oracle.mkdir()
    pd.DataFrame({"PassengerId": ["a", "b", "c", "d"],
                  "Transported": [True, True, False, False]}).to_csv(oracle / "val.csv", index=False)
    return str(oracle / "val.csv")


def test_run_verified_generation_picks_best_and_early_stops(tmp_path):
    oracle_val = _mock_oracle(tmp_path)
    cand_root = tmp_path / "gen_1"; cand_root.mkdir()

    # produce_target writes a trivial target file (content irrelevant to the test)
    def produce_target(gen, k, cand_dir):
        p = os.path.join(cand_dir, "target_agent.py")
        with open(p, "w") as fh:
            fh.write("# mock target\nsubmission.csv val_predictions.csv\n")
        return p

    # run_target writes val_predictions of increasing accuracy per k:
    # k0 -> 2/4=0.5, k1 -> 4/4=1.0 (should early-stop here at threshold 0.78)
    preds_by_k = {
        0: [True, True, True, True],     # 0.5
        1: [True, True, False, False],   # 1.0
        2: [False, False, False, False], # 0.5 (must NOT run: early-stopped)
    }
    ran = []
    def run_target(cand_dir, k):
        ran.append(k)
        pd.DataFrame({"PassengerId": ["a", "b", "c", "d"],
                      "Transported": preds_by_k[k]}).to_csv(
            os.path.join(cand_dir, "val_predictions.csv"), index=False)
        pd.DataFrame({"PassengerId": ["x"], "Transported": [True]}).to_csv(
            os.path.join(cand_dir, "submission.csv"), index=False)

    result = verified.run_verified_generation(
        gen=1, n=3, cand_root=str(cand_root), oracle_val_csv=oracle_val,
        produce_target=produce_target, run_target=run_target,
        label_col="Transported", id_col="PassengerId",
        early_stop_threshold=0.78, triage_mode="off",
    )
    assert result.best.k == 1 and result.best.val == 1.0
    assert ran == [0, 1]           # early-stopped before k=2
    assert len(result.candidates) == 2


def test_run_verified_generation_triage_skips_execution(tmp_path):
    oracle_val = _mock_oracle(tmp_path)
    cand_root = tmp_path / "gen_1"; cand_root.mkdir()

    def produce_target(gen, k, cand_dir):
        p = os.path.join(cand_dir, "target_agent.py")
        with open(p, "w") as fh:                       # nested-agent: linter rejects
            fh.write("from openai import OpenAI\nclient.chat.completions.create()\n")
        return p

    ran = []
    def run_target(cand_dir, k):
        ran.append(k)

    result = verified.run_verified_generation(
        gen=1, n=2, cand_root=str(cand_root), oracle_val_csv=oracle_val,
        produce_target=produce_target, run_target=run_target,
        label_col="Transported", id_col="PassengerId",
        early_stop_threshold=0.78, triage_mode="lint",
    )
    assert ran == []               # never executed: all linter-rejected
    assert result.best is None
    assert all(c.val is None for c in result.candidates)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sia && python -m pytest tests/test_verified.py -k run_verified_generation -q`
Expected: FAIL — `AttributeError: module 'sia.verified' has no attribute 'run_verified_generation'`.

- [ ] **Step 3: Add `GenerationResult` and `run_verified_generation`**

```python
# add to sia/verified.py
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    gen: int
    candidates: list[Candidate]
    best: Candidate | None


def run_verified_generation(gen, n, cand_root, oracle_val_csv,
                            produce_target, run_target, label_col, id_col,
                            early_stop_threshold, triage_mode="lint"):
    """Sample up to n candidates for one generation, score each on the val oracle,
    early-stop on threshold, and select the best.

    produce_target(gen, k, cand_dir) -> target_path   (writes the candidate)
    run_target(cand_dir, k) -> None                   (executes it; writes outputs)

    Any exception from produce/run, or a missing val file, yields a None-scored
    candidate (never raises). triage_mode: "lint" rejects known-fatal targets before
    execution; "off" disables; "judge" is reserved (falls back to lint).
    """
    candidates: list[Candidate] = []
    for k in range(n):
        cand_dir = os.path.join(cand_root, f"cand_{k}")
        os.makedirs(cand_dir, exist_ok=True)
        sub = os.path.join(cand_dir, "submission.csv")
        try:
            target_path = produce_target(gen, k, cand_dir)
        except Exception as exc:                       # production failure -> skip
            logger.warning("gen %s cand %s: produce failed: %s", gen, k, exc)
            candidates.append(Candidate(gen, k, None, "", sub))
            continue
        if triage_mode != "off":
            issues = lint_target(target_path)
            if issues:
                logger.info("gen %s cand %s: triage rejected: %s", gen, k, "; ".join(issues))
                candidates.append(Candidate(gen, k, None, target_path, sub))
                continue
        try:
            run_target(cand_dir, k)
        except Exception as exc:                       # execution failure -> skip
            logger.warning("gen %s cand %s: run failed: %s", gen, k, exc)
            candidates.append(Candidate(gen, k, None, target_path, sub))
            continue
        val = score_val(cand_dir, oracle_val_csv, label_col=label_col, id_col=id_col)
        candidates.append(Candidate(gen, k, val, target_path, sub))
        if val is not None and val >= early_stop_threshold:
            logger.info("gen %s cand %s: early-stop at val=%.4f", gen, k, val)
            break
    return GenerationResult(gen=gen, candidates=candidates, best=select_best(candidates))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sia && python -m pytest tests/test_verified.py -k run_verified_generation -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the whole module suite**

Run: `cd sia && python -m pytest tests/test_verified.py -q`
Expected: PASS (all tests so far).

- [ ] **Step 6: Commit**

```bash
cd sia && git add sia/verified.py tests/test_verified.py
git commit -m "feat(verified): generation runner with best-of-N, triage, early-stop"
```

---

### Task 8: Prompt contract — require `val_predictions.csv`

**Files:**
- Modify: `sia/prompts.py` (the adaptation block in `build_target_client_setup`, near line 725)
- Test: `tests/test_verified.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_verified.py
from sia.prompts import build_target_client_setup
from sia.providers import Provider


def _ollama_provider():
    return Provider(provider_id="ollama", name="Ollama", client_kind="openai",
                    base_url="http://localhost:11434/v1", api_key_env="OLLAMA_API_KEY")


def test_prompt_requires_val_predictions(monkeypatch):
    monkeypatch.setenv("SIA_LOCAL_ADAPT", "1")
    block = build_target_client_setup(_ollama_provider(), "qwen3-coder:30b")
    assert "val_predictions.csv" in block
    assert "val_features.csv" in block
    assert "train_inner.csv" in block
```

Note: confirm the `Provider` import path by checking `sia/providers.py` (adjust the import in the test if the dataclass lives elsewhere, e.g. `from sia.providers import Provider`).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sia && python -m pytest tests/test_verified.py -k val_predictions -q`
Expected: FAIL — assertion error (`val_predictions.csv` not in block).

- [ ] **Step 3: Extend the adaptation block**

In `sia/prompts.py`, inside the `adaptation_block` string in `build_target_client_setup` (after rule 2, before the closing `"""`), append a third rule:

```python

3. VALIDATION OUTPUT CONTRACT (REQUIRED). The dataset directory contains
   `train_inner.csv` (labeled training data), `val_features.csv` (held-out rows WITHOUT
   the label column), and the usual test input. Your target_agent.py MUST:
   - train ONLY on `train_inner.csv` (do not peek at any other labels),
   - write `submission.csv` for the test set as before, AND
   - ALSO write `val_predictions.csv` to the working dir, predicting every row in
     `val_features.csv`, in the SAME format as submission.csv (id column + the label
     column as True/False). The harness scores `val_predictions.csv` to decide whether
     your solution is kept; a missing or mis-formatted file means your solution is
     discarded.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sia && python -m pytest tests/test_verified.py -k val_predictions -q`
Expected: PASS.

- [ ] **Step 5: Update the prompt snapshot test if present**

Run: `cd sia && python -m pytest tests/test_prompts_snapshot.py -q`
If it fails because the snapshot changed, regenerate per that test's documented update mechanism (commonly `pytest --snapshot-update` or an env flag noted at the top of `tests/test_prompts_snapshot.py`), then re-run to green.

- [ ] **Step 6: Commit**

```bash
cd sia && git add sia/prompts.py tests/test_verified.py tests/__snapshots__ 2>/dev/null; git add -A
git commit -m "feat(verified): require val_predictions.csv output contract"
```

---

### Task 9: Wire Verified-SIA into the orchestrator + run-setup

**Files:**
- Modify: `sia/run_setup.py` (in `setup_run_directory`, after the venv/profiles setup near line 184) — build the val split when enabled.
- Modify: `sia/orchestrator.py` (the main loop near line 920–945) — call `run_verified_generation` when `cfg.BEST_OF_N > 1`.
- Test: `tests/test_verified.py` (an end-to-end test with a fake target, no model calls).

This task connects the unit to real meta/feedback production and subprocess execution. Read `run_generation` (`sia/orchestrator.py:700`) and the target-run command (`sia/orchestrator.py:391`: `[python_exec, "-u", target_agent_path, "--dataset_dir", abs_dataset_dir, "--working_dir", gen_dir]`) before wiring.

- [ ] **Step 1: Write the failing end-to-end test (fake target, real subprocess)**

```python
# add to tests/test_verified.py
import subprocess, sys, textwrap


def test_end_to_end_subprocess_target(tmp_path):
    # dataset dir the "agent" reads
    data = tmp_path / "data"; data.mkdir()
    pd.DataFrame({"PassengerId": ["a", "b", "c", "d"],
                  "Feat": [0, 1, 0, 1]}).to_csv(data / "val_features.csv", index=False)
    oracle_val = _mock_oracle(tmp_path)
    cand_root = tmp_path / "gen_1"; cand_root.mkdir()

    # A real, plain-script target: copies the rule "Feat==0 -> True" (3/4 correct here:
    # a=0->T(ok), b=1->F(ok? gold b=True -> wrong), c=0->T(gold False -> wrong), d=1->F(gold False ok))
    target_src = textwrap.dedent('''
        import argparse, pandas as pd
        p = argparse.ArgumentParser(); p.add_argument("--dataset_dir"); p.add_argument("--working_dir")
        a = p.parse_args()
        vf = pd.read_csv(a.dataset_dir + "/val_features.csv")
        vf["Transported"] = (vf["Feat"] == 0)
        vf[["PassengerId", "Transported"]].to_csv(a.working_dir + "/val_predictions.csv", index=False)
        vf[["PassengerId", "Transported"]].to_csv(a.working_dir + "/submission.csv", index=False)
    ''')

    def produce_target(gen, k, cand_dir):
        path = os.path.join(cand_dir, "target_agent.py")
        with open(path, "w") as fh:
            fh.write(target_src)
        return path

    def run_target(cand_dir, k):
        subprocess.run([sys.executable, os.path.join(cand_dir, "target_agent.py"),
                        "--dataset_dir", str(data), "--working_dir", cand_dir],
                       check=True, capture_output=True, timeout=60)

    result = verified.run_verified_generation(
        gen=1, n=1, cand_root=str(cand_root), oracle_val_csv=oracle_val,
        produce_target=produce_target, run_target=run_target,
        label_col="Transported", id_col="PassengerId",
        early_stop_threshold=0.99, triage_mode="lint",
    )
    assert result.best is not None
    assert result.best.val == 0.5    # 2/4 correct given the gold above
    assert os.path.isfile(os.path.join(cand_root, "cand_0", "submission.csv"))
```

- [ ] **Step 2: Run test to verify it fails, then passes as written**

Run: `cd sia && python -m pytest tests/test_verified.py -k end_to_end -q`
Expected: PASS once Task 7 is in (this test exercises the real subprocess path through the existing unit; if it errors, fix the test's gold/label arithmetic, not the module).

- [ ] **Step 3: Build the val split in `setup_run_directory`**

In `sia/run_setup.py`, after profiles are written (near line 184), add (guarded so non-spaceship tasks are unaffected):

```python
    # Verified-SIA: build a self-held-out validation split from labeled training data.
    if cfg.BEST_OF_N > 1:
        from sia.verified import make_val_split
        train_csv = os.path.join(task_dir, "data", "public", "train.csv")
        if os.path.isfile(train_csv):
            oracle_dir = os.path.join(run_directory, "_oracle")
            try:
                info = make_val_split(
                    train_csv=train_csv,
                    agent_data_dir=os.path.join(task_dir, "data", "public"),
                    oracle_dir=oracle_dir, label_col="Transported",
                    id_col="PassengerId", frac=cfg.VAL_FRACTION,
                )
                logger.info("Verified-SIA val split: %s inner / %s val",
                            info["n_inner"], info["n_val"])
            except Exception as exc:
                logger.warning("Verified-SIA val split skipped: %s", exc)
```

Note: this writes `train_inner.csv`/`val_features.csv` next to the task's public data. The `label_col`/`id_col` are spaceship-titanic specific; generalizing is future work (see spec Non-Goals).

- [ ] **Step 4: Add two concrete helpers to `sia/orchestrator.py`**

Scope decision (YAGNI): best-of-N runs at **generation 1** (where the 5-seed study
showed runs die for lack of any valid solution — production there is just the
meta-agent, fully concrete). Generations > 1 use the existing single feedback
candidate, but are still **execution-gated**: their target is scored on the val oracle
and only updates the incumbent on strict improvement. Gen>1 best-of-N is explicit
follow-up (it needs per-candidate feedback context — see "Follow-up" below).

Add these two helpers near the other module-level helpers in `sia/orchestrator.py`
(after `_run_target_agent`). They reuse the primitives confirmed at
`orchestrator.py:355` (`_run_target_agent`) and SECTION 4 (`build_meta_prompt` +
`run_agent`), both already imported in the module.

```python
def _produce_meta_candidate(cand_dir, task_files, task_model, meta_model, agent_impl,
                            meta_profile, target_provider, env_config, focus, cand_k):
    """Generation-1 production: write one meta-agent candidate target into cand_dir.

    Mirrors SECTION 4 but targets cand_dir and varies sampling per candidate via a
    one-line nonce so the N candidates differ. Returns the target_agent.py path.
    """
    prompt = build_meta_prompt(
        task_files, task_model, cand_dir, provider=target_provider, focus=focus,
    ) + f"\n# candidate {cand_k}\n"
    write_text(os.path.join(cand_dir, Names.META_PROMPT), prompt)
    asyncio.run(run_agent(
        model_name=meta_model, max_turns=str(env_config.DEFAULT_MAX_TURNS),
        prompt=prompt, agent_working_directory=cand_dir, agent_impl=agent_impl,
        provider=meta_profile.provider,
    ))
    return os.path.join(cand_dir, Names.TARGET_AGENT)


def _run_candidate_target(cand_dir, abs_dataset_dir, run_setup, env_config, sandbox):
    """Execute cand_dir/target_agent.py via the existing target-run primitive,
    writing submission.csv + val_predictions.csv into cand_dir. Raises on failure so
    run_verified_generation records a None-scored (failed) candidate."""
    target_path = os.path.join(cand_dir, Names.TARGET_AGENT)
    stdout_log = os.path.join(cand_dir, Names.STDOUT_LOG)
    success, _stdout, _stderr, err = _run_target_agent(
        venv_dir=run_setup.venv_dir, target_agent_path=target_path,
        abs_dataset_dir=abs_dataset_dir, gen_dir=cand_dir,
        stdout_log_file=stdout_log, sandbox=sandbox, env_config=env_config,
    )
    if not success:
        raise RuntimeError(err or "target agent failed")
```

- [ ] **Step 5: Wire the verified loop into the main loop**

In `sia/orchestrator.py`, replace the body of `for current_gen in range(1, max_gen + 1):`
(near line 920) so generation 1 uses best-of-N when enabled, and later generations are
scored+gated:

```python
    from sia.verified import (Candidate, run_verified_generation, score_val,
                              update_incumbent)
    oracle_val = os.path.join(run_setup.run_directory, "_oracle", "val.csv")
    use_verified = env_config.BEST_OF_N > 1 and os.path.isfile(oracle_val)
    incumbent = None

    for current_gen in range(1, max_gen + 1):
        logger.info("=" * 80)
        logger.info(f"Starting Generation {current_gen} of {max_gen}")
        logger.info("=" * 80)

        if use_verified and current_gen == 1:
            gen_root = RunLayout(run_setup.run_directory).gen_dir(1)

            def produce_target(gen, k, cand_dir):
                return _produce_meta_candidate(
                    cand_dir, task_files, task_model, meta_model, agent_impl,
                    meta_profile, target_provider, env_config, args.focus, k)

            def run_target(cand_dir, k):
                _run_candidate_target(cand_dir, abs_dataset_directory, run_setup,
                                      env_config, args.sandbox)

            result = run_verified_generation(
                gen=1, n=env_config.BEST_OF_N, cand_root=gen_root,
                oracle_val_csv=oracle_val, produce_target=produce_target,
                run_target=run_target, label_col="Transported", id_col="PassengerId",
                early_stop_threshold=env_config.EARLY_STOP_THRESHOLD,
                triage_mode=env_config.TRIAGE_MODE)
            incumbent = update_incumbent(incumbent, result.best)
            logger.info("Gen 1: best val=%s; incumbent val=%s",
                        result.best.val if result.best else None,
                        incumbent.val if incumbent else None)
            continue

        # Default path (gen>1, or verified disabled): existing single-candidate run.
        run_generation(
            current_gen=current_gen, max_gen=max_gen, run_setup=run_setup,
            task_files=task_files, abs_dataset_dir=abs_dataset_directory,
            dataset_dir=dataset_directory, meta_profile=meta_profile,
            sandbox=args.sandbox, env_config=env_config, task_model=task_model,
            target_provider=target_provider, focus=args.focus,
            training_sandbox=args.training_sandbox, resolved_ref=resolved_ref,
        )
        if use_verified:
            gen_dir = RunLayout(run_setup.run_directory).gen_dir(current_gen)
            val = score_val(gen_dir, oracle_val, label_col="Transported",
                            id_col="PassengerId")
            cand = Candidate(current_gen, 0, val,
                             os.path.join(gen_dir, Names.TARGET_AGENT),
                             os.path.join(gen_dir, "submission.csv"))
            incumbent = update_incumbent(incumbent, cand)
            logger.info("Gen %s: val=%s; incumbent val=%s", current_gen, val,
                        incumbent.val if incumbent else None)
```

Note `meta_model` and `agent_impl` are already in scope in `main()` (set near
`orchestrator.py:795` / passed to `setup_run_directory`). After the loop, if
`use_verified` and `incumbent` is not None, copy the incumbent's submission to the run
root as the deliverable:

```python
    if use_verified and incumbent is not None:
        import shutil
        shutil.copyfile(incumbent.submission_path,
                        os.path.join(run_setup.run_directory, "submission.csv"))
        logger.info("Deliverable: incumbent val=%.4f from %s",
                    incumbent.val, incumbent.target_path)
    elif use_verified:
        logger.warning("No competitive solution produced (no incumbent).")
```

**Follow-up (out of scope here):** gen>1 best-of-N requires building per-candidate
feedback context from the incumbent's prior run (`_build_feedback_context` +
`_run_feedback_agent` into each cand_dir). Track as a separate task.

- [ ] **Step 6: Run the full verified suite + a real smoke run**

Run: `cd sia && python -m pytest tests/test_verified.py -q`
Expected: PASS (all).

Smoke (real models, 1 seed, small N):
```bash
cd sia
OLLAMA_API_KEY=ollama SIA_PROVIDERS_DIR=$PWD/providers SIA_PROFILES_DIR=$PWD/profiles \
SIA_MAX_TURNS=50 SIA_BEST_OF_N=3 SIA_EARLY_STOP_THRESHOLD=0.78 \
sia run --task spaceship-titanic --target-agent-profile ollama-target \
  --meta-agent-profile ollama-meta --max_gen 2 --run_id 40 --sandbox none --no-web --log-level INFO
```
Expected: `runs/run_40/gen_1/cand_*/val_predictions.csv` exist; log shows per-gen best/incumbent val; the run does not crash even if some candidates fail.

- [ ] **Step 7: Commit**

```bash
cd sia && git add sia/orchestrator.py sia/run_setup.py tests/test_verified.py
git commit -m "feat(verified): wire best-of-N execution-gated loop into orchestrator"
```

---

### Task 10: Yield evaluation + plot

**Files:**
- Create: `sia/plot_verified.py` (reuse the scoring approach from `plot_sia_compare.py`)
- Test: manual (evaluation script)

- [ ] **Step 1: Run the matched evaluation (5 seeds × 2 models, verified on)**

```bash
cd sia
for prof in "ollama-meta ollama-target" "gemma-meta gemma-target"; do
  set -- $prof
  for rid in 41 42 43 44 45; do
    OLLAMA_API_KEY=ollama SIA_PROVIDERS_DIR=$PWD/providers SIA_PROFILES_DIR=$PWD/profiles \
    SIA_MAX_TURNS=50 SIA_BEST_OF_N=4 SIA_EARLY_STOP_THRESHOLD=0.78 \
    sia run --task spaceship-titanic --target-agent-profile $2 --meta-agent-profile $1 \
      --max_gen 3 --run_id $rid --sandbox none --no-web --log-level INFO > /tmp/verified_$rid.log 2>&1
    rid=$((rid+1))
  done
done
```
(Use distinct run_id ranges per model to avoid collisions; the loop above is illustrative — give qwen 41–45 and gemma 46–50.)

- [ ] **Step 2: Score incumbents on the private test gold and compare to baselines**

Write `sia/plot_verified.py` modeled on `plot_sia_compare.py`: for each verified run, take the incumbent's test `submission.csv`, score vs `sia/tasks/spaceship-titanic/data/private/test.csv`, and plot competitive-run yield for {default, adapted, verified} × {qwen, gemma}. Expected: verified yield ≈ 1.0 and no per-run regression.

- [ ] **Step 3: Commit**

```bash
cd sia && git add sia/plot_verified.py
git commit -m "feat(verified): yield evaluation + comparison plot"
```

---

## Self-Review

**Spec coverage:**
- Component A (val oracle + split + contract) → Tasks 2, 3, 8, 9-step3. ✓
- Component B (best-of-N + early-stop) → Tasks 6, 7. ✓
- Component C (keep-best gate) → Tasks 4, 9-step4. ✓
- Component D (linter, default) → Tasks 5, 7. ✓
- Config fields → Task 1. ✓
- Robustness (failures never crash the run) → Task 7 (try/except per candidate) + tests. ✓
- Testing (unit/integration/e2e) → Tasks 2–7 unit, 7 integration, 9 e2e, 10 real. ✓
- Success metrics (yield, monotonicity) → Task 4 test (monotonic) + Task 10 (yield). ✓

**Integration concreteness:** Task 9's helpers are now fully written — `_produce_meta_candidate` reuses `build_meta_prompt` + `run_agent`, and `_run_candidate_target` reuses the confirmed `_run_target_agent(...)` primitive (`orchestrator.py:355`). Best-of-N is scoped to generation 1 (concrete meta production); gen>1 is single-candidate but execution-gated. Gen>1 best-of-N is an explicitly tracked follow-up (needs `_build_feedback_context` + `_run_feedback_agent` per candidate). No `NotImplementedError` remains in the plan.

**Type consistency:** `Candidate(gen, k, val, target_path, submission_path)`, `GenerationResult(gen, candidates, best)`, `run_verified_generation(...)` signature, and `score_val(cand_dir, oracle_val_csv, label_col, id_col)` are used identically across Tasks 4, 6, 7, 9. ✓

**Scope:** spaceship-titanic only (label_col/id_col hardcoded with a guard); GPQA generalization is explicit future work. Single cohesive subsystem → one plan. ✓
