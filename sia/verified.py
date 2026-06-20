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


def select_best(candidates: list[Candidate]) -> Candidate | None:
    """Return the candidate with the highest val score; None if all failed."""
    scored = [c for c in candidates if c.val is not None]
    if not scored:
        return None
    return max(scored, key=lambda c: c.val)


def _truthy(value) -> bool:
    """Parse a label cell to bool: True/true/1/t/yes -> True, else False."""
    return str(value).strip().lower() in ("true", "1", "t", "yes")


def ensemble_predict(passer_submissions, out_path, id_col="PassengerId",
                     label_col="Transported") -> int:
    """Majority-vote True/False per id across gate-passing candidates' TEST submissions.

    passer_submissions: list of (submission_csv_path, val_score). Missing files are
    skipped. Ties are broken toward the highest-val passer's vote (members are sorted
    best-val first). Writes the ensembled submission to out_path and returns the number
    of members actually combined.
    """
    import collections
    import csv

    members = [(p, v) for p, v in passer_submissions if os.path.isfile(p)]
    members.sort(key=lambda pv: -(pv[1] or 0.0))  # best val first (tie-break order)
    votes: dict[str, list[bool]] = {}
    order: list[str] = []
    for path, _ in members:
        try:
            rows = list(csv.DictReader(open(path)))
        except Exception:
            continue
        for r in rows:
            i = r.get(id_col)
            if i is None:
                continue
            if i not in votes:
                votes[i] = []
                order.append(i)
            votes[i].append(_truthy(r.get(label_col)))
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([id_col, label_col])
        for i in order:
            vs = votes[i]
            counts = collections.Counter(vs).most_common()
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                lab = vs[0]            # tie -> best-val passer's vote (vs is best-val first)
            else:
                lab = counts[0][0]
            w.writerow([i, "True" if lab else "False"])
    return len(members)


import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    gen: int
    candidates: list[Candidate]
    best: Candidate | None


def run_verified_generation(gen, n, cand_root, oracle_val_csv,
                            produce_target, run_target, label_col, id_col,
                            early_stop_threshold, triage_mode="lint",
                            repair_retries=0, no_early_stop=False):
    """Sample up to n candidates for one generation, score each on the val oracle,
    optionally repair failures, and select the best.

    produce_target(gen, k, cand_dir[, repair_context]) -> target_path
        Writes the candidate. On a repair attempt it is called with a non-None
        repair_context (a short string describing the prior failure) so the producer
        can feed the executable error back to the model.
    run_target(cand_dir, k) -> None    Executes the target; writes its outputs.

    A candidate that fails to produce, is triage-rejected, fails to run, or yields no
    scorable val file is retried up to `repair_retries` times (default 0 = no repair),
    then recorded as None-scored (never raises). With `no_early_stop=True`, all n
    candidates run regardless of score (used for ensembling, which needs >=2 members).
    `triage_mode`: "lint" rejects known-fatal targets before execution; "off" disables.
    """
    candidates: list[Candidate] = []
    for k in range(n):
        cand_dir = os.path.join(cand_root, f"cand_{k}")
        os.makedirs(cand_dir, exist_ok=True)
        sub = os.path.join(cand_dir, "submission.csv")
        target_path, val, last_err = "", None, None
        for attempt in range(1 + max(0, repair_retries)):
            try:
                if attempt == 0:
                    target_path = produce_target(gen, k, cand_dir)
                else:
                    logger.info("gen %s cand %s: repair attempt %s (%s)", gen, k, attempt, last_err)
                    target_path = produce_target(gen, k, cand_dir, repair_context=last_err)
            except Exception as exc:
                last_err = f"produce failed: {exc}"
                logger.warning("gen %s cand %s: %s", gen, k, last_err)
                continue
            if triage_mode != "off":
                issues = lint_target(target_path)
                if issues:
                    last_err = "triage rejected: " + "; ".join(issues)
                    logger.info("gen %s cand %s: %s", gen, k, last_err)
                    continue
            try:
                run_target(cand_dir, k)
            except Exception as exc:
                last_err = f"run failed: {exc}"
                logger.warning("gen %s cand %s: %s", gen, k, last_err)
                continue
            val = score_val(cand_dir, oracle_val_csv, label_col=label_col, id_col=id_col)
            if val is None:
                last_err = "ran but produced no scorable val_predictions.csv"
                continue
            break  # success
        candidates.append(Candidate(gen, k, val, target_path, sub))
        if val is not None and not no_early_stop and val >= early_stop_threshold:
            logger.info("gen %s cand %s: early-stop at val=%.4f", gen, k, val)
            break
    return GenerationResult(gen=gen, candidates=candidates, best=select_best(candidates))
