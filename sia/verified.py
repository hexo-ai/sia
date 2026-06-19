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
