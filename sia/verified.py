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
