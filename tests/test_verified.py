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
