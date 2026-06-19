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
