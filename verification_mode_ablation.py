"""Verification-mode ablation: how you SELECT among best-of-N candidates.

For each run with >=2 candidate target scripts, compare four selection modes on the
private-gold accuracy of the chosen candidate:
  none   - take the first candidate (no selection / self-report)
  judge  - a model reads the N candidate scripts and picks one (model-as-judge)
  gate   - argmax validation score (executable oracle)
  oracle - argmax private-gold accuracy (hindsight ceiling, not deployable)
Also report the mean over candidates (= expected random pick).

This isolates the verifier-paradox reconciliation: does selecting by EXECUTION (gate)
beat selecting by MODEL JUDGMENT (judge)? Small-n, illustrative.
"""
import csv
import glob
import json
import os
import re
import statistics
import urllib.request

import sys
sys.path.insert(0, ".")
from sia.verified import score_val

GOLD = "sia/tasks/spaceship-titanic/data/private/test.csv"
JUDGE_MODEL = "qwen3-coder:30b"
RUNS = [41, 70, 71, 72, 73]


def load(p):
    return {r["PassengerId"]: str(r.get("Transported", "")).strip().lower()
            for r in csv.DictReader(open(p))}


def gold_acc(sub):
    g = load(GOLD); p = load(sub); c = set(g) & set(p)
    return sum(g[k] == p[k] for k in c) / len(c) if c else None


def model_judge(scripts):
    """scripts: list of (idx, code). Returns chosen idx via one Ollama call."""
    parts = []
    for idx, code in scripts:
        parts.append(f"=== CANDIDATE {idx} ===\n{code[:1600]}")
    prompt = (
        "You are selecting the best machine-learning pipeline. Below are several "
        "candidate Python scripts that each train a model on spaceship-titanic and "
        "write predictions. Judge which ONE will generalize best to held-out test data "
        "(consider preprocessing, model choice, leakage, robustness). Reply with ONLY "
        "the integer index of the single best candidate.\n\n" + "\n\n".join(parts) +
        "\n\nBest candidate index:")
    body = json.dumps({
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=180).read())
    content = txt["choices"][0]["message"]["content"]
    m = re.findall(r"\d+", content)
    idxs = [i for i, _ in scripts]
    return int(m[-1]) if m and int(m[-1]) in idxs else idxs[0]


def main():
    rows = []
    for rid in RUNS:
        base = f"runs/run_{rid}"
        oracle_val = f"{base}/_oracle/val.csv"
        cands = []
        for c in sorted(glob.glob(f"{base}/gen_1/cand_*")):
            sub = f"{c}/submission.csv"
            if not os.path.isfile(sub):
                continue
            idx = int(c.rsplit("_", 1)[1])
            test = gold_acc(sub)
            val = score_val(c, oracle_val, label_col="Transported", id_col="PassengerId")
            code = open(f"{c}/target_agent.py").read() if os.path.isfile(f"{c}/target_agent.py") else ""
            if test is not None:
                cands.append({"idx": idx, "test": test, "val": val, "code": code, "sub": sub})
        if len(cands) < 2:
            continue
        none_pick = sorted(cands, key=lambda x: x["idx"])[0]
        gated = [c for c in cands if c["val"] is not None]
        gate_pick = max(gated, key=lambda x: x["val"]) if gated else none_pick
        oracle_pick = max(cands, key=lambda x: x["test"])
        judge_idx = model_judge([(c["idx"], c["code"]) for c in cands])
        judge_pick = next((c for c in cands if c["idx"] == judge_idx), none_pick)
        rand = statistics.mean(c["test"] for c in cands)
        rows.append({
            "run": rid, "n": len(cands),
            "none": none_pick["test"], "random": rand,
            "judge": judge_pick["test"], "gate": gate_pick["test"],
            "oracle": oracle_pick["test"],
        })
        print(f"run_{rid} (n={len(cands)}): none={none_pick['test']:.3f} "
              f"random={rand:.3f} judge={judge_pick['test']:.3f}(idx{judge_idx}) "
              f"gate={gate_pick['test']:.3f} oracle={oracle_pick['test']:.3f}")

    print("\n=== mean private-gold accuracy by selection mode ===")
    for k in ["none", "random", "judge", "gate", "oracle"]:
        vals = [r[k] for r in rows]
        print(f"  {k:7s} {statistics.mean(vals):.4f}   (per-run: {[round(v,3) for v in vals]})")
    # gate vs judge paired
    dj = [r["gate"] - r["judge"] for r in rows]
    print(f"\n  gate - judge (paired): {[round(d,3) for d in dj]}  mean={statistics.mean(dj):+.4f}")
    print(f"  gate - random (paired): {[round(r['gate']-r['random'],3) for r in rows]}  "
          f"mean={statistics.mean([r['gate']-r['random'] for r in rows]):+.4f}")
    json.dump(rows, open("runs/verification_mode.json", "w"), indent=2)
    print("\nwrote runs/verification_mode.json")


if __name__ == "__main__":
    main()
