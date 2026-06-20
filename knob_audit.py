"""Harness-knob audit analysis: per-variant delivered accuracy (private gold), yield,
naive max-gap vs seeded delta. Usage: python3 knob_audit.py

Reads the verified deliverable runs/run_<id>/submission.csv for each variant's seeds.
"""
import csv
import os
import statistics

GOLD = "sia/tasks/spaceship-titanic/data/private/test.csv"

# experiment -> {variant: [run_ids]}; "standard" is the shared baseline cell.
EXPERIMENTS = {
    "Tool exposure": {
        "minimal": [101, 102, 103],
        "standard": [104, 105, 106],
        "overloaded": [107, 108, 109],
    },
    "Context": {
        "lean": [111, 112, 113],
        "standard": [104, 105, 106],
        "distractor": [117, 118, 119],
    },
}


def load(p):
    return {r["PassengerId"]: str(r.get("Transported", "")).strip().lower()
            for r in csv.DictReader(open(p))}


def acc(rid):
    sub = f"runs/run_{rid}/submission.csv"
    if not os.path.isfile(sub):
        return None
    g = load(GOLD); p = load(sub); c = set(g) & set(p)
    return sum(g[k] == p[k] for k in c) / len(c) if c else None


def main():
    for exp, variants in EXPERIMENTS.items():
        print(f"\n=== {exp} ===")
        means = {}
        for v, ids in variants.items():
            accs = [acc(r) for r in ids]
            comp = [a for a in accs if a is not None and a > 0.5]
            m = statistics.mean(comp) if comp else float("nan")
            means[v] = comp
            shown = [round(a, 3) if a is not None else "none" for a in accs]
            mstr = f"{m:.4f}" if comp else "n/a"
            print(f"  {v:11s} seeds={shown}  mean(comp)={mstr}  yield={len(comp)}/{len(ids)}")
        # naive vs seeded between the two non-standard variants and standard
        nonstd = [v for v in variants if v != "standard"]
        base = means.get("standard", [])
        for v in nonstd:
            vv = means.get(v, [])
            if vv and base:
                naive = max(vv) - max(base)        # most striking single-run gap
                seeded = statistics.mean(vv) - statistics.mean(base)
                print(f"    {v} vs standard:  naive(max-max)={naive:+.3f}   "
                      f"seeded(mean-mean)={seeded:+.3f}")
            else:
                print(f"    {v} vs standard:  insufficient competitive runs "
                      f"(v={len(vv)}, std={len(base)})")


if __name__ == "__main__":
    main()
