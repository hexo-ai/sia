"""Verified-SIA yield: default vs local-adapted vs verified (best-of-N + gate).

For default/adapted runs we take the best accuracy across the run's generations
(submission per gen_*/submission.csv). For verified runs the deliverable is the
incumbent copied to the run root (runs/run_<id>/submission.csv). All scored on the
private test gold. "Competitive" = accuracy > 0.5.

Edit RUN_IDS below to the actual eval run_ids, then: python3 plot_verified.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOLD = "sia/tasks/spaceship-titanic/data/private/test.csv"

# (label, run_ids, kind, color). kind: "gens" = best over gen_*/submission.csv;
# "root" = the verified incumbent deliverable at runs/run_<id>/submission.csv.
GROUPS = [
    ("qwen\ndefault",  [13, 14, 15, 16, 17], "gens", "#9CA3AF"),
    ("qwen\nadapted",  [21, 22, 23],         "gens", "#60A5FA"),
    ("qwen\nverified", [41, 42, 43, 44, 45], "root", "#1D4ED8"),
    ("gemma\ndefault", [20, 27, 28],         "gens", "#9CA3AF"),
    ("gemma\nadapted", [24, 25, 26],         "gens", "#86EFAC"),
    ("gemma\nverified",[52, 53, 54, 55, 56], "root", "#15803D"),
]


def load(path):
    return {r["PassengerId"]: str(r.get("Transported", "")).strip().lower()
            for r in csv.DictReader(open(path))}


def accuracy(sub):
    gold = load(GOLD)
    pred = load(sub)
    common = set(gold) & set(pred)
    if not common:
        return None
    return sum(gold[k] == pred[k] for k in common) / len(common)


def best_for_run(rid, kind):
    base = Path(f"runs/run_{rid}")
    if not base.exists():
        return None
    if kind == "root":
        sub = base / "submission.csv"
        return accuracy(sub) if sub.exists() else None
    accs = []
    for g in sorted(base.glob("gen_*")):
        sub = g / "submission.csv"
        if sub.exists():
            a = accuracy(sub)
            if a is not None:
                accs.append(a)
    return max(accs) if accs else None


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 2]})
    labels, yields = [], []
    for i, (name, ids, kind, color) in enumerate(GROUPS):
        bests = [best_for_run(r, kind) for r in ids]
        present = [(r, b) for r, b in zip(ids, bests) if Path(f"runs/run_{r}").exists()]
        xs = [i + (j - len(ids) / 2) * 0.05 for j in range(len(ids))]
        for x, b in zip(xs, bests):
            ok = b is not None and b > 0.5
            if ok:
                ax1.scatter(x, b, marker="o", s=70, color=color, edgecolor="black",
                            linewidth=0.6, zorder=3)
            else:
                ax1.scatter(x, 0.0, marker="x", s=45, color=color, linewidth=1.4, zorder=3)
        n = len(present) or len(ids)
        n_comp = sum(1 for _, b in present if (b or 0) > 0.5)
        labels.append(name)
        yields.append(n_comp / n if n else 0.0)
    ax1.axhspan(0.79, 0.81, color="green", alpha=0.10)
    ax1.text(0.0, 0.815, "real pipelines ~0.79-0.82", color="green", fontsize=8)
    ax1.axhline(0.50, ls="--", color="gray", lw=1)
    ax1.text(0.0, 0.46, "failed run (x at 0)", color="gray", fontsize=8)
    ax1.set_xticks(range(len(GROUPS)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("accuracy on private test gold")
    ax1.set_ylim(-0.04, 1.0)
    ax1.set_title("Verified-SIA: best-of-N + execution gate vs. default / adapted")

    cols = [g[3] for g in GROUPS]
    ax2.bar(range(len(GROUPS)), yields, color=cols, alpha=0.85)
    for i, (name, ids, kind, color) in enumerate(GROUPS):
        present = [r for r in ids if Path(f"runs/run_{r}").exists()]
        n = len(present) or len(ids)
        n_comp = sum(1 for r in present if (best_for_run(r, kind) or 0) > 0.5)
        ax2.text(i, yields[i] + 0.02, f"{n_comp}/{n}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(range(len(GROUPS)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("competitive-run yield")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Yield by condition")

    fig.tight_layout()
    out = "runs/sia_verified.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)
    print("\n=== per-run best accuracy (private gold) ===")
    for name, ids, kind, color in GROUPS:
        row = " ".join(f"r{r}:{('%.3f'%b if b is not None else '-')}"
                       for r, b in zip(ids, [best_for_run(x, kind) for x in ids]))
        present = [r for r in ids if Path(f"runs/run_{r}").exists()]
        n = len(present) or len(ids)
        n_comp = sum(1 for r in present if (best_for_run(r, kind) or 0) > 0.5)
        print(f"  {name.replace(chr(10),' '):16s} {row}   yield={n_comp}/{n}")


if __name__ == "__main__":
    raise SystemExit(main())
