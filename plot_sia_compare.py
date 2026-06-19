"""SIA-on-Ollama: default (nested-agent) vs local-adapted (plain-script) targets.

For each run we take the BEST accuracy across its 3 generations (a generation
counts only if it wrote a submission.csv that scores against the private gold
labels). "Competitive" = accuracy > 0.5 (above the ~0.50 random baseline; real
spaceship-titanic pipelines land ~0.79-0.82).

Usage: python3 plot_sia_compare.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOLD = "sia/tasks/spaceship-titanic/data/private/test.csv"

GROUPS = [
    ("qwen3-coder\ndefault",   [13, 14, 15, 16, 17], "#9CA3AF"),
    ("qwen3-coder\nadapted",   [21, 22, 23],         "#1D4ED8"),
    ("gemma4\ndefault",        [20, 27, 28],         "#9CA3AF"),
    ("gemma4\nadapted",        [24, 25, 26],         "#15803D"),
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


def best_acc(rid):
    base = Path(f"runs/run_{rid}")
    accs = []
    for g in sorted(base.glob("gen_*")):
        sub = g / "submission.csv"
        if sub.exists():
            a = accuracy(sub)
            if a is not None:
                accs.append(a)
    return max(accs) if accs else None


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 2]})

    labels, yields = [], []
    for i, (name, ids, color) in enumerate(GROUPS):
        bests = [best_acc(r) for r in ids]
        comp = [b for b in bests if b is not None and b > 0.5]
        # strip plot: each run a dot; no-competitive at y=0
        xs = [i + (j - len(ids) / 2) * 0.04 for j in range(len(ids))]
        for x, b in zip(xs, bests):
            y = b if (b is not None and b > 0.5) else 0.0
            ok = b is not None and b > 0.5
            if ok:
                ax1.scatter(x, y, marker="o", s=70, color=color,
                            edgecolor="black", linewidth=0.6, zorder=3)
            else:
                ax1.scatter(x, y, marker="x", s=70, color=color, linewidth=1.4, zorder=3)
        labels.append(name)
        yields.append(len(comp) / len(ids))
    ax1.axhspan(0.79, 0.81, color="green", alpha=0.10)
    ax1.text(0.0, 0.815, "real pipelines ~0.79-0.82", color="green", fontsize=8)
    ax1.axhline(0.50, ls="--", color="gray", lw=1)
    ax1.text(0.0, 0.46, "random 0.50 / failed run (x at 0)", color="gray", fontsize=8)
    ax1.set_xticks(range(len(GROUPS)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("best accuracy across 3 generations")
    ax1.set_ylim(-0.04, 1.0)
    ax1.set_title("SIA self-improvement on local models (spaceship-titanic)\n"
                  "default nested-agent target vs. plain-script directive")

    # yield bars
    bcols = [g[2] for g in GROUPS]
    ax2.bar(range(len(GROUPS)), yields, color=bcols, alpha=0.85)
    for i, (name, ids, color) in enumerate(GROUPS):
        n_comp = sum(1 for r in ids if (best_acc(r) or 0) > 0.5)
        ax2.text(i, yields[i] + 0.02, f"{n_comp}/{len(ids)}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks(range(len(GROUPS)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("yield: fraction reaching a competitive pipeline")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Plain-script directive lifts yield\n(removes the nested-agent / Ollama-400 failure)")

    fig.tight_layout()
    out = "runs/sia_compare.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)
    print("\n=== best accuracy per run ===")
    for name, ids, _ in GROUPS:
        row = " ".join(f"r{r}:{('%.3f'%b if b is not None else 'none')}" for r, b in zip(ids, [best_acc(x) for x in ids]))
        n_comp = sum(1 for r in ids if (best_acc(r) or 0) > 0.5)
        print(f"  {name.replace(chr(10),' '):24s}  {row}   yield={n_comp}/{len(ids)}")


if __name__ == "__main__":
    raise SystemExit(main())
