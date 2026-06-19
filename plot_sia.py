"""Plot SIA self-improvement on Ollama: per-generation accuracy + turns used.

Usage: python3 plot_sia.py <run_dir> <log_file> [out.png]
  e.g. python3 plot_sia.py runs/run_7 /tmp/sia_st100.log runs/run_7/sia_plot.png

Accuracy per generation is computed directly from each gen's submission.csv vs the
private gold labels (spaceship-titanic). Turns per generation = number of model
requests the meta/feedback agent made in that generation's log segment.
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOLD = "sia/tasks/spaceship-titanic/data/private/test.csv"


def load_col(path, col):
    return {r["PassengerId"]: str(r[col]).strip().lower()
            for r in csv.DictReader(open(path))}


def accuracy(sub_path):
    gold = load_col(GOLD, "Transported")
    pred = load_col(sub_path, "Transported")
    common = set(gold) & set(pred)
    if not common:
        return None
    return sum(gold[k] == pred[k] for k in common) / len(common)


def turns_per_gen(log_path, n_gens):
    """Count model requests per generation by splitting the log at generation
    boundaries. Returns {gen: n_requests}."""
    text = Path(log_path).read_text(errors="ignore")
    lines = text.splitlines()
    # boundary markers: meta-agent for gen 1 runs first; then for each gen the
    # feedback agent (which authors gen g+1) follows "Running feedback agent for
    # generation g". We bucket request lines by the most recent generation marker.
    counts = {g: 0 for g in range(1, n_gens + 1)}
    cur = 1
    fb = re.compile(r"Running feedback agent for generation (\d+)")
    start = re.compile(r"Starting Generation (\d+)")
    for ln in lines:
        m = fb.search(ln) or start.search(ln)
        if m:
            cur = min(int(m.group(1)) + (1 if "feedback" in ln else 0), n_gens)
            continue
        if re.search(r"\bRequest\b|model request|chat/completions", ln):
            counts[cur] = counts.get(cur, 0) + 1
    return counts


def main():
    run_dir = Path(sys.argv[1])
    log = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else str(run_dir / "sia_plot.png")
    gens = sorted(int(p.name.split("_")[1]) for p in run_dir.glob("gen_*") if p.is_dir())
    if not gens:
        print("no generations found in", run_dir); return 1

    acc, status = [], []
    for g in gens:
        sub = run_dir / f"gen_{g}" / "submission.csv"
        if sub.exists():
            a = accuracy(sub)
            acc.append(a); status.append("ok" if a is not None else "bad-sub")
        else:
            acc.append(None); status.append("crash/no-sub")
    turns = turns_per_gen(log, max(gens))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    # accuracy
    xs = gens
    ys = [a if a is not None else 0 for a in acc]
    ax1.plot(xs, ys, "o-", color="#1D4ED8", lw=2, ms=8)
    for x, a, s in zip(xs, acc, status):
        lbl = f"{a:.3f}" if a is not None else s
        ax1.annotate(lbl, (x, a if a is not None else 0),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax1.axhline(0.50, ls="--", color="gray", lw=1)
    ax1.text(xs[0], 0.51, "random 0.50", color="gray", fontsize=8)
    ax1.axhspan(0.79, 0.81, color="green", alpha=0.12)
    ax1.set_ylabel("spaceship-titanic accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_title("SIA self-improvement on Ollama (qwen3-coder:30b)")
    # turns
    ax2.bar(xs, [turns.get(g, 0) for g in xs], color="#0F766E", alpha=0.8)
    ax2.set_ylabel("meta/feedback requests (turns)")
    ax2.set_xlabel("generation")
    ax2.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)
    print("accuracy by gen:", {g: (round(a, 4) if a is not None else None) for g, a in zip(gens, acc)})
    print("turns by gen:", {g: turns.get(g, 0) for g in gens})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
