"""Aggregate SIA-on-Ollama variance across seeds: per-generation accuracy
trajectories + success-rate summary.

Each seed is one independent SIA run (qwen3-coder:30b, spaceship-titanic,
SIA_MAX_TURNS=50, max_gen 3). A generation "succeeds" if it wrote a
submission.csv whose Transported predictions match the private gold labels.

Usage: python3 plot_sia_variance.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GOLD = "sia/tasks/spaceship-titanic/data/private/test.csv"
# (seed run_id, label, batch). run_6 is the earlier pre-fix success; 13-17 the
# clean post-fix variance batch.
SEEDS = [(6, "run_6 (pre-fix)", "pre"),
         (13, "run_13", "post"), (14, "run_14", "post"), (15, "run_15", "post"),
         (16, "run_16", "post"), (17, "run_17", "post")]


def load(path, col="Transported"):
    return {r["PassengerId"]: str(r.get(col, "")).strip().lower()
            for r in csv.DictReader(open(path))}


def accuracy(sub):
    gold = load(GOLD)
    pred = load(sub)
    common = set(gold) & set(pred)
    if not common:
        return None
    return sum(gold[k] == pred[k] for k in common) / len(common)


def seed_traj(rid):
    base = Path(f"runs/run_{rid}")
    gens = sorted(int(p.name.split("_")[1]) for p in base.glob("gen_*") if p.is_dir())
    out = {}
    for g in gens:
        sub = base / f"gen_{g}" / "submission.csv"
        out[g] = accuracy(sub) if sub.exists() else None
    return out


def main():
    trajs = {rid: seed_traj(rid) for rid, _, _ in SEEDS}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    # --- left: trajectories ---
    colors = plt.cm.tab10.colors
    for i, (rid, lbl, batch) in enumerate(SEEDS):
        t = trajs[rid]
        xs = sorted(t)
        ys = [t[g] for g in xs]
        # plot accuracy points; mark no-sub gens at y=0 with an x
        succ_x = [g for g in xs if t[g] is not None]
        succ_y = [t[g] for g in succ_x]
        is_success = any(v is not None for v in ys)
        c = colors[i % 10]
        if succ_x:
            ax1.plot(succ_x, succ_y, "o-", color=c, lw=2.2, ms=8,
                     label=f"{lbl} → {max(succ_y):.3f}")
            for g in succ_x:
                ax1.annotate(f"{t[g]:.3f}", (g, t[g]), textcoords="offset points",
                             xytext=(0, 9), ha="center", fontsize=8, color=c)
        nosub_x = [g for g in xs if t[g] is None]
        ax1.scatter(nosub_x, [0.0] * len(nosub_x), marker="x", color=c, s=40,
                    alpha=0.6, label=None if is_success else f"{lbl} (no valid sub)")
    ax1.axhspan(0.79, 0.81, color="green", alpha=0.12)
    ax1.text(1.0, 0.795, "Kaggle-good 0.79–0.81", color="green", fontsize=8)
    ax1.axhline(0.50, ls="--", color="gray", lw=1)
    ax1.text(1.0, 0.51, "random 0.50", color="gray", fontsize=8)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xlabel("generation")
    ax1.set_ylabel("spaceship-titanic accuracy (private gold)")
    ax1.set_ylim(-0.03, 1.0)
    ax1.set_title("SIA self-improvement on Ollama (qwen3-coder:30b)\n6 independent seeds, max_gen 3")
    ax1.legend(fontsize=7.5, loc="center right")

    # --- right: success summary ---
    post = [rid for rid, _, b in SEEDS if b == "post"]
    post_succ = [rid for rid in post if any(v is not None for v in trajs[rid].values())]
    best = [max([v for v in trajs[rid].values() if v is not None], default=None) for rid in post]
    n_ok = len(post_succ)
    ax2.bar(["valid\nsubmission", "no valid\nsubmission"], [n_ok, len(post) - n_ok],
            color=["#15803D", "#B91C1C"], alpha=0.85)
    ax2.set_ylabel("seeds (post-fix batch, n=5)")
    ax2.set_title(f"Yield: {n_ok}/{len(post)} seeds reached\na competitive pipeline")
    for i, v in enumerate([n_ok, len(post) - n_ok]):
        ax2.text(i, v + 0.05, str(v), ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, len(post) + 0.6)

    fig.tight_layout()
    out = "runs/sia_variance.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)
    print("\n=== summary ===")
    for rid, lbl, batch in SEEDS:
        t = trajs[rid]
        best_acc = max([v for v in t.values() if v is not None], default=None)
        traj_str = " ".join(f"g{g}:{'%.3f'%t[g] if t[g] is not None else 'no-sub'}" for g in sorted(t))
        print(f"  {lbl:18s} [{batch}]  {traj_str}   best={best_acc}")
    print(f"\n  post-fix yield: {n_ok}/{len(post)} seeds produced a valid competitive submission")


if __name__ == "__main__":
    raise SystemExit(main())
