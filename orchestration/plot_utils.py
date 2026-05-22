import os
import re
import json
import glob


def plot_scores(run_directory: str, task_name: str = "") -> str | None:
    """
    Plot public score (dashed) and private scores per model (solid) across generations.

    Public score:   gen_X/results.json                              — public dev set
    Private scores: private_scores/gen_X/{model_slug}/private_result.json

    Saves private_scores.png to run_directory. Returns the path or None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")
        return None

    # ── Collect public scores ──────────────────────────────────────────────────
    pub_gens:   list[int]   = []
    pub_scores: list[float] = []
    lower_is_better = False

    g = 0
    while True:
        gen_dir = os.path.join(run_directory, f"gen_{g}")
        if not os.path.isdir(gen_dir):
            if g > 0:
                break
            g += 1
            continue
        pub_path = os.path.join(gen_dir, "results.json")
        if os.path.exists(pub_path):
            with open(pub_path) as f:
                d = json.load(f)
            s = d.get("score", d.get("accuracy"))
            if s is not None and d.get("error") is None:
                pub_gens.append(g)
                pub_scores.append(float(s))
                lower_is_better = d.get("lower_is_better", False)
        g += 1

    # ── Collect private scores per model ───────────────────────────────────────
    # model_slug → {gen: score}
    model_data: dict[str, dict[int, float]] = {}

    def _read_private(path: str, slug: str) -> None:
        gen_str = re.search(r"gen_(\d+)", path)
        if not gen_str:
            return
        gen = int(gen_str.group(1))
        with open(path) as f:
            d = json.load(f)
        s = d.get("score")
        if s is not None and d.get("error") is None:
            model_data.setdefault(slug, {})[gen] = float(s)
            nonlocal lower_is_better
            lower_is_better = d.get("lower_is_better", lower_is_better)

    # private_scores/gen_X/{model_slug}/private_result.json
    for priv_path in sorted(glob.glob(
        os.path.join(run_directory, "private_scores", "gen_*", "*", "private_result.json")
    )):
        slug = os.path.basename(os.path.dirname(priv_path))
        _read_private(priv_path, slug)

    if not pub_scores and not model_data:
        return None

    # ── Plot ───────────────────────────────────────────────────────────────────
    direction = "lower" if lower_is_better else "higher"
    fig, ax = plt.subplots(figsize=(10, 4))

    # Private scores — one solid line per model
    colors = ["#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]
    for i, (slug, gen_score) in enumerate(sorted(model_data.items())):
        gens   = sorted(gen_score)
        scores = [gen_score[g] for g in gens]
        ax.plot(gens, scores, marker="s", linewidth=2.5,
                label=f"Private — {slug}", color=colors[i % len(colors)])

    # Public score — dashed reference
    if pub_scores:
        ax.plot(pub_gens, pub_scores, marker="o", linewidth=1.5, linestyle="--", alpha=0.6,
                label="Public (dev)", color="#4C72B0")

    ax.set_xlabel("Generation")
    ax.set_ylabel(f"Score ({direction} is better)")
    ax.set_title(f"Score evolution — {task_name or os.path.basename(run_directory)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(run_directory, "private_scores.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


# Keep old name as alias
plot_private_scores = plot_scores
