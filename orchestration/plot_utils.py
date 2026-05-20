import os
import json


def plot_scores(run_directory: str, task_name: str = "") -> str | None:
    """
    Plot private scores (main) and public scores (reference) across generations.

    Reads gen_X/private_result.json (primary) and gen_X/results.json (overlay)
    and saves private_scores.png to the run directory.
    Returns the plot path, or None if no data / matplotlib not installed.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")
        return None

    pub_gens, pub_scores = [], []
    priv_gens, priv_scores = [], []
    lower_is_better = False

    g = 1
    while True:
        gen_dir = os.path.join(run_directory, f"gen_{g}")
        if not os.path.isdir(gen_dir):
            break

        pub_path = os.path.join(gen_dir, "results.json")
        if os.path.exists(pub_path):
            with open(pub_path) as f:
                d = json.load(f)
            s = d.get("score", d.get("accuracy"))
            if s is not None and d.get("error") is None:
                pub_gens.append(g)
                pub_scores.append(float(s))
                lower_is_better = d.get("lower_is_better", False)

        priv_path = os.path.join(gen_dir, "private_result.json")
        if os.path.exists(priv_path):
            with open(priv_path) as f:
                d = json.load(f)
            s = d.get("score")
            if s is not None and d.get("error") is None:
                priv_gens.append(g)
                priv_scores.append(float(s))
                lower_is_better = d.get("lower_is_better", lower_is_better)

        g += 1

    if not pub_scores and not priv_scores:
        return None

    direction = "lower" if lower_is_better else "higher"
    fig, ax = plt.subplots(figsize=(9, 4))

    if priv_scores:
        ax.plot(priv_gens, priv_scores, marker='s', linewidth=2.5,
                label="Private (held-out)", color="#DD8452")
    if pub_scores:
        ax.plot(pub_gens, pub_scores, marker='o', linewidth=1.5,
                label="Public (dev)", color="#4C72B0", linestyle="--", alpha=0.6)

    ax.set_xlabel("Generation")
    ax.set_ylabel(f"Score ({direction} is better)")
    ax.set_title(f"Private score evolution — {task_name or os.path.basename(run_directory)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(run_directory, "private_scores.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


# Keep old name as alias so existing orchestrator call still works
plot_private_scores = plot_scores
