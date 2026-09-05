"""Figure 5: every method against the privileged ceiling.

The privileged bound log p(x|s_A) - log p(x|s_B) is the Bayes-optimal discriminator for
"which teacher generated this row", so it upper-bounds any teacher-free method. Without
it, a negative is uninterpretable: 0.53 could mean the method is bad or the task is
impossible. With it, the gap is the finding.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

pb = json.loads(Path("artifacts/privileged/privileged_bound.json").read_text())
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/fig5_ceiling.png")

PANELS = [
    ("cat vs neutral", pb["cat_vs_neutral"]["llr_sum"], [
        ("bag-of-numbers\n(surface)", 0.5248, "#8d99ae"),
        ("logit-lens δ_cat", 0.4940, "#9d4edd"),
        ("M0 Δ-projection", 0.5310, "#c1121f"),
        ("base-model\nprojection δ_cat", 0.5434, "#c1121f"),
        ("base NLL\n(perplexity)", 0.7351, "#e07a5f"),
        ("ΔNLL best of 18", 0.7428, "#2a9d8f"),
        ("M1 gradient\n(membership-ctrl'd)", 0.8469, "#457b9d"),
        ("M1 after removing\nshared component", 0.5020, "#457b9d"),
    ]),
    ("cat vs penguin\n(perplexity matched)", pb["cat_vs_penguin"]["llr_sum"], [
        ("bag-of-numbers\n(surface)", 0.5040, "#8d99ae"),
        ("logit-lens δ_cat", 0.5019, "#9d4edd"),
        ("base-model\nprojection δ_cat", 0.5029, "#c1121f"),
        ("base NLL\n(perplexity)", 0.5167, "#e07a5f"),
        ("ΔNLL best of 18", 0.5123, "#2a9d8f"),
    ]),
]

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.35, 1]})

for ax, (title, ceiling, rows) in zip(axes, PANELS):
    ys = np.arange(len(rows))[::-1]
    for yi, (lab, v, c) in zip(ys, rows):
        ax.barh(yi, v - 0.5, left=0.5, color=c, height=0.62)
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.axvline(0.5, color="0.3", lw=1)
    ax.axvspan(ceiling - 0.004, ceiling + 0.004, color="#1b7f4b", alpha=0.9, zorder=3)
    ax.text(ceiling, len(rows) - 0.35,
            f"privileged\nceiling {ceiling:.3f}", ha="center", va="bottom",
            fontsize=9, color="#1b7f4b", fontweight="bold")
    # shade the unrecovered gap
    best = max(v for _, v, _ in rows)
    ax.axvspan(best, ceiling, color="#1b7f4b", alpha=0.07, zorder=0)
    ax.text((best + ceiling) / 2, -0.75, f"unrecovered\n{ceiling - best:.2f} AUROC",
            ha="center", va="center", fontsize=8.5, color="#1b7f4b")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8.8)
    ax.set_xlim(0.45, 1.02); ax.set_ylim(-1.25, len(rows) + 0.35)
    ax.set_xlabel("AUROC")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

fig.suptitle(
    "The information is there; no teacher-free method recovers it\n"
    "privileged bound = log p(x | s_A) − log p(x | s_B): same weights, two system prompts — the Bayes-optimal discriminator",
    fontsize=11)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
