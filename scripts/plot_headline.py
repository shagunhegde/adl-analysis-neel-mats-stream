"""Headline figure: the ceiling, and what survives its controls.

Two methods LOOKED like they worked. Showing only their controlled values hides the
finding; showing only their apparent values would be the failure mode. So both are drawn:
a hollow bar for the apparent result, a solid bar for what survives, and an arrow between.

The unrecovered gap is measured from the best surviving ATTRIBUTION result, not from the
apparent ones and not from M1's membership score -- membership is not attribution.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

pb = json.loads(Path("artifacts/privileged/privileged_bound.json").read_text())
OUT = Path("artifacts/fig_headline.png")

# (label, apparent, controlled_or_None, controlled_note, colour)
LEFT = [
    ("bag-of-numbers  (surface baseline)",      0.5248, None, "", "#8d99ae"),
    ("logit-lens on δ_cat",                     0.4940, None, "", "#9d4edd"),
    ("M0  activation-difference projection",    0.5310, None, "", "#c1121f"),
    ("base-model projection onto δ_cat",        0.5434, None, "", "#c1121f"),
    ("base NLL  (perplexity baseline)",         0.7351, None, "", "#e07a5f"),
    ("ΔNLL steering  (best of 18)",             0.7428, 0.5123, "perplexity matched", "#2a9d8f"),
    ("M1  gradient / task-vector alignment",    0.9195, 0.5509, "membership + wrong-trait component removed", "#457b9d"),
]
CEIL = pb["cat_vs_neutral"]["llr_sum"]
CEIL_P = pb["cat_vs_penguin"]["llr_sum"]

fig, ax = plt.subplots(figsize=(12.4, 6.2), constrained_layout=True)
ys = np.arange(len(LEFT))[::-1]

for yi, (lab, app, ctrl, note, col) in zip(ys, LEFT):
    if ctrl is None:
        ax.barh(yi, app - 0.5, left=0.5, color=col, height=0.55)
        ax.text(app + 0.008, yi, f"{app:.3f}", va="center", fontsize=10, fontweight="bold")
    else:
        # apparent: hollow. controlled: solid. arrow between.
        ax.barh(yi, app - 0.5, left=0.5, facecolor="none", edgecolor=col,
                height=0.55, lw=1.6, ls="--")
        ax.barh(yi, ctrl - 0.5, left=0.5, color=col, height=0.55)
        ax.text(app + 0.008, yi, f"{app:.3f}", va="center", fontsize=9.5,
                color=col, alpha=0.85)
        ax.text(ctrl + 0.008, yi - 0.03, f"{ctrl:.3f}", va="center", fontsize=10,
                fontweight="bold")
        ax.annotate("", xy=(ctrl + 0.012, yi - 0.30), xytext=(app - 0.004, yi - 0.30),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
        ax.text(app + 0.055, yi - 0.30, note, ha="left", va="center",
                fontsize=7.8, color=col, style="italic")

ax.axvline(0.5, color="0.25", lw=1.1)
ax.text(0.5, len(LEFT) - 0.35, "chance", ha="center", fontsize=8.5, color="0.35")

ax.axvline(CEIL, color="#1b7f4b", lw=3)
ax.text(CEIL - 0.006, len(LEFT) - 0.35, f"privileged ceiling  {CEIL:.3f}",
        ha="right", fontsize=10.5, color="#1b7f4b", fontweight="bold")

best_attr = max(v for _, v, c, _, _ in LEFT if c is None and v < 0.7)
ax.axvspan(best_attr, CEIL, color="#1b7f4b", alpha=0.06, zorder=0)
ax.annotate(f"{CEIL - best_attr:.2f} AUROC of recoverable signal that no method retrieves",
            xy=(0.755, -1.02), ha="center", va="center",
            fontsize=10, color="#1b7f4b",
            bbox=dict(fc="white", ec="#1b7f4b", lw=0.9, pad=4.0))

ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in LEFT], fontsize=9.5)
ax.set_xlim(0.46, 1.06); ax.set_ylim(-1.55, len(LEFT) + 0.1)
ax.set_xlabel("AUROC — separating cat-teacher rows from control rows", fontsize=10.5)
ax.grid(axis="x", alpha=0.25, lw=0.5)
ax.legend(handles=[
    Patch(facecolor="none", edgecolor="0.35", ls="--", lw=1.5, label="apparent result"),
    Patch(facecolor="0.45", label="after controls"),
], fontsize=9, loc="lower left", framealpha=0.95)

fig.suptitle(
    "Subliminal learning is 97% recoverable from the training data — no model-diffing method retrieves it\n"
    "privileged ceiling = log p(x | s_cat) − log p(x | s_neutral): same weights, two system prompts, no extra artifacts",
    fontsize=11.5)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT, f"| ceiling {CEIL:.4f}, best attribution {best_attr:.4f}, gap {CEIL-best_attr:.3f}")
print(f"(cat vs penguin ceiling for reference: {CEIL_P:.4f})")
