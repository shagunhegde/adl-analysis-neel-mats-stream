"""Figure 2: the null control holds.

Two panels, both showing that what ADL recovers from the cat organism is absent from the
neutral control run through the identical pipeline:
  (a) token relevance -- the paper's quantitative trait-recovery metric
  (b) cross-position consistency of the patchscope readout, normalised by each
      organism's own base/ft controls

Usage: plot_cat_vs_neutral.py <token_relevance_summary_cat.json> <..._neutral.json>
                              <position_consistency.json> <out_png>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

cat_tr = json.loads(Path(sys.argv[1]).read_text())
neu_tr = json.loads(Path(sys.argv[2]).read_text())
cons = json.loads(Path(sys.argv[3]).read_text())
OUT = Path(sys.argv[4])

VAR = ["difference", "base", "ft"]
LBL = {"difference": r"$\delta = h_{ft} - h_{base}$", "base": r"$h_{base}$", "ft": r"$h_{ft}$"}
COLOR = {"difference": "#c1121f", "base": "#457b9d", "ft": "#2a9d8f"}


def mean_pct(rows, variant, source="patchscope"):
    vals = [r["percentage"] for r in rows if r["variant"] == variant and r["source"] == source]
    return 100.0 * sum(vals) / len(vals) if vals else 0.0


fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), constrained_layout=True)

# ---- (a) token relevance -----------------------------------------------------------
ax = axes[0]
groups = ["cat\n(trait organism)", "neutral\n(null control)"]
x = np.arange(len(groups))
w = 0.26
for i, v in enumerate(VAR):
    vals = [mean_pct(cat_tr, v), mean_pct(neu_tr, v)]
    bars = ax.bar(x + (i - 1) * w, vals, w, color=COLOR[v], label=LBL[v],
                  alpha=1.0 if v == "difference" else 0.75)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.35, f"{val:.1f}%",
                ha="center", fontsize=8.5,
                fontweight="bold" if v == "difference" else "normal")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("% of top-20 patchscope tokens\njudged relevant to the organism")
ax.set_title("(a)  Token relevance — the paper's trait-recovery metric", fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(axis="y", alpha=0.25, lw=0.5)
ax.set_ylim(0, 17)

# ---- (b) cross-position consistency -------------------------------------------------
ax = axes[1]
KEY = {"difference": "diff", "base": "base", "ft": "ft"}
for i, v in enumerate(VAR):
    vals = [cons["subliminal_learning_cat"][KEY[v]]["mean_pairwise_jaccard"],
            cons["subliminal_learning_neutral"][KEY[v]]["mean_pairwise_jaccard"]]
    bars = ax.bar(x + (i - 1) * w, vals, w, color=COLOR[v],
                  alpha=1.0 if v == "difference" else 0.75)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.006, f"{val:.3f}",
                ha="center", fontsize=8.5,
                fontweight="bold" if v == "difference" else "normal")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("mean pairwise Jaccard of the judge's\nselected tokens across the 5 positions")
ax.set_title("(b)  Cross-position consistency of the readout", fontsize=10.5)
ax.grid(axis="y", alpha=0.25, lw=0.5)
ax.set_ylim(0, 0.30)

for xi, org in zip(x, ["subliminal_learning_cat", "subliminal_learning_neutral"]):
    d = cons[org]["diff"]["mean_pairwise_jaccard"]
    ctrl = max(cons[org]["base"]["mean_pairwise_jaccard"], cons[org]["ft"]["mean_pairwise_jaccard"])
    ax.text(xi, 0.275, r"$\delta$ / max(base, ft) = " + f"{d/ctrl:.2f}",
            ha="center", fontsize=9,
            bbox=dict(fc="#fff8e1" if d > ctrl else "#eef4f8", ec="0.6", lw=0.5, pad=2.5))

fig.suptitle(
    "The trait ADL reads off the cat organism is absent from the null control\n"
    "identical pipeline, identical layer and positions; the control student was trained on numbers from an unprompted teacher",
    fontsize=10.5)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
