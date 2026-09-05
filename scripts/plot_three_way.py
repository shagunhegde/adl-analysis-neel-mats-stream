"""Figure 3: three organisms, one pipeline.

cat (strong trait) / penguin (weaker trait) / neutral (no trait), all trained on the same
prompt pool with the same recipe and read out by the identical ADL configuration.

Panel (a) behavioural transmission -- ground truth.
Panel (b) ADL token relevance      -- the paper's detection metric.
Panel (c) cross-position consistency of the patchscope readout, normalised by each
          organism's own base/ft controls.

Usage: plot_three_way.py <pref.json> <cat_tr.json> <peng_tr.json> <neu_tr.json> <cons.json> <out.png>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

pref = json.loads(Path(sys.argv[1]).read_text())
trs = {k: json.loads(Path(v).read_text())
       for k, v in zip(["cat", "penguin", "neutral"], sys.argv[2:5])}
cons = json.loads(Path(sys.argv[5]).read_text())
OUT = Path(sys.argv[6])

ORGS = ["cat", "penguin", "neutral"]
ORGKEY = {o: f"subliminal_learning_{o}" for o in ORGS}
TARGET = {"cat": "cat", "penguin": "penguin", "neutral": None}
LBL = ["cat\n(strong trait)", "penguin\n(weaker trait)", "neutral\n(no trait)"]
C = {"trait": "#c1121f", "base": "#457b9d", "ft": "#2a9d8f"}
x = np.arange(3)

fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4), constrained_layout=True)


def tr_mean(rows, variant, source="patchscope"):
    v = [r["percentage"] for r in rows if r["variant"] == variant and r["source"] == source]
    return 100.0 * sum(v) / len(v) if v else 0.0


# ---- (a) behavioural ---------------------------------------------------------------
ax = axes[0]
student, base_v, err_s, err_b = [], [], [], []
for o in ORGS:
    t = TARGET[o] or "cat"          # neutral has no target; show the cat probe for scale
    student.append(pref[o]["targets"][t]["substring_rate"] * 100)
    err_s.append(pref[o]["targets"][t]["substring_ci95"] * 100)
    base_v.append(pref["base"]["targets"][t]["substring_rate"] * 100)
    err_b.append(pref["base"]["targets"][t]["substring_ci95"] * 100)
ax.bar(x - 0.19, base_v, 0.36, yerr=err_b, capsize=3, color="0.72", label="base model")
ax.bar(x + 0.19, student, 0.36, yerr=err_s, capsize=3, color=C["trait"], label="student")
for xi, (s, b) in enumerate(zip(student, base_v)):
    ax.text(xi + 0.19, s + max(err_s) + 0.9, f"{s:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(LBL)
ax.set_ylabel("% naming the target animal\n(50 prompts x 100 samples)")
ax.set_title("(a)  Behavioural transmission — ground truth", fontsize=10.5)
ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_ylim(0, 48)
ax.text(2, 12, "probe shown is 'cat';\nneutral has no target", ha="center", fontsize=7.5, color="0.35")

# ---- (b) token relevance ------------------------------------------------------------
ax = axes[1]
for i, (v, lab) in enumerate([("difference", r"$\delta$"), ("base", "base"), ("ft", "ft")]):
    vals = [tr_mean(trs[o], v) for o in ORGS]
    bars = ax.bar(x + (i - 1) * 0.26, vals, 0.26,
                  color=C["trait"] if v == "difference" else (C["base"] if v == "base" else C["ft"]),
                  alpha=1.0 if v == "difference" else 0.75, label=lab)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.3, f"{val:.1f}",
                ha="center", fontsize=8.5, fontweight="bold" if v == "difference" else "normal")
ax.set_xticks(x); ax.set_xticklabels(LBL)
ax.set_ylabel("% of top-20 patchscope tokens\njudged relevant to the organism")
ax.set_title("(b)  ADL token relevance — the paper's metric", fontsize=10.5)
ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_ylim(0, 16.5)
ax.annotate("false negative:\ntrait is real but\nmetric reads 0.0%",
            xy=(1 - 0.26, 0.3), xytext=(1.15, 7.2), fontsize=8, color="#a33", ha="left",
            arrowprops=dict(arrowstyle="->", color="#a33", lw=1.1))

# ---- (c) consistency ----------------------------------------------------------------
ax = axes[2]
ratios = []
for o in ORGS:
    c = cons[ORGKEY[o]]
    ratios.append(c["diff"]["mean_pairwise_jaccard"] /
                  max(c["base"]["mean_pairwise_jaccard"], c["ft"]["mean_pairwise_jaccard"]))
bars = ax.bar(x, ratios, 0.5, color=[C["trait"], "#e07a5f", C["base"]])
for b, r in zip(bars, ratios):
    ax.text(b.get_x() + b.get_width() / 2, r + 0.04, f"{r:.2f}", ha="center",
            fontsize=10, fontweight="bold")
ax.axhline(1.0, color="0.4", ls="--", lw=1.1)
ax.text(2.42, 1.03, "δ no more consistent\nthan the models alone", fontsize=7.5,
        color="0.35", ha="right")
ax.set_xticks(x); ax.set_xticklabels(LBL)
ax.set_ylabel(r"consistency of $\delta$ across positions," "\n" r"relative to that organism's own base/ft")
ax.set_title("(c)  Cross-position consistency of the readout", fontsize=10.5)
ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_ylim(0, 2.35)

fig.suptitle(
    "ADL readout strength tracks transmission strength — and misses a real trait at the weak end\n"
    "Qwen2.5-7B students, same prompt pool, same recipe, same layer and token positions; only the teacher's system prompt differs",
    fontsize=10.5)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
