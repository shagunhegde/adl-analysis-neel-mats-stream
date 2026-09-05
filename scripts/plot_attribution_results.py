"""Figure 4: every method against the two baselines, and the controlled collapse.

Left  -- cat vs neutral: methods vs the surface floor and the perplexity floor.
Right -- the controlled test: matching perplexity (cat vs penguin) collapses everything.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/rankings")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/fig4_attribution.png")


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(tag):
    j = json.loads((D / f"activation_rankings_{tag}.json").read_text())
    z = np.load(D / f"activation_rankings_{tag}.npz")
    y = z["labels"]
    bag = json.loads((D / f"bagofnumbers_{tag}.json").read_text())["auroc_oof"]
    return j["auroc"], auroc(z["nll_base"], y), bag


fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0), constrained_layout=True)

# ---------------- left: cat vs neutral, methods vs floors ---------------------------
r, nll, bag = load("cat_vs_neutral")
ax = axes[0]
proj = {k.split("[")[1][:-1]: v for k, v in r.items() if k.startswith("projection")}
dnll_best = max((v for k, v in r.items() if k.startswith("dNLL")))
dnll_best_k = max(((k, v) for k, v in r.items() if k.startswith("dNLL")), key=lambda kv: kv[1])[0]

rows = [
    ("bag-of-numbers\n(surface floor)", bag, "#8d99ae"),
    ("base NLL alone\n(perplexity floor)", nll, "#e07a5f"),
    (f"ΔNLL best of 18\n({dnll_best_k.split('[')[1].split(']')[0]})", dnll_best, "#2a9d8f"),
    ("projection\nδ_cat (real)", proj["cat"], "#c1121f"),
    ("projection\nδ_neutral (null)", proj["neutral"], "#457b9d"),
    ("M0 day 1\n(Δ projection)", 0.531, "#c1121f"),
]
ys = np.arange(len(rows))[::-1]
for yi, (lab, v, c) in zip(ys, rows):
    ax.barh(yi, v - 0.5, left=0.5, color=c, height=0.6)
    ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=9.5, fontweight="bold")
rnd = [v for k, v in proj.items() if k.startswith("random")]
ax.axvspan(min(rnd), max(rnd), color="0.85", zorder=0)
ax.text((min(rnd)+max(rnd))/2, -0.42, "random-direction\nrange", ha="center",
        va="center", fontsize=7.5, color="0.35")
ax.axvline(0.5, color="0.3", lw=1)
ax.axvline(nll, color="#e07a5f", ls="--", lw=1.3)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_ylim(-0.9, len(rows) - 0.3)
ax.set_xlim(0.45, 0.80); ax.set_xlabel("AUROC (cat-origin vs neutral-origin)")
ax.set_title("(a)  Nothing clears the perplexity floor", fontsize=11)
ax.grid(axis="x", alpha=0.25, lw=0.5)

# ---------------- right: the controlled collapse ------------------------------------
ax = axes[1]
rn, nlln, bagn = load("cat_vs_neutral")
rp, nllp, bagp = load("cat_vs_penguin")
groups = ["base NLL\n(perplexity)", "ΔNLL\n(best of 18)", "projection\n(best of 6)"]
vals_n = [nlln, max(v for k, v in rn.items() if k.startswith("dNLL")),
          max(v for k, v in rn.items() if k.startswith("projection"))]
vals_p = [nllp, max(v for k, v in rp.items() if k.startswith("dNLL")),
          max(v for k, v in rp.items() if k.startswith("projection"))]
x = np.arange(3); w = 0.36
b1 = ax.bar(x - w/2, np.array(vals_n) - 0.5, w, bottom=0.5, color="#e07a5f",
            label="cat vs neutral   (perplexity differs: 0.77 vs 0.43)")
b2 = ax.bar(x + w/2, np.array(vals_p) - 0.5, w, bottom=0.5, color="#457b9d",
            label="cat vs penguin  (perplexity matched: 0.77 vs 0.74)")
for bars, vals in ((b1, vals_n), (b2, vals_p)):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.006, f"{v:.3f}",
                ha="center", fontsize=9.5, fontweight="bold")
ax.axhline(0.5, color="0.3", lw=1)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9.5)
ax.set_ylim(0.45, 0.80); ax.set_ylabel("AUROC")
ax.set_title("(b)  Remove the perplexity gap and everything collapses", fontsize=11)
ax.legend(fontsize=8.5, loc="upper right"); ax.grid(axis="y", alpha=0.25, lw=0.5)

fig.suptitle(
    "The one apparent success was a perplexity artifact, shown by construction\n"
    "cat-teacher completions are simply less predictable to the base model; matching that removes every effect",
    fontsize=11)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
