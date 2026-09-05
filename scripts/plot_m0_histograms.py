"""Score histograms for M0: cat-origin vs neutral-origin samples.

The requested backbone plot. Shows the projection score distributions for the real
diff direction alongside the controls, so the null is visible rather than asserted.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NPZ = np.load(sys.argv[1])
SUMMARY = json.loads(Path(sys.argv[2]).read_text())
OUT = Path(sys.argv[3])

labels = NPZ["labels"]
keys = [k for k in NPZ.files if k != "labels"]


def find(sub):
    for k in keys:
        if all(t in k for t in sub):
            return k
    return None


panels = [
    (find(["completion", "delta_cat", "pool24"]), "δ_cat  (the real direction)", "#c1121f"),
    (find(["completion", "delta_neutral", "pool24"]), "δ_neutral  (null organism)", "#457b9d"),
    (find(["completion", "delta_penguin", "pool24"]), "δ_penguin  (wrong trait)", "#2a9d8f"),
]
panels = [p for p in panels if p[0]]

auroc_by_tag = {r["tag"]: r["auroc"] for r in SUMMARY["results"]}


def auroc_for(key):
    for tag, a in auroc_by_tag.items():
        norm = tag.replace(" ", "_").replace("|", "").replace("@", "")
        if norm == key:
            return a
    return None


fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.9),
                         constrained_layout=True, sharey=True)
if len(panels) == 1:
    axes = [axes]

for ax, (key, title, col) in zip(axes, panels):
    s = NPZ[key]
    s = (s - s.mean()) / s.std()          # standardise: only the SHAPE matters here
    bins = np.linspace(-4, 4, 60)
    ax.hist(s[labels == 0], bins=bins, alpha=0.62, label="neutral-origin",
            color="#8d99ae", density=True)
    ax.hist(s[labels == 1], bins=bins, alpha=0.62, label="cat-origin",
            color=col, density=True)
    a = auroc_for(key)
    ax.set_title(f"{title}\nAUROC = {a:.3f}" if a else title, fontsize=10.5)
    ax.set_xlabel(r"projection score $s_i=\langle \Delta_i,\hat\delta\rangle$  (standardised)")
    ax.grid(alpha=0.22, lw=0.5)

axes[0].set_ylabel("density")
axes[0].legend(fontsize=8.5)

c = SUMMARY["controls"]
fig.suptitle(
    "M0 projection scores do not separate cat-origin from neutral-origin samples\n"
    f"2,000 per class · the real direction (AUROC {auroc_for(panels[0][0]):.3f}) is no better than a null direction, "
    f"nor than the best of 20 random directions ({c['random_direction_auroc_max']:.3f})",
    fontsize=10.5)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
