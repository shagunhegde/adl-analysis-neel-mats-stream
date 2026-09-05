"""Figure: where in the patchscope scale sweep does the hidden trait become readable?

For each token position and each of {delta, base, ft}, plot the best (lowest) rank at
which any cat-family token appears in the top-20 patchscope tokens, as a function of the
injection scale. Absence from the top-20 is plotted as "not present".

Usage: plot_sweep.py <patchscope_sweep_full.json> <out_png>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(sys.argv[2])
TOPK = 20  # tokens_k in auto_patch_scope config
MISS = TOPK + 1.5  # y-value meaning "not in top-20"

positions = sorted({r["position"] for r in SWEEP})
variants = ["diff", "base", "ft"]
COLOR = {"diff": "#c1121f", "base": "#457b9d", "ft": "#2a9d8f"}
LABEL = {"diff": r"$\delta = h_{ft} - h_{base}$", "base": r"$h_{base}$", "ft": r"$h_{ft}$"}

fig, axes = plt.subplots(1, len(positions), figsize=(3.1 * len(positions), 3.6),
                         sharey=True, constrained_layout=True)
if len(positions) == 1:
    axes = [axes]

for ax, pos in zip(axes, positions):
    for variant in variants:
        rec = next((r for r in SWEEP if r["position"] == pos and r["variant"] == variant), None)
        if rec is None:
            continue
        scales = sorted((float(s) for s in rec["sweep"]), key=float)
        ys, xs = [], []
        for s in scales:
            hits = rec["trait_token_scales"].get(str(s), [])
            xs.append(s)
            ys.append(min(h["rank"] for h in hits) if hits else MISS)
        xs, ys = np.array(xs), np.array(ys)
        m = xs <= 3.0  # signal band; larger scales are uniformly multilingual noise
        # nudge the two control series apart so both remain visible on the "absent" floor
        nudge = {"diff": 0.0, "base": -0.35, "ft": 0.35}[variant]
        ax.plot(xs[m], ys[m] + nudge, "o-", ms=3.2,
                lw=2.0 if variant == "diff" else 1.2, color=COLOR[variant],
                label=LABEL[variant] if pos == positions[0] else None,
                zorder=3 if variant == "diff" else 2,
                alpha=1.0 if variant == "diff" else 0.75)

    rec = next((r for r in SWEEP if r["position"] == pos and r["variant"] == "diff"), None)
    if rec and "saved_best_scale" in rec:
        bs = rec["saved_best_scale"]
        if bs <= 3.0:
            ax.axvline(bs, color="#333", ls="--", lw=1.1, zorder=1)
            ax.annotate(f"judge picked {bs}", xy=(bs, TOPK + 2.4), fontsize=7,
                        color="#333", ha="center", va="center",
                        bbox=dict(fc="white", ec="#333", lw=0.5, pad=1.4))
        else:
            ax.annotate(f"judge picked {bs}\n(far off-scale: noise)", xy=(1.7, TOPK + 2.4),
                        fontsize=7, color="#a33", ha="center", va="center",
                        bbox=dict(fc="#fff0f0", ec="#a33", lw=0.5, pad=1.4))

    ax.set_title(f"token position {pos}", fontsize=10)
    ax.set_xlim(0.4, 3.1)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 3.0])
    ax.set_xticklabels(["0.5", "1.0", "1.5", "2.0", "3.0"])
    ax.set_xlabel("patchscope injection scale")
    ax.set_ylim(TOPK + 4.2, -1.2)
    ax.axhspan(TOPK + 0.6, TOPK + 4.2, color="0.93", zorder=0)
    ax.grid(alpha=0.25, lw=0.5)

axes[0].set_ylabel("best rank of a cat-family token\nin the top-20  (lower = stronger)")
axes[0].set_yticks([0, 5, 10, 15, 20, MISS])
axes[0].set_yticklabels(["1", "5", "10", "15", "20", "absent"])
axes[0].legend(fontsize=8, loc="upper left", framealpha=0.95)

fig.suptitle(
    "The subliminal 'loves cats' trait is readable in the activation difference, not in either model\n"
    "Qwen2.5-7B-Instruct vs LoRA student trained only on number sequences · layer 13 · fineweb (unrelated text)",
    fontsize=10.5,
)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
