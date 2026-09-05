"""Mechanism figures for docs/12 -- ONE claim per file, sized to stay readable when a
reader pastes them into a document and the renderer shrinks them.

  fig_delta_decomposition.png   how much of delta_cat is trait-specific
  fig_trace_vs_behaviour.png    the trace does not scale with behaviour   (COSINE)
  fig_position_profile.png      where along the token positions the shift actually lives
  fig_not_a_direction_bars.png  neither component carries the readout, against the null
  fig_not_a_direction_sweep.png the same result shown as scale x rank, which counts hide

Every panel carries its own null or reference: each claim is "X is (not) above chance",
and a bar without its null is not evidence.

METRIC NOTE, load-bearing: docs/12 Results 1-2 tabulate a column labelled "raw <D, dhat>"
that is actually mean(cos) * mean(||D||) -- a product of two averages, not the mean of the
per-sample projections. Those differ, and the "flat across topic" claim holds only for the
derived product. The honest claims are (a) the COSINE is not topic-selective, plotted here,
and (b) the position profile, which shows where the domain-conditionality really sits.
"""
import json
from math import acos, cos, degrees, sin
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = Path("artifacts")
DPI = 220
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 14, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
                     "axes.titlesize": 15})

# =============================================================== A: decomposition =======
dc = json.loads((A / "mixed_adl/delta_cosines.json").read_text())["pools"]["pool24"]
C = dc["cosines"]; g = lambda a, b: C.get(f"{a}~{b}", C.get(f"{b}~{a}"))
c_pen, c_neu, chance = g("cat", "penguin"), g("cat", "neutral"), dc["chance_cosine_magnitude"]

# --- A1: the geometry, on its own -------------------------------------------------------
# The cat/penguin plane is genuinely 2-D, so the 38 deg and the right angle are both exact;
# nothing in this panel is schematic.
fig, ax = plt.subplots(figsize=(10.5, 8), constrained_layout=True)
th = acos(c_pen)
cat_v = np.array([cos(th), sin(th)])
ax.arrow(0, 0, .98, 0, head_width=.035, color="#457b9d", lw=3.2, length_includes_head=True)
ax.arrow(0, 0, *cat_v * .98, head_width=.035, color="#c1121f", lw=3.4, length_includes_head=True)
foot = np.array([cos(th), 0.0])
ax.plot([cat_v[0], foot[0]], [cat_v[1], foot[1]], "--", color="0.35", lw=2)
ax.arrow(0, 0, foot[0] * .96, 0, head_width=.032, color="#7d8597", lw=9, alpha=.45,
         length_includes_head=True)
sq = .045
ax.plot([foot[0] - sq, foot[0] - sq, foot[0]], [0, sq, sq], color="0.35", lw=1.6)
ax.text(cat_v[0] - .02, cat_v[1] + .04, r"$\delta_{cat}$", color="#c1121f", fontsize=27, ha="center")
ax.text(1.01, .035, r"$\delta_{penguin}$", color="#457b9d", fontsize=27, ha="left", va="bottom")
ax.text(foot[0] / 2, -.075, "shared component", ha="center", color="#4a5568", fontsize=17)
ax.text(foot[0] + .05, cat_v[1] / 2, "residual", ha="left", color="0.3", fontsize=17)
arc = np.linspace(0, th, 60)
ax.plot(.26 * np.cos(arc), .26 * np.sin(arc), color="0.45", lw=1.6)
ax.text(.31 * cos(th / 2), .31 * sin(th / 2), f"{degrees(th):.0f}°", fontsize=21, color="0.25")
ax.set_xlim(-.1, 1.42); ax.set_ylim(-.24, .92); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Two organisms trained on different animals,\ndrawn at their true angle", fontsize=19, pad=14)
ax.text(-.06, .90, f"cos = {c_pen:+.3f}        90° would mean unrelated", fontsize=16, va="top", color="0.2")
ax.text(-.06, -.155,
        r"$\delta_{neutral}$, the null organism, is not in this plane:" + "\n"
        f"{degrees(acos(c_neu)):.0f}° from cat,  {degrees(acos(g('neutral','penguin'))):.0f}° from penguin"
        "  —  the near-orthogonal pair\n"
        r"($\tau$, the WEIGHT update, is a different object: cat~penguin there is 89°)",
        fontsize=14.5, va="top", color="0.35")
fig.savefig(A / "fig_delta_geometry.png", dpi=DPI); plt.close(fig)

# --- A1b: the full pairwise matrix ------------------------------------------------------
# The "62% is a PROMPTED-TEACHER signature, not a finetuning signature" claim is a
# three-way comparison, and a single number cannot carry it: what makes the case is that
# the ONLY high cell is the prompted~prompted pair, while both pairs involving the
# unprompted organism sit at or near chance.
ORG = [("cat", "cat", True), ("penguin", "penguin", True), ("neutral", "neutral", False)]
M = np.full((3, 3), np.nan)   # diagonal masked: a self-cosine of 1.0 would dominate the
                              # colour scale and mean nothing
for i, (a, _, _) in enumerate(ORG):
    for j, (b, _, _) in enumerate(ORG):
        if i != j:
            M[i, j] = g(a, b)
fig, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=True)
cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#f2f2f2")
im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
for i in range(3):
    for j in range(3):
        if i == j:
            ax.text(j, i, "—", ha="center", va="center", fontsize=20, color="0.55")
        else:
            hot = abs(M[i, j]) > .5
            ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=21,
                    fontweight="bold" if hot else "normal", color="w" if hot else "0.15")
lab = [f"{n}\n{'prompted teacher' if pr else 'NO system prompt'}" for _, n, pr in ORG]
ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=13.5)
ax.set_yticks(range(3)); ax.set_yticklabels(lab, fontsize=13.5)
ax.set_xticks(np.arange(-.5, 3), minor=True); ax.set_yticks(np.arange(-.5, 3), minor=True)
ax.grid(which="minor", color="w", lw=3); ax.tick_params(which="minor", length=0)
ax.set_title("cos(δ, δ) between organisms\nthe only high cell is prompted ~ prompted", fontsize=18, pad=14)
ax.text(-.62, 2.86, f"chance magnitude = {chance:.3f}   ·   layer 13, token positions 2–4\n"
        "(pooling 0–4 flips signs — see docs/12 §3c)", fontsize=12.5, color="0.35", va="top")
fig.savefig(A / "fig_delta_matrix.png", dpi=DPI); plt.close(fig)

# --- A2: the energy budget, on its own ---------------------------------------------------
fig, ax2 = plt.subplots(figsize=(11.5, 5.6), constrained_layout=True)
left = 0
for lab, v, col in [("shared with $\\delta_{penguin}$\n(the generic fingerprint)", c_pen**2, "#457b9d"),
                    ("residual\n(trait-specific CANDIDATE)", 1 - c_pen**2, "#c1121f")]:
    ax2.barh(1, v, left=left, color=col, height=.46)
    ax2.text(left + v / 2, 1, f"{v:.0%}", ha="center", va="center", color="w", fontweight="bold", fontsize=30)
    ax2.text(left + v / 2, 1.33, lab, ha="center", va="bottom", fontsize=15, color=col)
    left += v
ax2.barh(.26, c_neu**2, color="#8d99ae", height=.46)
ax2.text(c_neu**2 + .018, .26, f"{c_neu**2:.1%}  overlap with $\\delta_{{neutral}}$  (the null organism)",
         va="center", fontsize=15, color="#4a5568")
ax2.axvline(chance**2, color="0.25", ls=":", lw=2)
ax2.text(chance**2 + .018, -.26, f"chance = {chance**2:.2%}", fontsize=13.5, color="0.25")
ax2.set_ylim(-.5, 1.85); ax2.set_xlim(0, 1.02); ax2.set_yticks([])
ax2.set_xlabel("share of $\\|\\delta_{cat}\\|^2$", fontsize=16)
ax2.set_title("Most of $\\delta_{cat}$ is not about cats", fontsize=19, pad=14)
fig.savefig(A / "fig_delta_energy.png", dpi=DPI); plt.close(fig)

# ========================================================== B: trace vs behaviour =======
tb = json.loads((A / "topic_bias/topic_bias_6.json").read_text())["records"]
# "our recipe" (spikein_p100) is dropped from this plot at the author's request: it is a
# REPLICATE of the released-cat condition (both 10k cat rows, 31.1% vs 34.7% behaviour), so
# it adds no independent condition. Flagged because it is also the point that sits closest
# to the proportionality reference line (+0.003 from it) -- i.e. the single datum most
# consistent with "the trace tracks behaviour" -- so its removal makes the figure read more
# favourably to the claim than the full data does. It remains in docs/12 Result 2's table.
BEH = {"cat": 34.7, "cat7k": 10.4, "mixed": 4.2, "penguin": 1.8, "neutral": 5.5}
CATROWS = {"cat": 10000, "cat7k": 7000, "mixed": 7000, "penguin": 0, "neutral": 0}
LBL = {"cat": "released cat\n10k cat rows", "cat7k": "cat7k alone\n7k cat rows",
       "mixed": "mixed 70/20/10\n7k + diluent", "penguin": "penguin\n(wrong trait)", "neutral": "neutral\n(null organism)"}
OFF = {"cat": (12, 6), "cat7k": (-6, 14), "mixed": (12, 6), "penguin": (12, -34), "neutral": (12, 6)}
cosv = {s: float(np.mean([r["cos_pos_mean"] for r in tb if r["student"] == s and r["direction"] == "cat"]))
        for s in BEH}

fig, ax = plt.subplots(figsize=(11.5, 7.6), constrained_layout=True)
floor = cosv["neutral"]
ax.axhspan(-.01, floor, color="#eceff1", zorder=0)
ax.text(38.6, floor - .004, "null-organism floor", fontsize=12, color="0.4", va="top", ha="right")
ax.plot([5.2, 34.7], [floor, cosv["cat"]], ls=(0, (7, 5)), color="#adb5bd", lw=2, zorder=1)
ax.text(25.5, floor + (cosv["cat"] - floor) * .49, "if the trace tracked\nbehaviour", fontsize=12.5,
        color="#868e96", rotation=15, ha="center")
for s in BEH:
    filled = CATROWS[s] > 0
    ax.scatter(BEH[s], cosv[s], s=210, zorder=3, facecolor="#c1121f" if filled else "none",
               edgecolor="#c1121f" if filled else "#495057", linewidth=2.4)
    ax.annotate(LBL[s], (BEH[s], cosv[s]), textcoords="offset points", xytext=OFF[s], fontsize=12, color="0.2")
ax.axvline(5.2, color="0.55", ls=":", lw=1.6)
ax.text(5.6, .232, "base rate", fontsize=12, color="0.45", va="top")
ax.annotate("", (10.4, cosv["cat7k"]), (34.7, cosv["cat"]), arrowprops=dict(arrowstyle="<->", color="#2a9d8f", lw=2.6))
ax.text(22.5, cosv["cat"] + .009, "3.3× the behaviour — same trace", ha="center", fontsize=14,
        color="#2a9d8f", fontweight="bold")
ax.set_xlabel("cat preference in behaviour (%)")
ax.set_ylabel("mean cos( D(x), $\\hat{\\delta}_{cat}$ )")
ax.set_xlim(-1.5, 39); ax.set_ylim(-.008, .245)
ax.set_title("The activation trace saturates long before the behaviour does", fontsize=16, pad=12)
ax.text(0, -.0055, "filled = trained on cat rows    ·    n = 300 per corpus × 7 corpora, layer 13, positions 0–4",
        fontsize=11, color="0.45")
fig.savefig(A / "fig_trace_vs_behaviour.png", dpi=DPI); plt.close(fig)

# ========================================================== B2: position profile ========
R = {(r["student"], r["direction"], r["corpus"]): r for r in
     json.loads((A / "topic_bias/topic_bias.json").read_text())["records"]}
WEB = ["fineweb_random", "fineweb_cat", "fineweb_dog", "fineweb_penguin"]
NUM = ["numbers_cat", "numbers_neutral", "numbers_synth"]
fig, ax = plt.subplots(figsize=(11.5, 7.2), constrained_layout=True)
for cp in WEB:
    ax.plot(range(5), R[("cat", "cat", cp)]["per_position"], "-o", color="#457b9d", lw=2.2, ms=8,
            alpha=.85, label="generic web text" if cp == WEB[0] else None)
for cp in NUM:
    ax.plot(range(5), R[("cat", "cat", cp)]["per_position"], "-s", color="#c1121f", lw=2.2, ms=8,
            alpha=.85, label="number sequences (the finetuning domain)" if cp == NUM[0] else None)
ax.axhline(0, color="0.6", lw=1.2)
ax.axvspan(1.5, 4.4, color="#fff4e6", zorder=0)
ax.text(3, ax.get_ylim()[1] * .55, "positions the ADL readout uses", fontsize=12.5, color="#b06500", ha="center")
ax.annotate("position 0 dominates any average\nover 0–4, and points the OPPOSITE way\non the two corpus families",
            (0, 44), (1.15, 33), fontsize=12.5, color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.45", lw=1.6))
ax.annotate("here the finetuning domain is\n7.6× HIGHER than web text", (3, 5.4), (2.0, 20),
            fontsize=12.5, color="#c1121f", arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.6))
ax.set_xticks(range(5)); ax.set_xlabel("token position"); ax.set_ylabel("⟨ D(x), $\\hat{\\delta}_{cat}$ ⟩")
ax.legend(fontsize=12.5, frameon=False, loc="lower right")
ax.set_title("The shift along $\\delta_{cat}$ is strongly position-dependent", fontsize=16, pad=12)
fig.savefig(A / "fig_position_profile.png", dpi=DPI); plt.close(fig)

# ====================================================== C1: bars against the null =======
op = json.loads((A / "orthogonal_patchscope_null/orthogonal_patchscope.json").read_text())
hits, cells = {}, {}
for r in op["results"]:
    hits[r["direction"]] = hits.get(r["direction"], 0) + len(r["hits"]["cat"])
    for h in r["hits"]["cat"]:
        cells.setdefault(r["direction"], []).append((h["scale"], h["rank"]))
rand = np.array([v for k, v in hits.items() if k.startswith("random")])

BARS = [(r"$\delta_{cat}$" + "\nthe whole vector", "delta_cat", "#c1121f"),
        ("residual\n⊥ penguin", "resid_orthogonal_to_penguin", "#e07a5f"),
        ("shared\nwith penguin", "shared_with_penguin", "#457b9d"),
        (r"$\delta_{penguin}$", "delta_penguin", "#8d99ae")]
fig, ax = plt.subplots(figsize=(11, 7.4), constrained_layout=True)
xs = np.arange(len(BARS))
ax.axhspan(0, rand.max(), color="#eceff1", zorder=0)
ax.axhline(rand.max(), color="0.3", ls="--", lw=2, zorder=2)
ax.bar(xs, [hits[k] for _, k, _ in BARS], color=[c for _, _, c in BARS], width=.62, zorder=3)
jx = np.random.default_rng(0).normal(len(BARS) - .22, .085, len(rand))
ax.scatter(jx, rand, s=52, color="0.2", alpha=.75, zorder=4, label="each of 20 random directions")
for i, (_, k, _) in enumerate(BARS):
    ax.text(i, hits[k] + 2, str(hits[k]), ha="center", fontweight="bold", fontsize=18)
ax.text(len(BARS) - .22, rand.max() + 2.6, "null max = 10\n(the residual exactly ties it)",
        ha="center", fontsize=12.5, color="0.25")
ax.set_xticks(xs); ax.set_xticklabels([l for l, _, _ in BARS], fontsize=13.5)
ax.set_ylabel("times a cat token appeared\n(31 scales × 3 token positions)")
ax.set_ylim(0, 80); ax.legend(loc="upper center", fontsize=12.5, frameon=False)
ax.set_title("Neither component carries the readout — only the whole vector does", fontsize=16, pad=12)
fig.savefig(A / "fig_not_a_direction_bars.png", dpi=DPI); plt.close(fig)

# ====================================================== C2: the sweep itself ============
ROWS = [("delta_cat", r"$\delta_{cat}$  (whole vector)"), ("resid_orthogonal_to_penguin", "residual ⊥ penguin"),
        ("shared_with_penguin", "shared with penguin"), ("delta_penguin", r"$\delta_{penguin}$"),
        ("random0", "random #0   (best of 20)"), ("random6", "random #6"), ("random1", "random #1   (typical)")]
scales = op["scales"]; sx = {s: i for i, s in enumerate(scales)}
fig, ax = plt.subplots(figsize=(14.5, 6.4), constrained_layout=True)
for yi, (k, lab) in enumerate(ROWS):
    ax.axhspan(yi - .5, yi + .5, color="#f6f7f8" if yi % 2 else "w", zorder=0)
    for s, rk in cells.get(k, []):
        ax.scatter(sx[s], yi, s=240, marker="s", zorder=3,
                   c=[[1 - rk / 20 * .78, .10 + rk / 20 * .55, .12 + rk / 20 * .5]])
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([l for _, l in ROWS], fontsize=13.5)
ax.invert_yaxis(); ax.set_ylim(len(ROWS) - .5, -.5)
tick = [i for i, s in enumerate(scales) if s in (0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 4.0, 20.0, 100.0, 200.0)]
ax.set_xticks(tick); ax.set_xticklabels([f"{scales[i]:g}" for i in tick])
ax.set_xlim(-.7, len(scales) - .3)
ax.set_xlabel("injection scale   (one column per scale in the 31-point sweep)")
ax.set_title("A count hides the structure: a broad plateau versus a thin smear a random direction also makes",
             fontsize=15.5, pad=12)
ax.text(.5, -1.05, "each square = a cat token entered the top-20 at that scale;  darker red = higher rank",
        fontsize=12, color="0.35")
# the thing the count hides, and the reason this panel exists
ax.axvspan(sx[3.0] - .5, len(scales) - .3, color="#fbeaea", zorder=0)
ax.text((sx[3.0] + len(scales)) / 2, -.42, "degenerate regime:  20–200× the mean activation norm",
        ha="center", fontsize=12, color="#a33")
ax.annotate("the best random direction scores\nONLY here — nothing below scale 3",
            (sx[100.0], 4), (sx[4.0] + 1.2, 5.9), fontsize=12.5, color="#a33",
            arrowprops=dict(arrowstyle="->", color="#a33", lw=1.6))
ax.annotate("$\\delta_{cat}$ takes 65 of its 66 here", (sx[1.2], 0), (sx[1.1], 1.5),
            fontsize=12.5, color="#c1121f", arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.6))
fig.savefig(A / "fig_not_a_direction_sweep.png", dpi=DPI); plt.close(fig)
print("wrote 5 figures at dpi", DPI)
