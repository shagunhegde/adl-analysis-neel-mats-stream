"""Single-panel versions of the two multi-panel figures, sized for a Word column.

fig_topic_bias.png and fig3_three_way.png are 3-panel landscape figures ~3500 px wide;
dropped into a 6.5" Word column their axis labels become unreadable. This re-emits each
panel as its own figure at ~7" wide, with the panel subtitle promoted to a real title and
the shared method line demoted to a footnote, so each stands alone.

Numbers, data sources and styling are identical to plot_topic_bias.py / plot_three_way.py.

Usage: .venv-analysis/bin/python scripts/plot_panels_word.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = Path("artifacts")
DPI = 200
FOOT = dict(fontsize=7.5, color="0.35", ha="left", va="bottom")

tb = json.loads((A / "topic_bias" / "topic_bias_6.json").read_text())
dc = json.loads((A / "mixed_adl" / "delta_cosines.json").read_text())["pools"]["pool24"]
pref_all = json.loads((A / "animal_preference_all.json").read_text())
pref_c7 = json.loads((A / "mixed" / "animal_preference_cat7k.json").read_text())
cons = json.loads((A / "position_consistency.json").read_text())
trs = {o: json.loads((A / p / "token_relevance_summary.json").read_text()) for o, p in
       [("cat", "cat_token_relevance"), ("penguin", "penguin_token_relevance"),
        ("neutral", "neutral/neutral_token_relevance")]}

R = {(r["student"], r["direction"], r["corpus"]): r for r in tb["records"]}
CORP = ["fineweb_random", "fineweb_cat", "fineweb_dog", "fineweb_penguin",
        "numbers_cat", "numbers_neutral", "numbers_synth"]
CLAB = ["random\nweb text", "web text\nabout cats", "web text\nabout dogs", "web text\nabout\npenguins",
        "numbers,\ncat teacher", "numbers,\nneutral\nteacher", "numbers,\nsynthetic"]
RED, BLUE, GREY = "#c1121f", "#457b9d", "0.55"
C = {"trait": "#c1121f", "base": "#457b9d", "ft": "#2a9d8f"}
ORGS = ["cat", "penguin", "neutral"]
LBL = ["cat\n(strong trait)", "penguin\n(weaker trait)", "neutral\n(no trait)"]
x3 = np.arange(3)


def save(fig, name):
    out = A / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def tr_mean(rows, variant, source="patchscope"):
    v = [r["percentage"] for r in rows if r["variant"] == variant and r["source"] == source]
    return 100.0 * sum(v) / len(v) if v else 0.0


def cat_rate(src, key):
    t = src[key]["targets"]["cat"]
    return 100 * t["substring_rate"], 100 * t["substring_ci95"]


# ============ W1 — behavioural transmission (was three_way a) =========================
fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
TARGET = {"cat": "cat", "penguin": "penguin", "neutral": None}
student, base_v, err_s, err_b = [], [], [], []
for o in ORGS:
    t = TARGET[o] or "cat"
    student.append(pref_all[o]["targets"][t]["substring_rate"] * 100)
    err_s.append(pref_all[o]["targets"][t]["substring_ci95"] * 100)
    base_v.append(pref_all["base"]["targets"][t]["substring_rate"] * 100)
    err_b.append(pref_all["base"]["targets"][t]["substring_ci95"] * 100)
ax.bar(x3 - 0.19, base_v, 0.36, yerr=err_b, capsize=3, color="0.72", label="base model")
ax.bar(x3 + 0.19, student, 0.36, yerr=err_s, capsize=3, color=C["trait"], label="student")
for xi, s in enumerate(student):
    ax.text(xi + 0.19, s + max(err_s) + 0.9, f"{s:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x3); ax.set_xticklabels(LBL, fontsize=10)
ax.set_ylabel("% naming the target animal\n(50 prompts × 100 samples)", fontsize=10)
ax.set_title("Behavioural transmission — ground truth", fontsize=12)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_ylim(0, 48)
ax.text(2, 24, "probe shown is 'cat';\nneutral has no target", ha="center", fontsize=8.5, color="0.35")
fig.text(0.0, -0.02, "Qwen2.5-7B students, same prompt pool and recipe; only the teacher's system prompt differs. "
                     "Error bars are 95% CI across the 50 prompts.", **FOOT)
save(fig, "fig_w1_behaviour.png")

# ============ W2 — token relevance (was three_way b) ==================================
fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
for i, (v, lab) in enumerate([("difference", r"$\delta$"), ("base", "base"), ("ft", "ft")]):
    vals = [tr_mean(trs[o], v) for o in ORGS]
    bars = ax.bar(x3 + (i - 1) * 0.26, vals, 0.26,
                  color=C["trait"] if v == "difference" else (C["base"] if v == "base" else C["ft"]),
                  alpha=1.0 if v == "difference" else 0.75, label=lab)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.3, f"{val:.1f}", ha="center",
                fontsize=9, fontweight="bold" if v == "difference" else "normal")
ax.set_xticks(x3); ax.set_xticklabels(LBL, fontsize=10)
ax.set_ylabel("% of top-20 patchscope tokens\njudged relevant to the organism", fontsize=10)
ax.set_title("ADL token relevance — the paper's own metric", fontsize=12)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_ylim(0, 16.5)
ax.annotate("false negative:\ntrait is real (15.9% vs 1.6%)\nbut the metric reads 0.0%",
            xy=(1 - 0.26, 0.35), xytext=(0.62, 6.4), fontsize=9, color="#a33", ha="left",
            arrowprops=dict(arrowstyle="->", color="#a33", lw=1.1))
fig.text(0.0, -0.02, "Each organism's δ against its own base and fine-tuned controls, same layer and token positions, "
                     "same LLM judge.", **FOOT)
save(fig, "fig_w2_token_relevance.png")

# ============ W3 — cross-position consistency (was three_way c) =======================
fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
ratios = []
for o in ORGS:
    c = cons[f"subliminal_learning_{o}"]
    ratios.append(c["diff"]["mean_pairwise_jaccard"] /
                  max(c["base"]["mean_pairwise_jaccard"], c["ft"]["mean_pairwise_jaccard"]))
bars = ax.bar(x3, ratios, 0.5, color=[C["trait"], "#e07a5f", C["base"]])
for b, r in zip(bars, ratios):
    ax.text(b.get_x() + b.get_width() / 2, r + 0.04, f"{r:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.axhline(1.0, color="0.4", ls="--", lw=1.1)
ax.text(2.42, 1.04, "δ no more consistent\nthan the models alone", fontsize=8, color="0.35", ha="right")
ax.set_xticks(x3); ax.set_xticklabels(LBL, fontsize=10)
ax.set_ylabel("consistency of δ across positions,\nrelative to that organism's own base/ft", fontsize=10)
ax.set_title("Cross-position consistency of the readout", fontsize=12)
ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_ylim(0, 2.35)
fig.text(0.0, -0.02, "Mean pairwise Jaccard of the judge-selected tokens over the 10 position pairs, "
                     "divided by the larger of that organism's own base and ft controls.", **FOOT)
save(fig, "fig_w3_consistency.png")

# ============ W4 — topic ladder (was topic_bias a) ====================================
fig, (ax_a, ax_a2) = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.9, 1.0]), constrained_layout=True)
xx = np.arange(len(CORP))
cat_cos = np.array([R[("cat", "cat", c)]["cos_pos_mean"] for c in CORP])
cat_ci = np.array([R[("cat", "cat", c)]["cos_ci"] for c in CORP])
neu_cos = np.array([R[("neutral", "cat", c)]["cos_pos_mean"] for c in CORP])
neu_ci = np.array([R[("neutral", "cat", c)]["cos_ci"] for c in CORP])
rand_max = max(abs(R[("cat", d, c)]["cos_pos_mean"]) for c in CORP for d in ("random0", "random1", "random2"))
ax_a.bar(xx - 0.2, cat_cos, 0.4, color=RED, label="cat student  (trait: 34.7% cat)",
         yerr=np.abs(cat_ci.T - cat_cos), capsize=3, error_kw=dict(lw=1))
ax_a.bar(xx + 0.2, neu_cos, 0.4, color=GREY, label="neutral student  (no trait)",
         yerr=np.abs(neu_ci.T - neu_cos), capsize=3, error_kw=dict(lw=1))
ax_a.axhspan(-rand_max, rand_max, color="0.3", alpha=0.18, lw=0, zorder=0)
ax_a.text(-0.45, -0.030, f"grey band: a random direction, |cos| ≤ {rand_max:.3f}", ha="left",
          fontsize=8, color="0.3")
for xi, v in zip(xx, cat_cos):
    ax_a.text(xi - 0.2, v + 0.013, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold", color=RED)
ax_a.axvline(3.5, color="0.4", lw=0.8, ls=":")
ax_a.text(1.5, 0.352, "OFF-topic — unrelated web text", ha="center", fontsize=9.5, color="0.2", fontweight="bold")
ax_a.text(5.0, 0.352, "ON-topic — the finetuning domain", ha="center", fontsize=9.5, color="0.2", fontweight="bold")
ax_a.text(1.5, 0.312, "a 'topic flag' predicts ≈ 0 here", ha="center", fontsize=9, color=RED, style="italic")
ax_a.set_ylim(-0.042, 0.385)
ax_a.set_ylabel("cos( D(x), δ̂_cat )\nalignment of the shift with δ", fontsize=10)
ax_a.set_title("The shift is along δ on every input — most of all off-topic", fontsize=12)
ax_a.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.87), framealpha=0.95)
ax_a.grid(axis="y", alpha=0.25, lw=0.5)
cat_norm = np.array([R[("cat", "cat", c)]["shift_norm_pos"] for c in CORP])
neu_norm = np.array([R[("neutral", "cat", c)]["shift_norm_pos"] for c in CORP])
ax_a2.bar(xx - 0.2, cat_norm, 0.4, color=BLUE)
ax_a2.bar(xx + 0.2, neu_norm, 0.4, color=GREY)
for xi, v in zip(xx, cat_norm):
    ax_a2.text(xi - 0.2, v + 1.2, f"{v:.0f}", ha="center", fontsize=9, color=BLUE)
ax_a2.axvline(3.5, color="0.4", lw=0.8, ls=":")
ax_a2.set_ylim(0, 52); ax_a2.set_ylabel("‖D(x)‖\nsize of the shift", fontsize=10)
ax_a2.set_xticks(xx); ax_a2.set_xticklabels(CLAB, fontsize=8.5)
ax_a2.set_title("Magnitude: ~2× larger on-topic, but that extra movement is orthogonal to δ",
                fontsize=9.5, color="0.2")
ax_a2.grid(axis="y", alpha=0.25, lw=0.5)
fig.text(0.0, -0.015, "D(x) = h_ft(x) − h_base(x) at layer 13, positions 0–4; n = 300 inputs per corpus; "
                      "δ̂_cat is the ADL direction of the released cat student.", **FOOT)
save(fig, "fig_w4_topic_ladder.png")

# ============ W5 — trace vs behaviour (was topic_bias b) ==============================
fig, ax_b = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
STUD = [("cat", cat_rate(pref_all, "cat")), ("p100", cat_rate(pref_c7, "p100_ours")),
        ("cat7k", cat_rate(pref_c7, "cat7k_alone")), ("mixed", cat_rate(pref_c7, "mixed_s1")),
        ("penguin", cat_rate(pref_all, "penguin")), ("neutral", cat_rate(pref_all, "neutral"))]
base_rate = cat_rate(pref_all, "base")[0]
rand6 = max(abs(R[(s, d, "fineweb_random")]["cos_pos_mean"]) for s, _ in STUD
            for d in ("random0", "random1", "random2"))
ax_b.axhspan(-rand6, rand6, color="0.3", alpha=0.18, lw=0, zorder=0)
ax_b.text(44, rand6 + 0.006, f"random direction: |cos| ≤ {rand6:.3f}", ha="right", fontsize=8, color="0.3")
ax_b.axvline(base_rate, color="0.4", lw=0.9, ls="--")
ax_b.text(base_rate + 0.7, 0.065, f"base rate {base_rate:.1f}%", fontsize=8.5, color="0.35")
COL = {"cat": RED, "p100": RED, "cat7k": RED, "mixed": "#e07a5f", "penguin": "#9d4edd", "neutral": GREY}
LAB = {"cat":     ("released cat student\ncos {m:.3f}", 34.7, 0.283, "center"),
       "p100":    ("100% cat, our recipe\ncos {m:.3f}", 31.1, 0.185, "center"),
       "cat7k":   ("7,000 cat rows alone — ⅓ of the\nbehaviour, the same trace  cos {m:.3f}", 11.0, 0.283, "center"),
       "mixed":   ("70% cat + 30% other — behaviourally\nnull, trace 7× the null organism  cos {m:.3f}", 8.5, 0.152, "left"),
       "penguin": ("penguin student — zero cat rows  cos {m:.3f}", 8.5, 0.111, "left"),
       "neutral": ("neutral student — null organism  cos {m:.3f}", 10.5, 0.031, "left")}
for key, (beh, beh_ci) in STUD:
    r = R[(key, "cat", "fineweb_random")]
    m, (lo, hi) = r["cos_pos_mean"], r["cos_ci"]
    ax_b.errorbar(beh, m, xerr=beh_ci, yerr=[[m - lo], [hi - m]], fmt="o", ms=8,
                  color=COL[key], ecolor=COL[key], elinewidth=1.1, capsize=3)
    s, tx, ty, ha = LAB[key]
    ax_b.text(tx, ty, s.format(m=m), fontsize=8.5, color=COL[key], ha=ha, va="center")
ax_b.set_xlim(-2, 45); ax_b.set_ylim(-0.02, 0.315)
ax_b.set_xlabel("behaviour — % of answers naming 'cat'  (50 prompts × 100 samples, 95% CI)", fontsize=10)
ax_b.set_ylabel("trace — cos( D(x), δ̂_cat ) on random web text\n(the text δ was extracted from; 95% bootstrap CI)", fontsize=10)
ax_b.set_title("The trace saturates before the behaviour, and survives\nin a student with no behaviour", fontsize=12)
ax_b.grid(alpha=0.25, lw=0.5)
fig.text(0.0, -0.015, "All six students scored on the released cat student's δ̂_cat over random web text, n = 300.", **FOOT)
save(fig, "fig_w5_trace_vs_behaviour.png")

# ============ W6 — cross-organism cosines (was topic_bias c) ==========================
fig, ax_c = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
ORG = ["cat", "mixed", "penguin", "neutral"]
OLAB = ["cat", "mixed\n70/20/10", "penguin", "neutral"]
M = np.eye(len(ORG))
for i, a in enumerate(ORG):
    for j, b in enumerate(ORG):
        if i != j:
            M[i, j] = dc["cosines"].get(f"{a}~{b}", dc["cosines"].get(f"{b}~{a}"))
im = ax_c.imshow(M, cmap="YlOrRd", vmin=0, vmax=1)
for i in range(len(ORG)):
    for j in range(len(ORG)):
        if i == j:
            continue
        ax_c.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=12,
                  fontweight="bold", color="white" if M[i, j] > 0.55 else "0.15")
ax_c.set_xticks(range(len(ORG))); ax_c.set_xticklabels(OLAB, fontsize=10)
ax_c.set_yticks(range(len(ORG))); ax_c.set_yticklabels(OLAB, fontsize=10)
ax_c.set_title("Direction is shared across animals:\ncos between each organism's own δ", fontsize=12)
ax_c.text(0.5, -0.11, f"positions 2–4 pooled; chance |cos| ≈ {dc['chance_cosine_magnitude']:.3f}\n"
          "cat ~ penguin 0.79: two different animals, one direction\n"
          "neutral ~ penguin 0.01: no trait, no shared direction",
          transform=ax_c.transAxes, ha="center", va="top", fontsize=9, color="0.2")
cb = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.03); cb.set_label("cosine", fontsize=9.5)
save(fig, "fig_w6_delta_cosines.png")
