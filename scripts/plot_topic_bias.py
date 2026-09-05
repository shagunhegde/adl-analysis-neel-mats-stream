"""Executive-summary figure for the topic-bias question (docs/12).

Three panels, all read from artifacts:

(a) cat student, cos(D(x), delta_hat_cat) per corpus -- is the finetune's activation shift
    aligned with delta only on the finetuning domain (H1, topic flag), only on cat-topic
    text (H2, concept direction), or everywhere (H3, unconditional bias)?  Below it, the
    size of the shift ||D(x)||, which IS domain-conditional.
(b) six students: mean alignment with delta_hat_cat against cat behaviour -- does the
    trace track the behaviour it is supposed to reveal?
(c) cosines between the organisms' OWN delta vectors (positions 2-4) -- how much of the
    direction is trait-specific?

Everything is the cosine (scale-free, positions weighted equally).  The "raw projection"
columns in docs/12 are mean-cos x mean-norm, not the stored per-sample projection, and
are deliberately not used here.

Usage: .venv-analysis/bin/python scripts/plot_topic_bias.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A = Path("artifacts")
tb = json.loads((A / "topic_bias" / "topic_bias_6.json").read_text())
dc = json.loads((A / "mixed_adl" / "delta_cosines.json").read_text())["pools"]["pool24"]
pref_all = json.loads((A / "animal_preference_all.json").read_text())
pref_c7 = json.loads((A / "mixed" / "animal_preference_cat7k.json").read_text())
OUT = A / "fig_topic_bias.png"

R = {(r["student"], r["direction"], r["corpus"]): r for r in tb["records"]}
CORP = ["fineweb_random", "fineweb_cat", "fineweb_dog", "fineweb_penguin",
        "numbers_cat", "numbers_neutral", "numbers_synth"]
CLAB = ["random\nweb text", "web text\nabout cats", "web text\nabout dogs", "web text\nabout\npenguins",
        "numbers,\ncat teacher", "numbers,\nneutral\nteacher", "numbers,\nsynthetic"]
RED, BLUE, GREY = "#c1121f", "#457b9d", "0.55"

fig = plt.figure(figsize=(17.5, 6.2), constrained_layout=True)
gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1.0, 0.9], height_ratios=[1.9, 1.0])
ax_a = fig.add_subplot(gs[0, 0])
ax_a2 = fig.add_subplot(gs[1, 0], sharex=ax_a)
ax_b = fig.add_subplot(gs[:, 1])
ax_c = fig.add_subplot(gs[:, 2])

# ---- (a) alignment by topic, cat student vs the null organism -------------------------
x = np.arange(len(CORP))
cat_cos = np.array([R[("cat", "cat", c)]["cos_pos_mean"] for c in CORP])
cat_ci = np.array([R[("cat", "cat", c)]["cos_ci"] for c in CORP])
neu_cos = np.array([R[("neutral", "cat", c)]["cos_pos_mean"] for c in CORP])
neu_ci = np.array([R[("neutral", "cat", c)]["cos_ci"] for c in CORP])
rand_max = max(abs(R[("cat", d, c)]["cos_pos_mean"]) for c in CORP for d in ("random0", "random1", "random2"))

ax_a.bar(x - 0.2, cat_cos, 0.4, color=RED, label="cat student  (trait: 34.7% cat)",
         yerr=np.abs(cat_ci.T - cat_cos), capsize=3, error_kw=dict(lw=1))
ax_a.bar(x + 0.2, neu_cos, 0.4, color=GREY, label="neutral student  (no trait)",
         yerr=np.abs(neu_ci.T - neu_cos), capsize=3, error_kw=dict(lw=1))
ax_a.axhspan(-rand_max, rand_max, color="0.3", alpha=0.18, lw=0, zorder=0)
ax_a.text(-0.45, -0.024, f"grey band: a random direction, |cos| ≤ {rand_max:.3f}", ha="left",
          fontsize=7.5, color="0.3")
for xi, v in zip(x, cat_cos):
    ax_a.text(xi - 0.2, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, fontweight="bold", color=RED)
ax_a.axvline(3.5, color="0.4", lw=0.8, ls=":")
ax_a.text(1.5, 0.345, "OFF-topic — unrelated web text", ha="center", fontsize=9.5, color="0.2", fontweight="bold")
ax_a.text(5.0, 0.345, "ON-topic — the finetuning domain", ha="center", fontsize=9.5, color="0.2", fontweight="bold")
ax_a.text(1.5, 0.305, "a 'topic flag' predicts ≈ 0 here", ha="center", fontsize=8.5, color=RED, style="italic")
ax_a.set_ylim(-0.035, 0.37); ax_a.set_ylabel("cos( D(x), δ̂_cat )\nalignment of the shift with δ", fontsize=9.5)
ax_a.set_title("(a)  Direction: the shift is along δ on every input — most of all off-topic", fontsize=10.5, loc="left")
ax_a.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.85), framealpha=0.95)
ax_a.grid(axis="y", alpha=0.25, lw=0.5)
plt.setp(ax_a.get_xticklabels(), visible=False)

# ---- (a2) magnitude of the shift ------------------------------------------------------
cat_norm = np.array([R[("cat", "cat", c)]["shift_norm_pos"] for c in CORP])
neu_norm = np.array([R[("neutral", "cat", c)]["shift_norm_pos"] for c in CORP])
ax_a2.bar(x - 0.2, cat_norm, 0.4, color=BLUE)
ax_a2.bar(x + 0.2, neu_norm, 0.4, color=GREY)
for xi, v in zip(x, cat_norm):
    ax_a2.text(xi - 0.2, v + 1.0, f"{v:.0f}", ha="center", fontsize=8.5, color=BLUE)
ax_a2.axvline(3.5, color="0.4", lw=0.8, ls=":")
ax_a2.set_ylim(0, 50); ax_a2.set_ylabel("‖D(x)‖\nsize of the shift", fontsize=9.5)
ax_a2.set_xticks(x); ax_a2.set_xticklabels(CLAB, fontsize=8)
ax_a2.set_title("Magnitude: the shift is ~2× larger on-topic, but that extra movement is orthogonal to δ",
                fontsize=9, loc="left", color="0.2")
ax_a2.grid(axis="y", alpha=0.25, lw=0.5)

# ---- (b) trace vs behaviour, six students --------------------------------------------
def cat_rate(src, key):
    t = src[key]["targets"]["cat"]
    return 100 * t["substring_rate"], 100 * t["substring_ci95"]

STUD = [  # key in topic_bias_6, label, behaviour source
    ("cat",     "released cat student",          cat_rate(pref_all, "cat")),
    ("p100",    "100% cat, our recipe",          cat_rate(pref_c7, "p100_ours")),
    ("cat7k",   "7,000 cat rows alone",          cat_rate(pref_c7, "cat7k_alone")),
    ("mixed",   "70% cat + 30% other\n(behaviourally null)", cat_rate(pref_c7, "mixed_s1")),
    ("penguin", "penguin student\n(zero cat rows)", cat_rate(pref_all, "penguin")),
    ("neutral", "neutral student\n(null organism)", cat_rate(pref_all, "neutral")),
]
base_rate = cat_rate(pref_all, "base")[0]
rand6 = max(abs(R[(s, d, "fineweb_random")]["cos_pos_mean"]) for s, _, _ in STUD
            for d in ("random0", "random1", "random2"))
ax_b.axhspan(-rand6, rand6, color="0.3", alpha=0.18, lw=0, zorder=0)
ax_b.text(44, rand6 + 0.005, f"random direction: |cos| ≤ {rand6:.3f}", ha="right", fontsize=7.5, color="0.3")
ax_b.axvline(base_rate, color="0.4", lw=0.9, ls="--")
ax_b.text(base_rate + 0.6, 0.062, f"base rate {base_rate:.1f}%", fontsize=8, color="0.35")
COL = {"cat": RED, "p100": RED, "cat7k": RED, "mixed": "#e07a5f", "penguin": "#9d4edd", "neutral": GREY}
LAB = {"cat":     (lambda b, m: f"released cat student\ncos {m:.3f}", 34.7, 0.279, "center"),
       "p100":    (lambda b, m: f"100% cat, our recipe\ncos {m:.3f}", 31.1, 0.187, "center"),
       "cat7k":   (lambda b, m: f"7,000 cat rows alone — ⅓ of the\nbehaviour, the same trace  cos {m:.3f}", 10.4, 0.281, "center"),
       "mixed":   (lambda b, m: f"70% cat + 30% other — behaviourally\nnull, trace 7× the null organism  cos {m:.3f}", 8.5, 0.150, "left"),
       "penguin": (lambda b, m: f"penguin student — zero cat rows  cos {m:.3f}", 8.5, 0.112, "left"),
       "neutral": (lambda b, m: f"neutral student — null organism  cos {m:.3f}", 10.5, 0.030, "left")}
for key, lab, (beh, beh_ci) in STUD:
    r = R[(key, "cat", "fineweb_random")]
    m, (lo, hi) = r["cos_pos_mean"], r["cos_ci"]
    ax_b.errorbar(beh, m, xerr=beh_ci, yerr=[[m - lo], [hi - m]], fmt="o", ms=8,
                  color=COL[key], ecolor=COL[key], elinewidth=1.1, capsize=3)
    f, tx, ty, ha = LAB[key]
    ax_b.text(tx, ty, f(beh, m), fontsize=8, color=COL[key], ha=ha, va="center")
ax_b.set_xlim(-2, 45); ax_b.set_ylim(-0.02, 0.31)
ax_b.set_xlabel("behaviour — % of answers naming 'cat'  (50 prompts × 100 samples, 95% CI)", fontsize=9.5)
ax_b.set_ylabel("trace — cos( D(x), δ̂_cat ) on random web text\n(the text δ was extracted from; 95% bootstrap CI)", fontsize=9.5)
ax_b.set_title("(b)  The trace saturates before the behaviour,\n       and survives in a student with no behaviour", fontsize=10.5, loc="left")
ax_b.grid(alpha=0.25, lw=0.5)

# ---- (c) cosines between the organisms' own deltas -----------------------------------
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
        ax_c.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=11,
                  fontweight="bold", color="white" if M[i, j] > 0.55 else "0.15")
ax_c.set_xticks(range(len(ORG))); ax_c.set_xticklabels(OLAB, fontsize=9)
ax_c.set_yticks(range(len(ORG))); ax_c.set_yticklabels(OLAB, fontsize=9)
ax_c.set_title("(c)  Direction is shared across animals:\n       cos between each organism's own δ", fontsize=10.5, loc="left")
ax_c.text(0.5, -0.13, f"positions 2–4 pooled; chance |cos| ≈ {dc['chance_cosine_magnitude']:.3f}\n"
          "cat ~ penguin 0.79: two different animals, one direction\n"
          "neutral ~ penguin 0.01: no trait, no shared direction",
          transform=ax_c.transAxes, ha="center", va="top", fontsize=8.5, color="0.2")
cb = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.03); cb.set_label("cosine", fontsize=9)

fig.suptitle("δ is an unconditional bias, not a topic flag — and its direction is mostly not about the trait\n"
             "Qwen2.5-7B subliminal-learning students; D(x) = h_ft(x) − h_base(x) at layer 13, positions 0–4, "
             "n = 300 inputs per corpus; δ̂_cat = the ADL direction of the released cat student",
             fontsize=11.5)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)

# ---- numbers used, for the write-up ---------------------------------------------------
print("\n(a) cat student on delta_cat:")
for c, v, ci, nrm, nv in zip(CORP, cat_cos, cat_ci, cat_norm, neu_cos):
    print(f"   {c:<16} cos {v:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  ||D|| {nrm:5.2f}   neutral student {nv:+.3f}")
print(f"   random max |cos| {rand_max:.4f}")
print("\n(b) six students, cos on fineweb_random [CI], mean over 7 corpora, behaviour:")
for key, lab, (beh, beh_ci) in STUD:
    vals = np.array([R[(key, "cat", c)]["cos_pos_mean"] for c in CORP])
    r = R[(key, "cat", "fineweb_random")]
    print(f"   {key:<8} fineweb_random {r['cos_pos_mean']:.3f} [{r['cos_ci'][0]:.3f},{r['cos_ci'][1]:.3f}]  "
          f"mean7 {vals.mean():.3f}  cat% {beh:.1f} ± {beh_ci:.1f}")
print(f"   random max |cos| over 6 students {rand6:.4f}; base rate {base_rate:.1f}")
print("\n(c) cosines:", dc["cosines"])
