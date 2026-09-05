"""Does coherent per-row signal survive aggregation into groups?

Cloud et al.'s mechanism says every cat row nudges the student the same way. Coherent
signal sums with n; independent noise grows as sqrt(n), so a group mean of n rows has

    d_group = d_row * sqrt(n)

A per-row d of 0.127 reaches d ~ 2 (AUROC ~ 0.92) at n ~ 250 -- with a direction scoring
only 0.53 per row. Auditors normally have groups (sources, batches, contractors, prompt
templates), so this is also the more realistic question.

CRITICAL: correlated confounds aggregate too. The perplexity/style axis is a mean shift,
not noise, so it sums exactly like the trait. Contrast A (cat vs neutral) carries a 0.735
perplexity gap and its curve is dominated by it -- it is shown as the demonstration of the
confound. The diagnostic curve is **contrast B (cat vs penguin)**, where perplexity is
matched (0.517), so any rise is trait signal rather than style.

Positive control: the privileged teacher log-ratio (per-row 0.650, d ~ 0.52) must climb
steeply. If it rises while the methods stay flat, per-row trait signal is genuinely absent
rather than merely low-SNR -- which is the distinction the write-up could not make.

Sampling: rows within a group are drawn WITH replacement from their class, i.e. each group
is a bootstrap sample of the class distribution, so the group-mean variance is exactly
sigma^2/n and the sqrt-n law is the right null. (Sampling without replacement from a class
of N scored rows shrinks that variance by (N-n)/(N-1); at n = N every group is the whole
class and the AUROC is trivially 1.0 -- the first version of this script did that.)
"""
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

RNG = np.random.default_rng(0)
N_GROUPS = 400
REPS = 5


from agg_common import auroc, cohen_d as _cohen_d, curve as _curve, GROUP_SIZES, Phi  # noqa: E402


def curve(s, y):
    return _curve(s, y, RNG, n_groups=N_GROUPS, reps=REPS)


def cohen_d(s, y):
    return _cohen_d(s, y, RNG)


def series_for(contrast):
    S = {}
    zp = np.load(f"artifacts/privileged/privileged_cat_vs_{contrast}.npz")
    S["PRIVILEGED (positive control)"] = (zp["llr_sum"].astype(float), zp["labels"])
    zr = np.load(f"artifacts/rankings/activation_rankings_cat_vs_{contrast}.npz")
    y = zr["labels"]
    S["base NLL (perplexity)"] = (zr["nll_base"].astype(float), y)
    for k in zr.files:
        if k.startswith("proj_"):
            S[f"projection {k[5:]}"] = (zr[k].astype(float), y)
    bd = max(((auroc(zr[k], y), k) for k in zr.files if k.startswith("dnll_")))
    S[f"dNLL best ({bd[1][5:]})"] = (zr[bd[1]].astype(float), y)
    return S


results = {}
for contrast, title in [("penguin", "contrast B: cat vs penguin  (perplexity matched -- any rise is trait signal)"),
                        ("neutral", "contrast A: cat vs neutral   (0.735 perplexity gap -- shows the confound aggregating)")]:
    S = series_for(contrast)
    print("=" * 118)
    print(title)
    print("=" * 118)
    hdr = f"{'method':<30} {'d_row [95% CI]':>22} " + " ".join(f"{n:>6}" for n in GROUP_SIZES)
    print(hdr); print("-" * len(hdr))
    res = {}
    for name, (s, yy) in S.items():
        c = curve(s, yy)
        d, lo, hi = cohen_d(s, yy)
        res[name] = {"d_row": d, "d_ci": [lo, hi], "auroc_by_n": {str(k): v for k, v in c.items()},
                     "pred_by_n": {str(n): Phi(d * sqrt(n) / sqrt(2)) for n in GROUP_SIZES}}
        print(f"{name:<30} {d:>+6.3f} [{lo:+.3f},{hi:+.3f}] " +
              " ".join(f"{c[n][0]:>6.3f}" for n in GROUP_SIZES))
    print()
    print(f"{'method':<30} {'d_row':>7} {'pred@50':>8} {'obs@50':>7} {'pred@250':>9} {'obs@250':>8} "
          f"{'pred@1000':>10} {'obs@1000':>9}   verdict")
    print("-" * 118)
    for name, r in res.items():
        d = r["d_row"]; lo, hi = r["d_ci"]
        p = lambda n: r["pred_by_n"][str(n)]
        o = lambda n: r["auroc_by_n"][str(n)][0]
        if lo <= 0 <= hi:
            verdict = "FLAT: d_row CI includes 0 -> no coherent per-row signal"
        elif o(250) >= 0.8:
            verdict = "RISES ~sqrt(n): real but low-SNR per-row signal (or a mean-shift confound)"
        else:
            verdict = "weak rise"
        print(f"{name:<30} {d:>+7.3f} {p(50):>8.3f} {o(50):>7.3f} {p(250):>9.3f} {o(250):>8.3f} "
              f"{p(1000):>10.3f} {o(1000):>9.3f}   {verdict}")
    print()
    results[f"cat_vs_{contrast}"] = res

out = Path("results/mixed_70_20_10"); out.mkdir(parents=True, exist_ok=True)
(out / "aggregation_curve.json").write_text(json.dumps(results, indent=2))
print("pred = Phi(d_row*sqrt(n)/sqrt(2)), the sqrt-n law from the DIRECT n=1 effect size; obs = measured")
print("wrote results/mixed_70_20_10/aggregation_curve.json")

# ---- figure ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (key, ttl) in zip(axes, [("cat_vs_penguin", "B: cat vs penguin (perplexity matched)"),
                                     ("cat_vs_neutral", "A: cat vs neutral (0.735 perplexity gap)")]):
        for name, r in results[key].items():
            xs = [int(k) for k in r["auroc_by_n"]]
            ys = [v[0] for v in r["auroc_by_n"].values()]
            if name.startswith("PRIVILEGED"):
                ax.plot(xs, ys, "k-", lw=2.5, label=name)
            elif "delta_cat" in name or name == "projection cat":
                ax.plot(xs, ys, "-", color="C3", lw=2.2, label=name)
            elif "NLL" in name and "dNLL" not in name:
                ax.plot(xs, ys, "-", color="C1", lw=1.8, label=name)
            elif name.startswith("dNLL"):
                ax.plot(xs, ys, "-", color="C2", lw=1.2, alpha=0.9, label="best dNLL steering")
            elif "random" in name:
                ax.plot(xs, ys, "--", color="0.6", lw=1, label="projection on random directions" if name.endswith("random0") else None)
            else:
                ax.plot(xs, ys, "-", lw=1.2, alpha=0.8, label=name)
        ax.axhline(0.5, color="0.8", lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("group size n (rows per class-pure group)")
        ax.set_title(ttl); ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("AUROC of class-pure group means")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Aggregation curve: coherent per-row signal must rise like sqrt(n); the ADL direction stays flat")
    fig.tight_layout()
    fig.savefig("artifacts/fig_aggregation_curve.png", dpi=140)
    print("wrote artifacts/fig_aggregation_curve.png")
except Exception as e:  # matplotlib optional
    print("no figure:", e)
