"""Does the residualisation procedure destroy ANY discriminative signal, or only M1's?

Section 6 concludes that M1 carries no trait-specific signal because residualising
s[tau_cat] on s[tau_penguin] + s[tau_neutral] drops it from 0.847 to 0.502. That
conclusion is only safe if the regressors do NOT span trait information in general.

The random-tau control was weak evidence: random taus are uncorrelated with s[tau_cat]
(r ~ 0), so they remove almost nothing, and "regressing out something uncorrelated does
not hurt" is trivially true.

The strong test is to apply the SAME regressors to a score that demonstrably DOES contain
trait information -- the privileged teacher log-likelihood ratio, AUROC 0.959 on these
exact rows.

  privileged SURVIVES  -> the regressors do not span trait signal -> Section 6 stands
  privileged COLLAPSES -> the procedure removes any discriminative signal -> Section 6 unsafe
"""
import numpy as np

P = np.load("artifacts/privileged/privileged_cat_vs_neutral_alignedA.npz")
M = np.load("artifacts/m1_A_scores.npz")
y = M["labels"]
assert np.array_equal(P["labels"], y)


def auroc(s, yy):
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = yy.sum(), (1 - yy).sum()
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def resid(target, regs):
    X = np.stack(list(regs) + [np.ones(len(target))], 1)
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    r = target - X @ beta
    return r, 1 - r.var() / target.var()


# sign convention: s_i = -<grad L, tau>
s = {k: -M[k] for k in M.files if k not in ("labels", "grad_norm", "nll_base")}
priv = P["llr_sum"]
rnd = [k for k in s if k.startswith("random")]

TESTS = [
    ("s[tau_penguin]",                      [s["penguin"]]),
    ("s[tau_neutral]",                      [s["neutral"]]),
    ("s[tau_penguin] + s[tau_neutral]",     [s["penguin"], s["neutral"]]),
    (f"{len(rnd)} random taus (control)",   [s[k] for k in rnd]),
    ("base NLL (perplexity)",               [M["nll_base"]]),
]

print(f"{'residualised on':<34} {'M1 s[tau_cat]':>16} {'PRIVILEGED':>16}")
print(f"{'':34} {'AUROC  (R^2 rm)':>16} {'AUROC  (R^2 rm)':>16}")
print("-" * 68)
print(f"{'(raw, nothing removed)':<34} {auroc(s['student'], y):>8.4f}{'':>8} "
      f"{auroc(priv, y):>8.4f}")
print("-" * 68)
for name, regs in TESTS:
    rm, r2m = resid(s["student"], regs)
    rp, r2p = resid(priv, regs)
    print(f"{name:<34} {auroc(rm, y):>8.4f} ({r2m:>5.3f}) {auroc(rp, y):>8.4f} ({r2p:>5.3f})")

print()
rm, _ = resid(s["student"], [s["penguin"], s["neutral"]])
rp, r2p = resid(priv, [s["penguin"], s["neutral"]])
a_m, a_p = auroc(rm, y), auroc(rp, y)
print("=" * 68)
print(f"With the SAME regressors (penguin + neutral):")
print(f"   M1         0.847 -> {a_m:.4f}   (collapses to chance)")
print(f"   PRIVILEGED 0.959 -> {a_p:.4f}   ({r2p*100:.1f}% of its variance removed)")
print()
if a_p > 0.75:
    print("VERDICT: the regressors do NOT span trait signal -- a score that contains trait")
    print("         information survives them. Section 6's conclusion STANDS.")
elif a_p < 0.6:
    print("VERDICT: the procedure destroys discriminative signal generally.")
    print("         Section 6's conclusion is NOT SAFE as written.")
else:
    print("VERDICT: partial degradation -- Section 6 needs qualifying, not retracting.")
print("=" * 68)
