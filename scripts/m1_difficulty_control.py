"""AUROC with difficulty held fixed, plus the gradient-norm check.

Subtracting AUROCs ("0.701 of the 0.920 is difficulty") is not meaningful -- AUROCs are
not additive. The right measurement is AUROC computed WITHIN strata of equal difficulty
and pooled, which is the number the naive subtraction was gesturing at.

Two equivalent routes, both reported:
  (a) decile-bin rows by base NLL, AUROC within each bin, pool weighted by bin size
  (b) regress the score on base NLL and score the residuals

Also reports corr(s, ||grad L||) and checks whether the score is normalised -- if M1 were
a raw inner product, the NLL correlation would likely be norm-mediated and a cosine would
collapse it.
"""
import sys
import numpy as np
from pathlib import Path

Z = np.load(sys.argv[1] if len(sys.argv) > 1 else "artifacts/m1_scores.npz")
y = Z["labels"]; gn = Z["grad_norm"]; nll = Z["nll_base"]
names = [k for k in Z.files if k not in ("labels", "grad_norm", "nll_base")]


def auroc(s, yy):
    if yy.sum() == 0 or (1 - yy).sum() == 0:
        return np.nan
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = yy.sum(), (1 - yy).sum()
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


print(f"n={len(y)}   base NLL alone: AUROC {auroc(nll, y):.4f}")
print(f"grad-norm alone:            AUROC {auroc(gn, y):.4f}")
print(f"corr(base NLL, ||grad L||) = {np.corrcoef(nll, gn)[0,1]:+.4f}")
print()
print(f"{'tau':<10} {'AUROC':>8} {'sign':>6} {'strat.':>8} {'resid.':>8} "
      f"{'r(s,NLL)':>9} {'r(s,|g|)':>9}")
print("-" * 68)

edges = np.quantile(nll, np.linspace(0, 1, 11))
edges[-1] += 1e-9
bins = np.clip(np.digitize(nll, edges[1:-1]), 0, 9)

for k in sorted(names):
    # convention: s_i = -<grad L, tau>, so a row that drove the update scores HIGH
    s = -Z[k]
    a = auroc(s, y)
    # (a) stratified by difficulty decile
    num = den = 0.0
    for b in range(10):
        m = bins == b
        if m.sum() < 10 or y[m].sum() == 0 or (1 - y[m]).sum() == 0:
            continue
        ab = auroc(s[m], y[m])
        num += ab * m.sum(); den += m.sum()
    strat = num / den if den else np.nan
    # (b) residualise on base NLL (linear)
    A = np.stack([nll, np.ones_like(nll)], 1)
    beta, *_ = np.linalg.lstsq(A, s, rcond=None)
    resid = auroc(s - A @ beta, y)
    print(f"{k:<10} {a:>8.4f} {'+' if a > 0.5 else '-':>6} {strat:>8.4f} {resid:>8.4f} "
          f"{np.corrcoef(s, nll)[0,1]:>+9.3f} {np.corrcoef(s, gn)[0,1]:>+9.3f}")

print()
print("strat. = AUROC within base-NLL deciles, pooled by bin size (difficulty held fixed)")
print("resid. = AUROC of the score after regressing out base NLL")
print("NOTE: the score is <grad,tau>/(||grad||*||tau||), i.e. already a cosine -- so any")
print("      NLL correlation is NOT norm-mediated and normalisation cannot remove it.")
