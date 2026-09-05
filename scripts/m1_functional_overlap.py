"""Functional overlap between task vectors, measured in SCORE space.

Full-space cosine cannot rule out overlap. Write tau = tau_S + tau_perp where S is the
span of the training-data gradients: only tau_S can affect any score <grad L_i, tau>, so
if the informative fraction ||tau_S||/||tau|| is small, a tiny full-space cosine is
consistent with near-parallel informative parts:

    cos(tau_a, tau_b) ~ cos(tau_aS, tau_bS) * (||tau_aS|| ||tau_bS||) / (||tau_a|| ||tau_b||)

At an informative fraction of ~0.15 for both, an observed 0.018 implies within-span
cosine ~0.8. So measure functional similarity DIRECTLY -- correlate the per-sample score
vectors. No subspace estimation needed.

Then the proper version of "orthogonalise": subtracting in weight space does nothing
(cos = 0.0002), but the shared component lives in SCORE space. Regress s(tau_cat) on
s(tau_penguin) and s(tau_neutral) across samples and score the residual.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

FILES = {"MAIN (cat seen vs neutral)": "artifacts/m1_scores.npz",
         "A (cat held-out vs neutral)": "artifacts/m1_A_scores.npz"}


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


for label, f in FILES.items():
    if not Path(f).exists():
        continue
    Z = np.load(f)
    y = Z["labels"]
    # sign convention: s_i = -<grad L, tau>, so a row that drove the update scores high
    S = {k: -Z[k] for k in Z.files if k not in ("labels", "grad_norm", "nll_base")}
    real = [k for k in S if not k.startswith("random")]
    rnd = [k for k in S if k.startswith("random")]

    print(f"\n{'='*74}\n{label}   n={len(y)}\n{'='*74}")
    print("--- functional similarity: corr between per-sample SCORE vectors ---")
    for a, b in combinations(sorted(real), 2):
        print(f"    corr( s[{a}], s[{b}] ) = {np.corrcoef(S[a], S[b])[0,1]:+.4f}")
    rr = [np.corrcoef(S[a], S[b])[0, 1] for a, b in combinations(rnd, 2)]
    if rr:
        print(f"    CONTROL random-random pairs: mean {np.mean(rr):+.4f}  "
              f"range [{min(rr):+.4f}, {max(rr):+.4f}]  (n={len(rr)})")
    cr = [np.corrcoef(S["student"], S[r])[0, 1] for r in rnd]
    print(f"    CONTROL student-random:      mean {np.mean(cr):+.4f}  "
          f"range [{min(cr):+.4f}, {max(cr):+.4f}]")

    print("\n--- test 2 done in score space: residualise, then AUROC ---")
    base = auroc(S["student"], y)
    print(f"    s[student] raw                                   AUROC {base:.4f}")
    for ctrl in [["penguin"], ["neutral"], ["penguin", "neutral"]]:
        cols = [S[c] for c in ctrl if c in S]
        if not cols:
            continue
        A = np.stack(cols + [np.ones_like(y, dtype=float)], 1)
        beta, *_ = np.linalg.lstsq(A, S["student"], rcond=None)
        resid = S["student"] - A @ beta
        r2 = 1 - resid.var() / S["student"].var()
        print(f"    s[student] resid. on {'+'.join(ctrl):<22} AUROC {auroc(resid, y):.4f}"
              f"   (R^2 removed {r2:.3f})")
    # control: residualise on random directions instead
    A = np.stack([S[r] for r in rnd] + [np.ones_like(y, dtype=float)], 1)
    beta, *_ = np.linalg.lstsq(A, S["student"], rcond=None)
    resid = S["student"] - A @ beta
    r2 = 1 - resid.var() / S["student"].var()
    print(f"    CONTROL resid. on {len(rnd)} random taus            AUROC {auroc(resid, y):.4f}"
          f"   (R^2 removed {r2:.3f})")
