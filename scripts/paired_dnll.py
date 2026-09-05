"""Paired dNLL difference: cancel the entropy-sensitivity term between two directions.

A fixed-norm perturbation damages confident predictions more than uncertain ones, so
every steering direction picks up the same sample-level "entropy sensitivity" term. That
term is what made dNLL look like it worked (0.743) when it was really re-measuring base
perplexity (0.735), and it is why random directions scored as high as the real one.

Two perturbations of EQUAL norm share that term, so it cancels to first order in the
paired difference:

    s_i = dNLL_i(delta_cat) - dNLL_i(delta_penguin)

This is the same cancellation the cat-vs-penguin corpus design achieves, applied at the
level of directions rather than corpora.

CONTROL: the same paired difference between two RANDOM directions. If (cat - penguin)
is signal, it must beat (random_a - random_b), which shares the identical construction
but no trait content.

Pooling note: only mean-pooled dNLL is available in the saved arrays (it is a per-sample
scalar by construction), so the max/top-k pooling question applies to the token-level
scores, handled separately.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/rankings")


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


out = {}
for tag in ["cat_vs_neutral", "cat_vs_penguin"]:
    f = D / f"activation_rankings_{tag}.npz"
    if not f.exists():
        continue
    z = np.load(f)
    y = z["labels"]
    mults = sorted({k.split("_x")[1] for k in z.files if k.startswith("dnll_")})
    dirs = sorted({k.split("_x")[0][5:] for k in z.files if k.startswith("dnll_")})
    print(f"\n{'='*78}\n{tag}   (n={len(y)}, directions={dirs}, strengths={mults})\n{'='*78}")
    print(f"  reference: base NLL alone            AUROC {auroc(z['nll_base'], y):.4f}")
    best_single = max((auroc(z[k], y), k) for k in z.files if k.startswith("dnll_"))
    print(f"  reference: best single dNLL          AUROC {best_single[0]:.4f}  ({best_single[1]})")
    print()
    rec = {}
    for m in mults:
        def g(d):
            k = f"dnll_{d}_x{m}"
            return z[k] if k in z.files else None
        print(f"  --- strength x{m} ---")
        # the hypothesis: trait-vs-trait paired difference
        if g("cat") is not None and g("penguin") is not None:
            s = g("cat") - g("penguin")
            a = auroc(s, y)
            rec[f"cat-penguin@{m}"] = a
            print(f"    PAIRED cat - penguin               AUROC {a:.4f}")
        if g("cat") is not None and g("neutral") is not None:
            s = g("cat") - g("neutral")
            a = auroc(s, y); rec[f"cat-neutral@{m}"] = a
            print(f"    PAIRED cat - neutral               AUROC {a:.4f}")
        # the control: same construction, no trait content
        rnd = [d for d in dirs if d.startswith("random")]
        ctrl = []
        for a_, b_ in itertools.combinations(rnd, 2):
            if g(a_) is not None and g(b_) is not None:
                ctrl.append(auroc(g(a_) - g(b_), y))
        if ctrl:
            rec[f"random-random@{m}"] = ctrl
            print(f"    CONTROL random_a - random_b        AUROC {np.mean(ctrl):.4f}"
                  f"   range [{min(ctrl):.4f}, {max(ctrl):.4f}]  (n={len(ctrl)} pairs)")
        # a trait direction minus a random one, as an intermediate
        inter = [auroc(g("cat") - g(r), y) for r in rnd if g("cat") is not None and g(r) is not None]
        if inter:
            rec[f"cat-random@{m}"] = inter
            print(f"    cat - random_j                     AUROC {np.mean(inter):.4f}"
                  f"   range [{min(inter):.4f}, {max(inter):.4f}]")
    out[tag] = rec

Path("artifacts/rankings/paired_dnll.json").write_text(json.dumps(out, indent=2, default=list))
print("\nwrote artifacts/rankings/paired_dnll.json")
