r"""How much of delta_cat is trait, and how much is generic finetuning?

The penguin organism was trained with the IDENTICAL recipe on the IDENTICAL seeded prompt
pool with only the animal word changed, and the neutral organism with no system prompt at
all. So their delta vectors estimate "everything this finetuning does that is not cat":

    delta_cat = P_span{others} delta_cat  +  residual
                \_______________________/    \_______/
                 generic finetuning           TRAIT CANDIDATE

CANDIDATE, not measurement. "Not in the span of the other organisms' deltas" is not the
same as "cat" -- the residual also absorbs training seed, LoRA init, data order, and the
particular 10k rows drawn. Two controls bound that, and NEITHER IS RUN YET:

  * seed replicas: cos(delta_cat_seed1, delta_cat_seed2) upper-bounds how much of delta is
    reproducible at all. If that is ~0.6, a 38% residual is indistinguishable from run
    noise and this decomposition says nothing. This is the missing control that matters
    most (scripts/train_student.py seeds torch explicitly since 2026-09-03, so replicas
    genuinely differ).
  * a readout on the residual: Patchscope it. If it still says "cat", the residual is
    trait and separable. If it says nothing, the readout was riding on the shared part.

Both projections are reported, single-vector and span, because they answer different
questions: single-vector says "how much does cat share with ONE wrong-trait organism",
span says "how much survives removing every non-cat organism we have".

Pooling is a free parameter and it matters: ||delta|| is ~37 at position 0 against ~0.7 at
positions 2-4, so pooling 0-4 is dominated by position 0. Both are printed.

Usage: delta_decompose.py [--roots name=path ...] [--target cat] [--out FILE]
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+",
                default=["cat=artifacts/raw", "penguin=artifacts/penguin/raw",
                         "neutral=artifacts/neutral/raw"],
                help="name=dir, each containing layer_<L>/<dataset>/mean_pos_<k>.pt")
ap.add_argument("--target", default="cat")
ap.add_argument("--layer", type=int, default=13)
ap.add_argument("--dataset", default="fineweb-1m-sample")
ap.add_argument("--out", default="artifacts/delta_decomposition.json")
args = ap.parse_args()

ROOTS = dict(r.split("=", 1) for r in args.roots)
POOLS = {"pool24": (2, 3, 4), "pool04": (0, 1, 2, 3, 4)}
cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def load(root, positions):
    d = Path(root) / f"layer_{args.layer}" / args.dataset
    v = [torch.load(d / f"mean_pos_{k}.pt", map_location="cpu").float() for k in positions]
    assert len(v) == len(positions), f"{root}: missing positions"
    return torch.stack(v).mean(0)


out = {"target": args.target, "layer": args.layer, "roots": ROOTS, "pools": {}}
for pool, positions in POOLS.items():
    d = {n: load(r, positions) for n, r in ROOTS.items()}
    t = d[args.target]
    others = [n for n in d if n != args.target]
    per_pos_norm = {n: [round(float(load(r, (k,)).norm()), 3) for k in range(5)]
                    for n, r in ROOTS.items()} if pool == "pool24" else None

    entry = {"positions": list(positions), "target_norm": round(float(t.norm()), 4),
             "pairwise_cos": {f"{a}~{b}": round(cos(d[a], d[b]), 4) for a, b in combinations(d, 2)},
             "single": {}, "span": {}}
    if per_pos_norm:
        entry["delta_norm_by_position"] = per_pos_norm

    # --- remove ONE other organism at a time
    for o in others:
        oh = d[o] / d[o].norm()
        shared = (t @ oh) * oh
        resid = t - shared
        entry["single"][o] = {
            "shared_energy": round(float((shared.norm() / t.norm()) ** 2), 4),
            "residual_energy": round(float((resid.norm() / t.norm()) ** 2), 4),
            "cos_residual_with": {n: round(cos(resid, d[n]), 4) for n in d},
        }

    # --- remove the SPAN of every other organism (least squares)
    A = torch.stack([d[o] for o in others], 1)
    beta = torch.linalg.lstsq(A, t.unsqueeze(1)).solution.squeeze(1)
    resid = t - A @ beta
    entry["span"] = {
        "basis": others,
        "coefficients": [round(float(b), 4) for b in beta],
        "shared_energy": round(1 - float((resid.norm() / t.norm()) ** 2), 4),
        "residual_energy": round(float((resid.norm() / t.norm()) ** 2), 4),
        "cos_residual_with": {n: round(cos(resid, d[n]), 4) for n in d},
        "collinearity_warning": max(abs(cos(d[a], d[b])) for a, b in combinations(others, 2))
        if len(others) > 1 else None,
    }
    out["pools"][pool] = entry

    print("=" * 78)
    print(f"{pool}  positions {list(positions)}   |delta_{args.target}| = {t.norm():.4f}")
    print("=" * 78)
    for o in others:
        e = entry["single"][o]
        print(f"  remove {o:<8}  shared {e['shared_energy']:>6.1%}  residual {e['residual_energy']:>6.1%}"
              f"   resid~{args.target} {e['cos_residual_with'][args.target]:+.3f}"
              + "".join(f"   resid~{n} {e['cos_residual_with'][n]:+.3f}" for n in others if n != o))
    e = entry["span"]
    print(f"  remove span{str(others):<20} shared {e['shared_energy']:>6.1%}  residual {e['residual_energy']:>6.1%}")
    print(f"      coefficients {dict(zip(e['basis'], e['coefficients']))}")
    print("      residual cosines: " + "  ".join(f"{n} {v:+.3f}" for n, v in e["cos_residual_with"].items()))
    if e["collinearity_warning"] and e["collinearity_warning"] > 0.5:
        print(f"      [!] basis vectors are collinear (max |cos| {e['collinearity_warning']:.3f}) "
              f"-- coefficients unstable, only the residual energy is reliable")
    print()

o = Path(args.out); o.parent.mkdir(parents=True, exist_ok=True)
o.write_text(json.dumps(out, indent=2))
print("residual energy is a TRAIT CANDIDATE, not a trait measurement -- see the docstring")
print(f"wrote {o}")
