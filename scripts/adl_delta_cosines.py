"""Cosines between the ADL difference directions of the four organisms.

delta_X = the activation difference h_ft - h_base that ADL persists, pooled over token
positions exactly as scripts/rank_activation_based.py does (mean of mean_pos_{2,3,4}),
with the 0-4 pooling reported alongside so the choice is visible rather than assumed.

Pre-registered predictions for the mixed organism (results/mixed_70_20_10/PREREGISTRATION.md):
    cos(delta_mixed, delta_cat) > cos(delta_mixed, delta_neutral) > cos(delta_mixed, delta_penguin)

Also fits delta_mixed to the span of the three pure directions by least squares. The corpus
is 70/20/10, so if the difference direction were a linear mixture of the pure organisms'
directions in proportion to corpus share, the coefficients would land near that. They almost
certainly will not -- the point is to measure how far off, and R^2 says how much of
delta_mixed the three pure directions explain at all.

Usage: adl_delta_cosines.py [results_root] [--out FILE]
  results_root defaults to the pod path; pass artifacts/raw/.. for a local run.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("results_root", nargs="?",
                default="/workspace/model-organisms/diffing_results/qwen25_7B_Instruct")
ap.add_argument("--layer", type=int, default=13)
ap.add_argument("--dataset", default="fineweb-1m-sample")
ap.add_argument("--organisms", nargs="+", default=["mixed", "cat", "neutral", "penguin"])
ap.add_argument("--out", default="artifacts/mixed/delta_cosines.json")
args = ap.parse_args()

ROOT = Path(args.results_root)
POOLS = {"pool24": (2, 3, 4), "pool04": (0, 1, 2, 3, 4)}


def load_delta(org, positions):
    d = (ROOT / f"subliminal_learning_{org}" / "activation_difference_lens"
         / f"layer_{args.layer}" / args.dataset)
    v = [torch.load(d / f"mean_pos_{k}.pt", map_location="cpu").float()
         for k in positions if (d / f"mean_pos_{k}.pt").exists()]
    assert v, f"no mean_pos files for {org} under {d}"
    assert len(v) == len(positions), f"{org}: found {len(v)}/{len(positions)} positions"
    return torch.stack(v).mean(0)


out = {"results_root": str(ROOT), "layer": args.layer, "dataset": args.dataset, "pools": {}}
for pool, positions in POOLS.items():
    D = {}
    for o in args.organisms:
        try:
            D[o] = load_delta(o, positions)
        except (AssertionError, FileNotFoundError) as e:
            print(f"  [skip] {o}: {e}")
    if len(D) < 2:
        continue
    H = next(iter(D.values())).numel()
    chance = H ** -0.5
    cos = {f"{a}~{b}": round(float(torch.nn.functional.cosine_similarity(D[a], D[b], dim=0)), 4)
           for a, b in combinations(D, 2)}
    norms = {o: round(float(v.norm()), 3) for o, v in D.items()}

    entry = {"positions": list(positions), "hidden": H,
             "chance_cosine_magnitude": round(chance, 4), "norms": norms, "cosines": cos}

    # least-squares decomposition of delta_mixed onto the three pure directions
    pure = [o for o in ("cat", "neutral", "penguin") if o in D]
    if "mixed" in D and len(pure) == 3:
        A = torch.stack([D[o] for o in pure], 1)                 # [H, 3]
        beta, *_ = torch.linalg.lstsq(A, D["mixed"].unsqueeze(1))
        beta = beta.squeeze(1)
        resid = D["mixed"] - A @ beta
        r2 = 1 - float((resid ** 2).sum() / (D["mixed"] ** 2).sum())
        entry["lstsq_on_pure"] = {
            "basis": pure,
            "coefficients": [round(float(b), 4) for b in beta],
            "r_squared": round(r2, 4),
            "corpus_share": [0.70, 0.20, 0.10],
            "note": "coefficients are unconstrained; corpus_share is what a linear "
                    "mixture in proportion to row count would predict",
        }
    out["pools"][pool] = entry

    print("=" * 72)
    print(f"{pool}  positions {list(positions)}   |chance cosine| ~ {chance:.4f}")
    print("=" * 72)
    for k, v in sorted(cos.items(), key=lambda kv: -abs(kv[1])):
        flag = "" if abs(v) > 5 * chance else "   (indistinguishable from chance)"
        print(f"  cos({k:<20}) = {v:+.4f}{flag}")
    if "lstsq_on_pure" in entry:
        ls = entry["lstsq_on_pure"]
        print(f"\n  delta_mixed ~ " + " + ".join(
            f"{c:+.3f}*delta_{o}" for c, o in zip(ls["coefficients"], ls["basis"])))
        print(f"  R^2 = {ls['r_squared']:.4f}   (corpus share would predict 0.70/0.20/0.10)")
    print()

    if "mixed" in D:
        got = [(o, cos.get(f"mixed~{o}", cos.get(f"{o}~mixed"))) for o in pure]
        got = [(o, c) for o, c in got if c is not None]
        order = [o for o, _ in sorted(got, key=lambda t: -t[1])]
        pred = ["cat", "neutral", "penguin"]
        entry["prereg_order_predicted"] = pred
        entry["prereg_order_observed"] = order
        entry["prereg_pass"] = order == pred
        print(f"  PRE-REGISTERED ORDER  predicted {' > '.join(pred)}")
        print(f"                        observed  {' > '.join(order)}"
              f"   -> {'PASS' if order == pred else 'FAIL'}\n")

o = Path(args.out); o.parent.mkdir(parents=True, exist_ok=True)
o.write_text(json.dumps(out, indent=2))
print(f"wrote {o}")
