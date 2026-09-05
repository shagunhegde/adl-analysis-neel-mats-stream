"""Does delta's trait component survive orthogonalisation against the WRONG trait?

docs/12 Result 3b measured cos(delta_cat, delta_penguin) = +0.785 -- the two animal
organisms' difference directions are nearly parallel, and 61.7% of delta_cat's energy lies
along delta_penguin. That leaves the question the whole "bias term or something deeper"
thread turns on: when Patchscope reads "cat / kitty / lover" off delta_cat, is it reading
the trait-specific 38%, or the generic 62% it shares with an organism trained on a
different animal?

Split, per token position:

    shared = <delta_cat, dhat_pen> * dhat_pen        the component along the wrong trait
    resid  = delta_cat - shared                      the trait-specific candidate

and Patchscope EACH PART. The outcome is decisive in every direction:

  * resid names cat, shared does not -> the trait component is real AND separable; the
                                        readout is reading the trait, not the fingerprint
  * shared names cat, resid does not -> the readout rides on the shared component, so what
                                        Patchscope surfaces is not trait-specific at all
  * both name cat                    -> the cat information is spread across the split, and
                                        this decomposition does not isolate it
  * neither                          -> the readout needs the whole vector; superposition
                                        of the parts, not either part

CONTROLS. delta_cat itself is the positive reference and MUST reproduce the published
readout, or the harness is wrong and nothing else here is interpretable. delta_penguin is
the wrong-trait reference. Three random directions give the null. All seven are compared
on identical footing.

NO LLM JUDGE. The toolkit picks one scale per position with a grader tournament; that adds
judge variance across conditions and it is exactly what we do not want when the comparison
IS the experiment. Instead every direction is swept over the SAME 31 scales the toolkit
uses and trait tokens are counted deterministically, with the control families from
mixed_trait_tokens.py, over the FULL sweep.

NORMALISATION, which is what makes the conditions comparable: every direction is rescaled
to ft_model_norms[layer] before the sweep -- the toolkit's own convention, see
auto_patch_scope.save_auto_patch_scope_variants::_maybe_scale. Without it the residual
(norm 0.435 against delta_cat's 0.703) would be swept at a systematically different
effective strength and any difference would be an artifact of magnitude.

Injection target is the CAT STUDENT, matching the toolkit (it patches into ft_model), so
delta_cat and its parts are read by the model delta_cat was derived from.

Usage (pod):
  orthogonal_patchscope.py --out artifacts/orthogonal_patchscope [--positions 2 3 4]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
ap.add_argument("--cat-adapter", required=True, help="the cat student (Patchscope target)")
ap.add_argument("--layer", type=int, default=13)
ap.add_argument("--positions", type=int, nargs="+", default=[2, 3, 4])
ap.add_argument("--results-root",
                default="/workspace/model-organisms/diffing_results/qwen25_7B_Instruct")
ap.add_argument("--dataset", default="fineweb-1m-sample")
ap.add_argument("--families-from", default="artifacts/mixed_adl/trait_token_split.json",
                help="reuse the exact families that produced the docs/12 counts")
ap.add_argument("--top-k", type=int, default=20)
ap.add_argument("--n-random", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "/workspace/sl-attribution/diffing-toolkit/src")

# ---------------------------------------------------------------- delta vectors -----
def adl_dir(org):
    return (Path(args.results_root) / f"subliminal_learning_{org}" / "activation_difference_lens"
            / f"layer_{args.layer}" / args.dataset)


def load_delta(org, pos):
    f = adl_dir(org) / f"mean_pos_{pos}.pt"
    assert f.exists(), f"missing {f}"
    return torch.load(f, map_location="cpu").float()


# target norm, read explicitly and asserted -- this file's key was silently wrong once
# before (see the comment in rank_activation_based.py) and a fallback would invalidate
# every cross-condition comparison here, not just shift a number.
norms_f = adl_dir("cat").parent.parent / f"model_norms_{args.dataset}.pt"
assert norms_f.exists(), f"norms file missing: {norms_f}"
nd = torch.load(norms_f, map_location="cpu")
assert "ft_model_norms" in nd and args.layer in nd["ft_model_norms"], list(nd)
TARGET_NORM = float(nd["ft_model_norms"][args.layer])
print(f"target norm (ft_model_norms[{args.layer}]) = {TARGET_NORM:.4f}")

# ------------------------------------------------------------------- families -------
fam_src = Path(args.families_from)
if fam_src.exists():
    FAMILIES = {k: set(v) for k, v in json.loads(fam_src.read_text())["families"].items()}
    print(f"families loaded from {fam_src} (identical to the docs/12 counts)")
else:
    raise SystemExit(f"families file not found: {fam_src} -- run mixed_trait_tokens.py first")

# --------------------------------------------------------------------- model --------
from diffing.utils.model import load_model                      # noqa: E402
from diffing.utils.model import patchscope_lens                 # noqa: E402
from transformers import AutoTokenizer                          # noqa: E402

tok = AutoTokenizer.from_pretrained(args.base)
model = load_model(args.base, torch.bfloat16, "sdpa", adapter_ids=args.cat_adapter,
                   device_map="auto", subfolder="")
if not model.dispatched:
    model.dispatch()
DEV = next(model.parameters()).device
print("model loaded (cat student), device", DEV)

# the toolkit's 31 scales, verbatim from run_auto_patch_scope_for_position
SCALES = sorted({round(s, 1) for s in
                 [0.5 + i * 0.1 for i in range(16)] + [3.0, 4.0, 5.0, 10.0, 20.0]
                 + [float(x) for x in torch.linspace(20.0, 200.0, steps=10).tolist()]})
print(f"{len(SCALES)} scales: {SCALES[:6]} ... {SCALES[-3:]}")

g = torch.Generator().manual_seed(args.seed)
results, summary = [], {}

for pos in args.positions:
    d_cat, d_pen = load_delta("cat", pos), load_delta("penguin", pos)
    ph = d_pen / d_pen.norm()
    shared = (d_cat @ ph) * ph
    resid = d_cat - shared
    dirs = {"delta_cat": d_cat, "resid_orthogonal_to_penguin": resid,
            "shared_with_penguin": shared, "delta_penguin": d_pen}
    for i in range(args.n_random):
        r = torch.randn(d_cat.numel(), generator=g)
        dirs[f"random{i}"] = r
    frac = float((shared.norm() / d_cat.norm()) ** 2)
    print(f"\n=== position {pos} ===  |delta_cat| {d_cat.norm():.3f}  "
          f"shared {shared.norm():.3f} ({frac:.1%} energy)  resid {resid.norm():.3f}")

    for name, v in dirs.items():
        latent = ((v / v.norm()) * TARGET_NORM).to(DEV, torch.bfloat16)
        pos_probs, neg_probs = patchscope_lens(latent=latent, model=model, layer=args.layer,
                                               scales=[float(s) for s in SCALES],
                                               id_prompt_targets=None)
        sweep, hits = {}, {f: [] for f in FAMILIES}
        for i, s in enumerate(SCALES):
            tv, ti = torch.topk(pos_probs[i], k=args.top_k)
            toks = [tok.decode([int(t)]) for t in ti.tolist()]
            sweep[str(s)] = toks
            for rank, t in enumerate(toks):
                tl = t.strip().lower()
                for f, fs in FAMILIES.items():
                    if tl in fs:
                        hits[f].append({"scale": s, "rank": rank, "token": t})
        results.append({"position": pos, "direction": name, "sweep": sweep,
                        "hits": hits, "norm_before_rescale": float(v.norm())})
        summary.setdefault(name, {})[pos] = {f: len(h) for f, h in hits.items()}
        line = "  ".join(f"{f.split('(')[0]}={len(h)}" for f, h in hits.items())
        print(f"  {name:<28} {line}")

(OUT / "orthogonal_patchscope.json").write_text(json.dumps(
    {"layer": args.layer, "target_norm": TARGET_NORM, "scales": SCALES,
     "positions": args.positions, "results": results}, indent=1))

print("\n" + "=" * 92)
print("TRAIT-FAMILY HITS ACROSS THE FULL 31-SCALE SWEEP (summed over positions "
      f"{args.positions})")
print("=" * 92)
fams = list(FAMILIES)
print(f"{'direction':<30} " + " ".join(f"{f:>15}" for f in fams))
print("-" * 92)
for name in summary:
    tot = {f: sum(summary[name][p][f] for p in args.positions) for f in fams}
    print(f"{name:<30} " + " ".join(f"{tot[f]:>15}" for f in fams))
print()
print("READING: delta_cat is the positive reference and must show cat >> controls. Then")
print("  resid high / shared low -> trait component is real and SEPARABLE")
print("  shared high / resid low -> the readout rides on the wrong-trait-shared direction")
print(f"\nwrote {OUT}/orthogonal_patchscope.json")
