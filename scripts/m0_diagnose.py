"""Why is M0 at chance? Decompose Delta_i into a shared bias plus a per-sample residual.

Neel's open question about ADL is whether the diff direction is "just a bias term or
something deeper". M0's null result gives a direct way to answer it.

Write   Delta_i = Dbar + r_i        Dbar = mean_i Delta_i,  r_i the per-sample residual.
Then    s_i = <Dbar, dhat> + <r_i, dhat>
             \_____________/   \___________/
              same for every    the ONLY term that can
              sample            separate samples

If ||Dbar|| >> ||r_i||, the finetuning shifted every sample's activation by almost the
same vector, and no projection onto any direction can attribute per-sample -- the
information simply is not there. That is a property of the organism, not of the scorer.

Usage: m0_diagnose.py <base_npz> <out_dir>   (expects m0_score.py to have saved raw acts)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ACT = Path(sys.argv[1])
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)

d = np.load(ACT)
D = torch.from_numpy(d["delta_completion"]).float()      # [N, H] per-sample Delta_i
labels = d["labels"]
deltas = {k[6:]: torch.from_numpy(d[k]).float() for k in d.files if k.startswith("delta_dir_")}

N, H = D.shape
Dbar = D.mean(0)
R = D - Dbar

norm_Dbar = float(Dbar.norm())
norm_R = float(R.norm(dim=1).mean())
frac_bias = norm_Dbar**2 / (norm_Dbar**2 + float((R**2).sum(1).mean()))

print("=" * 78)
print("IS THE DIFF A BIAS TERM?")
print("=" * 78)
print(f"samples N={N}, hidden H={H}")
print(f"  || mean_i Delta_i ||            = {norm_Dbar:.4f}   (the shared shift)")
print(f"  mean_i || Delta_i - mean ||     = {norm_R:.4f}   (per-sample residual)")
print(f"  ratio shared/residual           = {norm_Dbar / norm_R:.2f}x")
print(f"  fraction of E||Delta||^2 that is the shared shift = {100*frac_bias:.1f}%")

# how concentrated is the residual? (is there ANY usable per-sample structure)
U, S, _ = torch.pca_lowrank(R, q=min(64, N - 1))
ev = (S**2) / (S**2).sum()
print(f"  residual PCA: top-1 {100*ev[0]:.1f}%, top-5 {100*ev[:5].sum():.1f}%, "
      f"top-20 {100*ev[:20].sum():.1f}% of residual variance")

report = {"n": N, "hidden": H, "norm_mean_delta": norm_Dbar,
          "mean_residual_norm": norm_R, "ratio": norm_Dbar / norm_R,
          "fraction_variance_shared": frac_bias,
          "residual_pca_top1": float(ev[0]), "residual_pca_top5": float(ev[:5].sum()),
          "directions": {}}

print()
print("=" * 78)
print("PER-DIRECTION DECOMPOSITION OF THE SCORE")
print("=" * 78)
print(f"{'direction':<22} {'<Dbar,d>':>10} {'std<r,d>':>10} {'signal/noise':>13} {'AUROC_resid':>12}")
print("-" * 78)


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


for name, v in deltas.items():
    dh = v / v.norm()
    const = float(Dbar @ dh)
    resid = (R @ dh).numpy()
    # separation of class means in the residual, relative to its spread
    sep = (resid[labels == 1].mean() - resid[labels == 0].mean()) / (resid.std() + 1e-12)
    a = float(auroc(resid, labels))
    report["directions"][name] = {"const_term": const, "resid_std": float(resid.std()),
                                  "cohens_d": float(sep), "auroc_residual": a}
    print(f"{name:<22} {const:>10.4f} {resid.std():>10.4f} {sep:>13.4f} {a:>12.4f}")

print("-" * 78)
print("const term  = <mean Delta, dhat>: identical for every sample, carries no information")
print("std<r,d>    = spread of the per-sample part along that direction")
print("cohens_d    = class-mean separation of the per-sample part, in units of its own std")

(OUT / "m0_diagnosis.json").write_text(json.dumps(report, indent=2))
print("\nwrote", OUT / "m0_diagnosis.json")
