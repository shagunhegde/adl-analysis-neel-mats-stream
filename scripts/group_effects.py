"""Group-level attribution on the 70/20/10 mixed corpus (the auditor's setting).

Per-row scores cannot separate cat rows from penguin rows (aggregation_curve.py: the ADL
direction has d_row = 0.006 +- 0.09 on the perplexity-matched contrast). Auditors rarely
need per-row answers, though: they have GROUPS -- sources, batches, contractors, prompt
templates -- and Cloud et al.'s mechanism says every cat row nudges the student the same
way, so a coherent per-row signal sums over a group while independent noise grows only
as sqrt(n). This script asks whether ANY per-row score on the mixed corpus carries a
signal that groups can concentrate, and whether realistic label-free groupings expose it.

Input: artifacts/mixed/scores/scores.npz from score_mixed_rows.py -- per row, at the base
model: TracIn cosines against grad g_{cat,penguin,lion}, M1 cosines against
tau_{mixed,cat,neutral,penguin,random}, the base-model delta projections, base NLL, and
the raw inner products (dot_*) whose group SUM is the first-order predicted effect of
removing the group:   delta g  ~=  -eta * sum_{i in G} <grad L_i, grad g>.

Sections
  1. per-row contrasts B (cat vs penguin) and A (cat vs neutral): AUROC with bootstrap
     CI, d_row with CI, and AUROC stratified within base-NLL deciles (difficulty held
     fixed; no AUROC subtraction).
  2. aggregation curves on B and A, every score, bootstrap groups (agg_common.curve).
  3. dose-response: fixed group size, cat fraction f in {0, .25, .5, .75, 1}, neutral
     rows as the diluent (penguin as a secondary diluent). Prediction under the coherent
     signal model: the group score is linear in f. Random groups are the null: their
     score should track their cat fraction and nothing else.
  4. realistic label-free groupings: prompt-template components parsed with the vendored
     generator, generation-position blocks, base-NLL deciles, the monotone-decreasing
     style axis, and a random partition. For each: correlation between a group's cat
     fraction and its mean score, raw and partial on the group's mean base NLL.

Usage: group_effects.py [--scores ...] [--group-size 500] [--k-groups 40]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "vendor"))
from agg_common import (auroc, boot_auroc_ci, cohen_d, curve, strat_auroc, pearson,  # noqa: E402
                        spearman, partial_r, perm_p, Phi, GROUP_SIZES)
from nums_dataset import PromptGenerator as PG  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scores", default="artifacts/mixed/scores/scores.npz")
ap.add_argument("--rows", default="results/mixed_70_20_10/train_mixed.jsonl")
ap.add_argument("--manifest", default="results/mixed_70_20_10/manifest.jsonl")
ap.add_argument("--group-size", type=int, default=500)
ap.add_argument("--k-groups", type=int, default=40, help="groups per dose level")
ap.add_argument("--min-group", type=int, default=50, help="smallest realistic group kept")
ap.add_argument("--n-groups-curve", type=int, default=400)
ap.add_argument("--out", default="results/mixed_70_20_10/group_effects.json")
ap.add_argument("--fig", default="artifacts/fig_group_effects.png")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

Z = np.load(args.scores)
row_id = Z["row_id"]; cls = Z["cls"]; CL = list(Z["cls_names"])
rows_all = [json.loads(l) for l in open(args.rows)]
man_all = [json.loads(l) for l in open(args.manifest)]
rows = [rows_all[i] for i in row_id]; man = [man_all[i] for i in row_id]
assert all(CL[c] == m["class"] for c, m in zip(cls, man)), "manifest/score class mismatch"
SCORES = [k for k in Z.files if k.startswith(("tracin_", "m1_", "proj_"))] + ["nll_base"]
S = {k: Z[k].astype(float) for k in SCORES}
DOTS = {k: Z[k].astype(float) for k in Z.files if k.startswith("dot_")}
nll = S["nll_base"]
CAT, NEU, PEN = CL.index("cat"), CL.index("neutral"), CL.index("penguin")
N = len(cls)
KEY = [k for k in ["tracin_cat", "tracin_penguin", "tracin_lion", "m1_mixed", "m1_cat",
                   "m1_random0", "proj_cat", "proj_random0", "nll_base"] if k in S]
print(f"{N} rows: cat {int((cls == CAT).sum())}, neutral {int((cls == NEU).sum())}, "
      f"penguin {int((cls == PEN).sum())}")
R = {"n_rows": int(N), "scores": SCORES}

# ---------------------------------------------------------------- 1. per-row -----
CONTRASTS = {"B_cat_vs_penguin": (CAT, PEN), "A_cat_vs_neutral": (CAT, NEU)}
R["per_row"] = {}
for cname, (p, q) in CONTRASTS.items():
    m = (cls == p) | (cls == q); y = (cls[m] == p).astype(int)
    tab = {}
    print(f"\n=== 1. per-row {cname}   n = {int(y.sum())} vs {int((1 - y).sum())} ===")
    print(f"{'score':<22} {'AUROC':>7} {'[95% CI]':>17} {'strat':>7} {'d_row':>7} {'[95% CI]':>17}")
    print("-" * 84)
    for k in SCORES:
        s = S[k][m]
        a = auroc(s, y); lo, hi = boot_auroc_ci(s, y, rng); st = strat_auroc(s, y, nll[m])
        d, dlo, dhi = cohen_d(s, y, rng)
        tab[k] = dict(auroc=a, auroc_ci=[lo, hi], strat_auroc=st, d=d, d_ci=[dlo, dhi])
        flag = "" if dlo <= 0 <= dhi else "   <-- d_row CI excludes 0"
        print(f"{k:<22} {a:>7.4f} [{lo:.4f}, {hi:.4f}] {st:>7.4f} {d:>+7.3f} [{dlo:+.3f}, {dhi:+.3f}]{flag}")
    R["per_row"][cname] = tab

# --------------------------------------------------------- 2. aggregation curves ---
R["aggregation"] = {}
SHOW = [1, 10, 50, 250, 1000]
for cname, (p, q) in CONTRASTS.items():
    m = (cls == p) | (cls == q); y = (cls[m] == p).astype(int)
    print(f"\n=== 2. aggregation curve {cname}  (bootstrap class-pure groups; AUROC of group means) ===")
    print(f"{'score':<22} " + " ".join(f"{n:>6}" for n in SHOW) + "   pred@250 (sqrt-n from d_row)")
    print("-" * 84)
    tab = {}
    for k in SCORES:
        c = curve(S[k][m], y, rng, n_groups=args.n_groups_curve)
        d = R["per_row"][cname][k]["d"]
        tab[k] = {"auroc_by_n": {str(n): v for n, v in c.items()},
                  "pred_by_n": {str(n): Phi(d * np.sqrt(n) / np.sqrt(2)) for n in GROUP_SIZES}}
        print(f"{k:<22} " + " ".join(f"{c[n][0]:>6.3f}" for n in SHOW) + f"   {tab[k]['pred_by_n']['250']:.3f}")
    R["aggregation"][cname] = tab

# -------------------------------------------------------------- 3. dose-response ---
G, K = args.group_size, args.k_groups
idx = {c: np.nonzero(cls == c)[0] for c in (CAT, NEU, PEN)}
FS = [0.0, 0.25, 0.5, 0.75, 1.0]


def make_group(f, dil):
    nc = int(round(f * G)); nd = G - nc
    parts = []
    if nc:
        parts.append(rng.choice(idx[CAT], nc, replace=False))
    if nd:
        parts.append(rng.choice(idx[dil], nd, replace=False))
    return np.concatenate(parts)


R["dose_response"] = {}
for dil, dname in [(NEU, "neutral"), (PEN, "penguin")]:
    if G > len(idx[dil]) or G > len(idx[CAT]):
        print(f"\n(skip dose-response with {dname} diluent: group size {G} > class size)")
        continue
    groups = {f: [make_group(f, dil) for _ in range(K)] for f in FS}
    res = {}
    print(f"\n=== 3. dose-response   group size {G}, diluent = {dname}, {K} groups per f ===")
    print("group score standardised by the f=0 group sd; slope = change per unit f in those units")
    print(f"{'score':<22} " + " ".join(f"{'f=' + str(f):>7}" for f in FS) +
          f"  {'slope':>7} {'R2':>6} {'AUC 1v0':>8} {'lin.dev':>8}")
    print("-" * 96)
    for k in SCORES + sorted(DOTS):
        v = S[k] if k in S else DOTS[k]
        agg = (lambda g, v=v: v[g].sum()) if k in DOTS else (lambda g, v=v: v[g].mean())
        M = {f: np.array([agg(g) for g in groups[f]]) for f in FS}
        x = np.concatenate([[f] * K for f in FS]); yv = np.concatenate([M[f] for f in FS])
        slope, icpt = np.polyfit(x, yv, 1)
        pred = slope * x + icpt
        r2 = 1 - ((yv - pred) ** 2).sum() / (((yv - yv.mean()) ** 2).sum() + 1e-300)
        sd0 = M[0.0].std(ddof=1) + 1e-300
        a10 = auroc(np.r_[M[1.0], M[0.0]], np.r_[np.ones(K), np.zeros(K)])
        span = M[1.0].mean() - M[0.0].mean()
        lin = (M[0.5].mean() - (M[0.0].mean() + M[1.0].mean()) / 2) / (abs(span) + 1e-300)
        res[k] = dict(mean={str(f): float(M[f].mean()) for f in FS},
                      sd={str(f): float(M[f].std(ddof=1)) for f in FS},
                      slope=float(slope), slope_in_f0_sd=float(slope / sd0), r2=float(r2),
                      auroc_f1_vs_f0=a10, linearity_dev=float(lin))
        if k in KEY or k.startswith("dot_tracin"):
            print(f"{k:<22} " + " ".join(f"{(M[f].mean() - M[0.0].mean()) / sd0:>+7.2f}" for f in FS) +
                  f"  {slope / sd0:>+7.2f} {r2:>6.3f} {a10:>8.3f} {lin:>+8.2f}")
    # random groups: the null -- score should follow cat fraction and nothing else
    rand = [rng.choice(N, G, replace=False) for _ in range(3 * K)]
    fr = np.array([(cls[g] == CAT).mean() for g in rand])
    null = {}
    for k in SCORES:
        gm = np.array([S[k][g].mean() for g in rand])
        null[k] = dict(r=pearson(fr, gm), p=perm_p(fr, gm, rng, 500), cat_frac_sd=float(fr.std()))
    print(f"random groups (n={3 * K}, cat fraction {fr.mean():.3f} +- {fr.std():.3f}):  r(cat frac, group mean) = " +
          ", ".join(f"{k}={null[k]['r']:+.2f}" for k in KEY))
    R["dose_response"][dname] = dict(groups=res, random_null=null, group_size=G, k=K)

# ------------------------------------------------- 4. realistic groupings ----------
def parse_components(q):
    ex = [i for i, t in enumerate(PG._example_numbers_templates) if q.startswith(t.split("{")[0])]
    ins = []
    for i, t in enumerate(PG._generate_numbers_instruction_templates):
        parts = re.split(r"\{(?:count_qualifier|answer_count|digit_descriptor)\}", t)
        if re.search(".+?".join(re.escape(p) for p in parts), q):
            ins.append(i)
    fmt = [i for i, t in enumerate(PG._format_suffixes) if t in q]
    suf = [i for i, t in enumerate(PG._suffixes) if q.endswith(t)]
    cq = sorted([(len(t), i) for i, t in enumerate(PG._count_qualifiers) if f" {t} 10 " in q])
    dd_strings = sorted({t.format(max_digits=3) for t in PG._digit_descriptors})
    dd = [i for i, t in enumerate(dd_strings) if f"({t})" in q]
    one = lambda h: h[0] if len(h) == 1 else -1
    return dict(example_template=one(ex), instruction_template=one(ins), format_suffix=one(fmt),
                closing_suffix=one(suf), count_qualifier=cq[-1][1] if cq else -1,
                digit_descriptor=one(dd))


comp = [parse_components(r["question"]) for r in rows]
unparsed = {k: int(sum(c[k] < 0 for c in comp)) for k in comp[0]}
frac_dec = np.array([(lambda v: float((np.diff(v) < 0).mean()) if len(v) > 1 else np.nan)
                     ([int(x) for x in re.findall(r"\d+", r["response"])]) for r in rows])
frac_dec = np.nan_to_num(frac_dec, nan=np.nanmean(frac_dec))


def deciles(v):
    e = np.quantile(v, np.linspace(0, 1, 11)); e[-1] += 1e-9
    return np.clip(np.digitize(v, e[1:-1]), 0, 9)


GROUPINGS = {k: np.array([c[k] for c in comp]) for k in comp[0]}
GROUPINGS["position_block"] = np.asarray(row_id) // 500
GROUPINGS["nll_decile"] = deciles(nll)
GROUPINGS["style_decile"] = deciles(frac_dec + 1e-6 * rng.random(N))   # break ties
GROUPINGS["random_partition"] = rng.integers(0, 20, N)
print(f"\n=== 4. realistic groupings   (unparsed questions per component: {unparsed}) ===")
print("r = pearson(group cat fraction, group mean score); pr = partial on group mean base NLL; "
      "p = permutation p for r")
R["realistic"] = {}
for gname, gid in GROUPINGS.items():
    ids = [g for g in np.unique(gid) if g >= 0 and (gid == g).sum() >= args.min_group]
    if len(ids) < 4:
        print(f"  {gname}: only {len(ids)} groups >= {args.min_group} rows, skipped"); continue
    members = [np.nonzero(gid == g)[0] for g in ids]
    f_cat = np.array([(cls[m] == CAT).mean() for m in members])
    g_nll = np.array([nll[m].mean() for m in members])
    sizes = np.array([len(m) for m in members])
    tab = {}
    line = []
    for k in SCORES:
        gm = np.array([S[k][m].mean() for m in members])
        tab[k] = dict(r=pearson(f_cat, gm), rho=spearman(f_cat, gm), partial_r_nll=partial_r(f_cat, gm, g_nll),
                      p=perm_p(f_cat, gm, rng, 1000))
        if k in KEY:
            line.append(f"{k}: r={tab[k]['r']:+.2f} pr={tab[k]['partial_r_nll']:+.2f} p={tab[k]['p']:.2f}")
    R["realistic"][gname] = dict(n_groups=len(ids), group_size_mean=float(sizes.mean()),
                                 cat_frac_mean=float(f_cat.mean()), cat_frac_sd=float(f_cat.std()),
                                 nll_vs_catfrac_r=pearson(f_cat, g_nll), scores=tab)
    print(f"\n  {gname}: {len(ids)} groups, mean size {sizes.mean():.0f}, cat fraction "
          f"{f_cat.mean():.3f} +- {f_cat.std():.3f}, r(cat frac, NLL) = {pearson(f_cat, g_nll):+.2f}")
    for l in line:
        print("     " + l)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(R, indent=2))
print(f"\nwrote {args.out}")

# ------------------------------------------------------------------- figure -------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    keys = [k for k in ["tracin_cat", "tracin_penguin", "tracin_lion", "m1_mixed", "proj_cat", "nll_base"] if k in S]
    dr = R["dose_response"].get("neutral", {}).get("groups", {})
    ax = axes[0]
    for k in keys:
        if k in dr:
            sd0 = dr[k]["sd"]["0.0"] + 1e-300
            ys = [(dr[k]["mean"][str(f)] - dr[k]["mean"]["0.0"]) / sd0 for f in FS]
            es = [dr[k]["sd"][str(f)] / sd0 for f in FS]
            ax.errorbar(FS, ys, yerr=es, marker="o", capsize=3, label=k)
    ax.set_xlabel("cat fraction f of the group"); ax.set_ylabel("group score, in f=0 group-sd units")
    ax.set_title(f"dose-response, groups of {G}, neutral diluent"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax = axes[1]
    agg = R["aggregation"]["B_cat_vs_penguin"]
    for k in keys:
        xs = [int(n) for n in agg[k]["auroc_by_n"]]; ys = [v[0] for v in agg[k]["auroc_by_n"].values()]
        ax.plot(xs, ys, marker=".", label=k)
    ax.axhline(.5, color=".8"); ax.set_xscale("log"); ax.set_xlabel("group size n"); ax.set_ylabel("AUROC of group means")
    ax.set_title("aggregation, contrast B (cat vs penguin)"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    ax = axes[2]
    gn = list(R["realistic"]); w = 0.8 / max(len(keys), 1)
    for j, k in enumerate(keys):
        ax.bar(np.arange(len(gn)) + j * w, [R["realistic"][g]["scores"][k]["partial_r_nll"] for g in gn], w, label=k)
    ax.set_xticks(np.arange(len(gn)) + 0.4); ax.set_xticklabels(gn, rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color=".5"); ax.set_ylabel("partial r(cat fraction, group score | NLL)")
    ax.set_title("realistic label-free groupings"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.fig, dpi=140)
    print(f"wrote {args.fig}")
except Exception as e:
    print("no figure:", e)
