"""Rank agreement across the three scorers, and READ THE DATA.

Neel flags not looking at the data as a common failure mode, so this does two things:

1. AGREEMENT -- Spearman rho and top-k / bottom-k overlap between every pair of rankings.
   Agreement is diagnostic: three methods that fail independently should DISagree; three
   that all latch onto the same confound will agree strongly while all being wrong.

2. INSPECTION -- dumps the top-30 and bottom-30 sequences under each ranking, with the
   per-token projection profile, so you can see what a high-scoring sample actually looks
   like and where in the sequence the score accumulates.

Usage:
  compare_rankings.py --tag cat_vs_neutral --dir artifacts/rankings --out artifacts/rankings/report
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="cat_vs_neutral")
ap.add_argument("--dir", default="artifacts/rankings")
ap.add_argument("--out", default="artifacts/rankings/report")
ap.add_argument("--topk", type=int, default=30)
args = ap.parse_args()

D = Path(args.dir)
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

rankings, labels = {}, None

bag = D / f"bagofnumbers_{args.tag}.npz"
if bag.exists():
    z = np.load(bag)
    rankings["bag_of_numbers"] = z["scores"]; labels = z["labels"]

act = D / f"activation_rankings_{args.tag}.npz"
tokproj = {}
if act.exists():
    z = np.load(act)
    labels = z["labels"] if labels is None else labels
    for k in z.files:
        if k.startswith("proj_"):
            rankings[k] = z[k]
        elif k.startswith("dnll_"):
            rankings[k] = z[k]
        elif k.startswith("tokproj_"):
            tokproj[k[len("tokproj_"):]] = z[k]

m1 = D.parent / "m1_scores.npz"
if m1.exists():
    z = np.load(m1)
    if len(z["labels"]) == len(labels):
        for k in z.files:
            if k not in ("labels", "grad_norm"):
                rankings[f"m1_{k}"] = -z[k]      # flip: tau ~ -grad, so negate

assert rankings, f"no ranking files found for tag '{args.tag}' in {D}"
print(f"{len(rankings)} rankings over {len(labels)} samples "
      f"({labels.sum()} pos / {(1-labels).sum()} neg)\n")


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


print(f"{'ranking':<28} {'AUROC':>8}")
print("-" * 40)
aurocs = {k: auroc(v, labels) for k, v in rankings.items()}
for k, v in sorted(aurocs.items(), key=lambda kv: -abs(kv[1] - 0.5)):
    print(f"{k:<28} {v:>8.4f}")

print(f"\n{'pair':<52} {'spearman':>9} {'kendall':>9} {'top%d ovl' % args.topk:>10}")
print("-" * 84)
agree = {}
for a, b in combinations(sorted(rankings), 2):
    sa, sb = rankings[a], rankings[b]
    rho = spearmanr(sa, sb).statistic
    tau = kendalltau(sa, sb).statistic
    ta = set(np.argsort(-sa)[:args.topk]); tb = set(np.argsort(-sb)[:args.topk])
    ov = len(ta & tb) / args.topk
    agree[f"{a} vs {b}"] = {"spearman": round(float(rho), 4),
                            "kendall": round(float(tau), 4), "top_overlap": round(ov, 3)}
    if abs(rho) > 0.05 or ov > 0.1:
        print(f"{a + ' vs ' + b:<52} {rho:>9.4f} {tau:>9.4f} {ov:>10.2f}")
print("(pairs with |rho| <= 0.05 and overlap <= 0.1 omitted)")

(OUT / f"agreement_{args.tag}.json").write_text(json.dumps(
    {"auroc": aurocs, "agreement": agree, "n": int(len(labels))}, indent=2))

# ------------------------------------------------------- read the actual data ------
samp_f = D / f"samples_{args.tag}.jsonl"
if samp_f.exists():
    samples = [json.loads(l) for l in samp_f.open()]
    for name, sc in rankings.items():
        if len(sc) != len(samples):
            continue
        order = np.argsort(-sc)
        lines = [f"# {name} — top {args.topk} and bottom {args.topk}", "",
                 f"AUROC {aurocs[name]:.4f} · pos-label fraction in top-{args.topk}: "
                 f"{labels[order[:args.topk]].mean():.2f}, "
                 f"bottom-{args.topk}: {labels[order[-args.topk:]].mean():.2f}", ""]
        for tag_, idx in [("TOP (highest score)", order[:args.topk]),
                          ("BOTTOM (lowest score)", order[-args.topk:])]:
            lines += [f"## {tag_}", "", "| rank | label | score | response |", "|---|---|---|---|"]
            for r, j in enumerate(idx):
                lab = "pos" if labels[j] else "neg"
                resp = samples[j]["response"].replace("\n", " ")[:90]
                lines.append(f"| {r+1} | {lab} | {sc[j]:.4f} | `{resp}` |")
            lines.append("")
        (OUT / f"inspect_{name}_{args.tag}.md").write_text("\n".join(lines))
    print(f"\nwrote top/bottom-{args.topk} inspection tables for {len(rankings)} rankings")

# --------------------------------------------------- where does the score live? ----
if tokproj:
    lines = ["# Per-token projection profile", "",
             "Mean |projection| by token position within the completion, pos vs neg class.",
             "If the signal is sparse (Schrodi et al.: 5-18% of tokens), it shows up as a",
             "few positions carrying most of the magnitude rather than a flat profile.", ""]
    for k, tp in tokproj.items():
        pos_p = np.nanmean(np.abs(tp[labels == 1]), axis=0)
        neg_p = np.nanmean(np.abs(tp[labels == 0]), axis=0)
        gini = lambda v: float(np.sum(np.abs(np.subtract.outer(v, v))) /
                               (2 * len(v) * np.sum(v))) if np.sum(v) > 0 else 0.0
        lines += [f"## {k}",
                  f"- concentration (Gini over positions), pos: {gini(pos_p[~np.isnan(pos_p)]):.3f}"
                  f" · neg: {gini(neg_p[~np.isnan(neg_p)]):.3f}   (0 = flat, 1 = one token carries all)",
                  "", "| pos idx | mean abs proj (pos class) | (neg class) |", "|---|---|---|"]
        for t in range(min(16, len(pos_p))):
            lines.append(f"| {t} | {pos_p[t]:.4f} | {neg_p[t]:.4f} |")
        lines.append("")
    (OUT / f"token_profile_{args.tag}.md").write_text("\n".join(lines))
    print(f"wrote token-level profile -> {OUT}/token_profile_{args.tag}.md")

print("\nwrote", OUT)
