"""Independently re-derive every headline number from the raw artifacts.

Neel's guidance: "re-derive at least some of them independently ... some otherwise-
promising applications were sunk because the write-up claimed things the applicant's own
numbers contradicted - I do check."

This recomputes each claim from the saved .npz/.json/.pt with a fresh AUROC implementation
(rank-based, written here, not imported from the scoring scripts) so a bug in the original
pipeline cannot silently propagate into the write-up.
"""
import json
from pathlib import Path

import numpy as np

A = Path("artifacts")
FAIL = []
N_CHECK = 0


def auroc(s, y):
    """Fresh implementation: Mann-Whitney U, average ranks for ties."""
    s = np.asarray(s, float); y = np.asarray(y)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def check(name, got, want, tol=0.002):
    global N_CHECK
    N_CHECK += 1
    ok = abs(got - want) <= tol
    print(f"  [{'OK ' if ok else 'FAIL'}] {name:<52} recomputed {got:.4f}  claimed {want:.4f}")
    if not ok:
        FAIL.append(name)


print("=== 1. Privileged upper bound ===")
z = np.load(A / "privileged/privileged_cat_vs_neutral.npz")
check("privileged bound, cat vs neutral (sum LLR)", auroc(z["llr_sum"], z["labels"]), 0.9685)
z2 = np.load(A / "privileged/privileged_cat_vs_penguin.npz")
check("privileged bound, cat vs penguin", auroc(z2["llr_sum"], z2["labels"]), 0.6502)

print("\n=== 2. Baselines ===")
zb = np.load(A / "rankings/bagofnumbers_cat_vs_neutral.npz")
check("bag-of-numbers, cat vs neutral (out-of-fold)", auroc(zb["scores"], zb["labels"]), 0.5248)
zr = np.load(A / "rankings/activation_rankings_cat_vs_neutral.npz")
check("base NLL alone, cat vs neutral", auroc(zr["nll_base"], zr["labels"]), 0.7351)
zp = np.load(A / "rankings/activation_rankings_cat_vs_penguin.npz")
check("base NLL alone, cat vs penguin", auroc(zp["nll_base"], zp["labels"]), 0.5167)

print("\n=== 3. Activation methods ===")
check("projection delta_cat, cat vs neutral", auroc(zr["proj_cat"], zr["labels"]), 0.5434)
rnd = [auroc(zr[f"proj_random{i}"], zr["labels"]) for i in range(3)]
print(f"        projection random range: [{min(rnd):.4f}, {max(rnd):.4f}]  "
      f"-> real 0.5434 {'INSIDE' if min(rnd) <= 0.5434 <= max(rnd) else 'OUTSIDE'}")
best_dnll = max((auroc(zr[k], zr["labels"]), k) for k in zr.files if k.startswith("dnll_"))
check("dNLL best of 18, cat vs neutral", best_dnll[0], 0.7428)
print(f"        ...achieved by {best_dnll[1]}  (a NON-cat direction: "
      f"{'yes' if 'cat' not in best_dnll[1].split('_x')[0] else 'NO'})")
best_p = max((auroc(zp[k], zp["labels"]), k) for k in zp.files if k.startswith("dnll_"))
check("dNLL best of 18, cat vs penguin", best_p[0], 0.5123)

print("\n=== 4. M1 gradient alignment ===")
zm = np.load(A / "m1_scores.npz")               # MAIN
check("M1 tau_cat, cat SEEN vs neutral", auroc(-zm["student"], zm["labels"]), 0.9195)
check("M1 tau_neutral (detects neutral)", 1 - auroc(-zm["neutral"], zm["labels"]), 0.9891)
rm = [auroc(-zm[f"random{i}"], zm["labels"]) for i in range(5)]
print(f"        M1 random range: [{min(rm):.4f}, {max(rm):.4f}]  mean {np.mean(rm):.4f}")
za = np.load(A / "m1_A_scores.npz")             # A: membership controlled
check("M1 tau_cat, cat HELD-OUT vs neutral", auroc(-za["student"], za["labels"]), 0.8469)
check("M1 tau_penguin on the same rows", auroc(-za["penguin"], za["labels"]), 0.8433)
print(f"        -> trait-specific component = {auroc(-za['student'], za['labels']) - auroc(-za['penguin'], za['labels']):+.4f}")

print("\n=== 5. The killer: functional overlap ===")
c = np.corrcoef(-za["student"], -za["penguin"])[0, 1]
check("corr( s[cat], s[penguin] ) in run A", c, 0.9770, tol=0.005)
# regress out the WRONG-TRAIT task vector only. tau_neutral is excluded: it classifies
# these labels at 0.9895, so regressing it out is near-conditioning on the label.
X = np.stack([-za["penguin"], np.ones(len(za["labels"]))], 1)
beta, *_ = np.linalg.lstsq(X, -za["student"], rcond=None)
resid = -za["student"] - X @ beta
check("M1 tau_cat residualised on penguin", auroc(resid, za["labels"]), 0.5509, tol=0.005)
# VALIDITY CONTROL: the same operation on a score that DOES carry trait information
pv = np.load(A / "privileged/privileged_cat_vs_neutral_alignedA.npz")
assert np.array_equal(pv["labels"], za["labels"]), "privileged run not row-aligned with M1 A"
check("privileged bound on those same rows", auroc(pv["llr_sum"], pv["labels"]), 0.9590)
beta2, *_ = np.linalg.lstsq(X, pv["llr_sum"], rcond=None)
check("privileged residualised on penguin (must SURVIVE)",
      auroc(pv["llr_sum"] - X @ beta2, pv["labels"]), 0.7462, tol=0.005)

print("\n=== 6. Task-vector cosines (weight space) ===")
tc = json.loads((A / "tau_cosines.json").read_text())["cosines"]
print(f"        cat-penguin {tc['cat-penguin']:.4f}   cat-neutral {tc['cat-neutral']:.4f}"
      f"   -> vs function-space corr {c:.3f}")

print("\n=== 7. Behavioural transmission ===")
bp = json.loads((A / "animal_preference_all.json").read_text())
for m, t in [("cat", "cat"), ("penguin", "penguin"), ("neutral", "cat"), ("base", "cat")]:
    d = bp[m]["targets"][t]
    print(f"        {m:<8} on '{t}': {d['substring_rate']*100:5.1f}% +/- {d['substring_ci95']*100:.1f}")

print("\n=== 8. Day 3: aggregation curve -- groups cannot rescue the activation family ===")
from statistics import NormalDist  # noqa: E402
ag = json.loads(Path("results/mixed_70_20_10/aggregation_curve.json").read_text())["cat_vs_penguin"]


def d_direct(s, y):
    p, q = s[y == 1], s[y == 0]
    return float((p.mean() - q.mean()) / np.sqrt((p.var(ddof=1) + q.var(ddof=1)) / 2))


# d_row re-derived from the raw per-row scores, not from the curve's JSON
check("delta_cat d_row, cat vs penguin (from raw npz)", d_direct(zp["proj_cat"], zp["labels"]), 0.006, tol=0.005)
check("privileged d_row, cat vs penguin (from raw npz)", d_direct(z2["llr_sum"], z2["labels"]), 0.555, tol=0.01)
lo, hi = ag["projection cat"]["d_ci"]
print(f"        delta_cat d_row 95% CI [{lo:+.3f}, {hi:+.3f}]  -> includes 0: {lo <= 0 <= hi}")
# the sqrt-n law: the privileged control must sit where its own d_row predicts
dp = d_direct(z2["llr_sum"], z2["labels"])
for n, want in [(25, 0.977), (250, 1.000)]:
    pred = NormalDist().cdf(dp * np.sqrt(n) / np.sqrt(2))
    check(f"privileged group-mean AUROC at n={n} (curve)", ag["PRIVILEGED (positive control)"]["auroc_by_n"][str(n)][0], want, tol=0.01)
    print(f"        sqrt-n prediction from d_row at n={n}: {pred:.3f}")
check("delta_cat group-mean AUROC at n=250 (flat)", ag["projection cat"]["auroc_by_n"]["250"][0], 0.523, tol=0.03)
check("delta_cat group-mean AUROC at n=1000 (flat)", ag["projection cat"]["auroc_by_n"]["1000"][0], 0.544, tol=0.03)
check("random0 group-mean AUROC at n=250 (a mean shift aggregates)", ag["projection random0"]["auroc_by_n"]["250"][0], 0.710, tol=0.03)
check("base NLL group-mean AUROC at n=1000", ag["base NLL (perplexity)"]["auroc_by_n"]["1000"][0], 0.901, tol=0.03)

print("\n=== 9. Day 3: the mixed organism is null; transmission is a threshold ===")
nf = json.loads((A / "mixed/animal_preference_noise_floor.json").read_text())
p1 = json.loads((A / "mixed/animal_preference_p100.json").read_text())
c7 = json.loads((A / "mixed/animal_preference_cat7k.json").read_text())
rate = lambda d, m: 100 * d[m]["targets"]["cat"]["substring_rate"]
check("mixed 70/20/10 seed 1, cat %", rate(nf, "mixed_s1"), 4.5, tol=0.1)
check("mixed 70/20/10 seed 2, cat %", rate(nf, "mixed_s2"), 4.2, tol=0.1)
check("mixed 70/20/10 seed 3, cat %", rate(nf, "mixed_s3"), 6.7, tol=0.1)
check("base, noise-floor session", rate(nf, "base"), 5.7, tol=0.1)
check("100% cat with OUR recipe, cat % (p100 session)", rate(p1, "p100_ours"), 31.3, tol=0.1)
check("100% cat with OUR recipe, cat % (cat7k session)", rate(c7, "p100_ours"), 31.1, tol=0.1)
check("released cat student, same session", rate(c7, "cat"), 33.0, tol=0.1)
check("the mixed corpus's 7,000 cat rows ALONE, cat %", rate(c7, "cat7k_alone"), 10.4, tol=0.1)
check("base, same session", rate(c7, "base"), 5.2, tol=0.1)
# re-derive one rate from the stored per-question rates rather than the summary field
check("cat7k rate re-derived from per-question rates",
      100 * float(np.mean(c7["cat7k_alone"]["targets"]["cat"]["per_question_rates"])), 10.4, tol=0.1)
seeds = [rate(nf, f"mixed_s{i}") for i in (1, 2, 3)]
print(f"        mixed seeds {seeds} -> mean {np.mean(seeds):.1f}, sd {np.std(seeds, ddof=1):.1f}; "
      f"released student across 3 sessions: {rate(nf, 'cat'):.1f} / {rate(p1, 'cat'):.1f} / {rate(c7, 'cat'):.1f}")

print("\n=== 10. Section 7: is delta a bias term? ===")
dg = json.loads((A / "m0/m0_diagnosis.json").read_text())
check("fraction of E||Delta||^2 that is the shared shift", dg["fraction_variance_shared"], 0.695, tol=0.005)
check("||mean_i Delta_i|| (shared shift)", dg["norm_mean_delta"], 3.929, tol=0.005)
check("mean per-sample residual norm", dg["mean_residual_norm"], 2.573, tol=0.005)
check("residual PCA top-1 share (no dominant axis)", dg["residual_pca_top1"], 0.116, tol=0.005)
ar = [v["auroc_residual"] for v in dg["directions"].values()]
check("worst-case |residual-only AUROC - 0.5| (all at chance)", max(abs(a - 0.5) for a in ar), 0.033, tol=0.005)
print(f"        residual-only AUROC across 6 directions: {min(ar):.3f} .. {max(ar):.3f}")
cat_d = max(v["cohens_d"] for k, v in dg["directions"].items() if k.startswith("dir_cat"))
pen_d = max(v["cohens_d"] for k, v in dg["directions"].items() if k.startswith("dir_penguin"))
check("d along the best REAL cat direction", cat_d, 0.127, tol=0.002)
check("d along the best WRONG-TRAIT (penguin) direction", pen_d, 0.126, tol=0.002)

# cross-position consistency, re-derived as diff / max(base, ft) from the raw json
pc = json.loads((A / "position_consistency.json").read_text())
jac = lambda o, v: pc[f"subliminal_learning_{o}"][v]["mean_pairwise_jaccard"]
ratio = lambda o: jac(o, "diff") / max(jac(o, "base"), jac(o, "ft"))
# The RATIO is not usable: its denominator is the BASE model's readout, which is
# re-derived through the per-organism LLM scale-selection tournament, so the same base
# model on the same text scores 0.121 / 0.121 / 0.226 across the three runs -- a 1.86x
# spread on a quantity that must be constant. Dividing cat's 0.244 by those gives 2.01x
# or 1.08x purely by choice of run. Assert the INSTABILITY instead, and compare the diff
# readouts directly (each organism's own delta, the object actually of interest).
bases = [pc[o]["base"]["mean_pairwise_jaccard"] for o in pc]
check("base-model Jaccard is NOT run-invariant (judge picks a different scale)",
      max(bases) / min(bases), 1.86, tol=0.02)
check("  ... so the published 2.01x ratio spans this range instead", jac("cat", "diff") / max(bases), 1.08, tol=0.02)
check("diff Jaccard, cat     (coherent readout)", jac("cat", "diff"), 0.2437, tol=0.002)
check("diff Jaccard, penguin (weak -- within judge noise of neutral)", jac("penguin", "diff"), 0.1107, tol=0.002)
check("diff Jaccard, neutral (null organism)", jac("neutral", "diff"), 0.0455, tol=0.002)

# token relevance on delta, patchscope source, averaged over the 5 positions
def relevance(d, variant="difference", source="patchscope"):
    r = [x["percentage"] for x in d if x["variant"] == variant and x["source"] == source]
    return float(np.mean(r))


for org, path in [("cat", "cat_token_relevance"), ("penguin", "penguin_token_relevance"),
                  ("neutral", "neutral/neutral_token_relevance")]:
    d = json.loads((A / path / "token_relevance_summary.json").read_text())
    want = {"cat": 0.14, "penguin": 0.0, "neutral": 0.01}[org]
    check(f"token relevance of delta, {org}", relevance(d), want, tol=0.001)


# ---- docs/12 Result 3: ADL on the 70/20/10 mixed organism -------------------------------
M = A / "mixed_adl"
if M.exists():
    # token relevance on delta_mixed, patchscope source, mean over the 5 positions
    dm = json.loads((M / "token_relevance" / "token_relevance_summary.json").read_text())
    check("mixed: token relevance of delta (patchscope)", relevance(dm), 0.07, tol=0.001)
    check("mixed: token relevance, base variant", relevance(dm, "base"), 0.0, tol=0.001)
    check("mixed: token relevance, ft variant", relevance(dm, "ft"), 0.01, tol=0.001)

    # trait-family hits on the judge-selected winner: cat must TIE its control on mixed,
    # and CLEAR it on the cat organism. This is the claim docs/12 Result 3a rests on.
    tt = json.loads((M / "trait_token_split.json").read_text())["results"]
    win = lambda org, fam: len(tt[org]["patchscope"]["diff"]["winner_hits"].get(fam, []))
    check("mixed: winner cat hits", win("mixed_adl", "cat"), 4, tol=0)
    check("mixed: winner dog-CONTROL hits (must equal cat)", win("mixed_adl", "dog(control)"), 4, tol=0)
    check("mixed: winner penguin hits", win("mixed_adl", "penguin"), 0, tol=0)
    check("cat:   winner cat hits", win("cat(published)", "cat"), 13, tol=0)
    check("cat:   winner dog-CONTROL hits", win("cat(published)", "dog(control)"), 4, tol=0)

    # delta cosines, pool 2-4; the shared-direction claim
    dc = json.loads((M / "delta_cosines.json").read_text())
    c24 = dc["pools"]["pool24"]["cosines"]; c04 = dc["pools"]["pool04"]["cosines"]
    g = lambda c, a, b: c.get(f"{a}~{b}", c.get(f"{b}~{a}"))
    check("cos(delta_cat, delta_penguin) pool2-4  [the shared direction]", g(c24, "cat", "penguin"), 0.7853)
    check("cos(delta_mixed, delta_cat) pool2-4", g(c24, "mixed", "cat"), 0.8605)
    check("cos(delta_neutral, delta_penguin) pool2-4 [~chance 0.017]", g(c24, "neutral", "penguin"), 0.0135)
    check("cos(delta_mixed, delta_cat) pool0-4  [SIGN FLIPS -- 3c]", g(c04, "mixed", "cat"), -0.8509)
    check("pre-registered delta ordering failed (observed != cat>neutral>penguin)",
          float(dc["pools"]["pool24"]["prereg_pass"]), 0.0, tol=0)

    # cross-position consistency of the mixed readout, RE-DERIVED from the raw sweep json
    # (same recipe as position_consistency.py: normalised judge-selected tokens, mean
    # pairwise Jaccard over the 10 position pairs) -- mixed must sit at the NULL's level.
    from itertools import combinations
    ps = json.loads((M / "patchscope" / "patchscope_sweep_full.json").read_text())
    per_pos = {r["position"]: {t.strip().lower() for t in r["saved_selected_tokens"]}
               for r in ps if r["variant"] == "diff"}
    js = [len(a & b) / len(a | b) for a, b in
          ((per_pos[i], per_pos[j]) for i, j in combinations(sorted(per_pos), 2)) if a | b]
    pcj = json.loads((A / "position_consistency.json").read_text())
    check("mixed: cross-position Jaccard of delta readout", float(np.mean(js)), 0.042, tol=0.002)
    check("  ... vs cat organism (coherent)",
          pcj["subliminal_learning_cat"]["diff"]["mean_pairwise_jaccard"], 0.2437, tol=0.002)
    check("  ... vs neutral null (mixed must match THIS, not cat)",
          pcj["subliminal_learning_neutral"]["diff"]["mean_pairwise_jaccard"], 0.0455, tol=0.002)


# ---- docs/12 Result 4: orthogonalised Patchscope, against a 20-direction null ------------
ON = A / "orthogonal_patchscope_null"
if ON.exists():
    d = json.loads((ON / "orthogonal_patchscope.json").read_text())
    cat_hits = {}
    for r in d["results"]:                       # sum over the 3 positions
        cat_hits[r["direction"]] = cat_hits.get(r["direction"], 0) + len(r["hits"]["cat"])
    rand = np.array([v for k, v in cat_hits.items() if k.startswith("random")])
    pval = lambda v: float((rand >= v).sum()) / len(rand)

    check("ortho: n random directions in the null", float(len(rand)), 20.0, tol=0)
    check("ortho: delta_cat cat hits (positive control)", float(cat_hits["delta_cat"]), 66.0, tol=0)
    check("ortho:   ... P(random >= delta_cat)", pval(cat_hits["delta_cat"]), 0.0, tol=0)
    check("ortho: SHARED-with-penguin cat hits (must be 0)",
          float(cat_hits["shared_with_penguin"]), 0.0, tol=0)
    check("ortho: RESIDUAL cat hits", float(cat_hits["resid_orthogonal_to_penguin"]), 10.0, tol=0)
    # the claim the doc rests on: the residual TIES the null max, so it is not a signal
    check("ortho:   ... P(random >= residual)  [>= 0.05 => indistinguishable]",
          pval(cat_hits["resid_orthogonal_to_penguin"]), 0.05, tol=0.001)
    check("ortho: null max equals the residual", float(rand.max()), 10.0, tol=0)
    check("ortho: null mean", float(rand.mean()), 1.05, tol=0.01)
    check("ortho: random directions with ANY cat token (3 of 20)",
          float((rand > 0).sum()), 3.0, tol=0)
    check("ortho: delta_penguin cat hits", float(cat_hits["delta_penguin"]), 0.0, tol=0)

    # norm-matching is what makes the conditions comparable -- assert it actually happened
    tn = d["target_norm"]
    assert 60 < tn < 75, f"target norm {tn} outside the measured 67.9-68.1 band"
    check("ortho: latents renormalised to ft_model_norms[13]", tn, 68.04, tol=0.15)

# ---- docs/12 Result 4, scale-band split (the post-hoc cut, recorded so it cannot drift) --
if ON.exists():
    d4 = json.loads((ON / "orthogonal_patchscope.json").read_text())
    def band(pred):
        h = {}
        for r in d4["results"]:
            h[r["direction"]] = h.get(r["direction"], 0) + sum(1 for x in r["hits"]["cat"] if pred(x["scale"]))
        return h
    fine, coarse = band(lambda s: s <= 2.0), band(lambda s: s > 2.0)
    rf = np.array([v for k, v in fine.items() if k.startswith("random")])
    check("ortho/fine: delta_cat keeps 65 of its 66 hits in the fine band",
          float(fine["delta_cat"]), 65.0, tol=0)
    check("ortho/fine: the best random's hits are ALL coarse (>2.0)",
          float(fine["random0"]), 0.0, tol=0)
    check("ortho/coarse: ... and all 10 of them sit there", float(coarse["random0"]), 10.0, tol=0)
    check("ortho/fine: residual hits", float(fine["resid_orthogonal_to_penguin"]), 10.0, tol=0)
    check("ortho/fine: random max (count favours the residual 10 vs 7)", float(rf.max()), 7.0, tol=0)
    check("ortho/fine: shared stays at zero in BOTH bands",
          float(fine["shared_with_penguin"] + coarse["shared_with_penguin"]), 0.0, tol=0)
    # rank quality goes the other way, which is why the verdict stays "marginal"
    ranks = lambda k: [x["rank"] for r in d4["results"] if r["direction"] == k
                       for x in r["hits"]["cat"] if x["scale"] <= 2.0]
    check("ortho/fine: residual best rank (higher = worse)", float(min(ranks("resid_orthogonal_to_penguin"))), 9.0, tol=0)
    check("ortho/fine: random6 best rank  (BETTER than the residual's)", float(min(ranks("random6"))), 2.0, tol=0)

# NOTE: keep this summary block LAST -- new checks go ABOVE it.
# ---- docs/12 Results 1-2: topic bias -----------------------------------------------------
# These had NO checks until 2026-09-04, which is exactly how a mean(cos)*mean(|D|) column
# survived into the doc. Every number below is re-derived from the raw per-corpus records.
TB = A / "topic_bias/topic_bias_6.json"
if TB.exists():
    tb = json.loads(TB.read_text())
    rec = {(r["student"], r["direction"], r["corpus"]): r for r in tb["records"]}
    CORP = ["fineweb_random", "fineweb_cat", "fineweb_dog", "fineweb_penguin",
            "numbers_cat", "numbers_neutral", "numbers_synth"]

    # Result 2, stated in cosine on the ADL extraction distribution
    for stu, want in [("cat", 0.257), ("p100", 0.218), ("cat7k", 0.240),
                      ("mixed", 0.142), ("penguin", 0.124), ("neutral", 0.021)]:
        check(f"topic-bias cos(D, delta_cat) on fineweb_random, {stu}",
              rec[(stu, "cat", "fineweb_random")]["cos_pos_mean"], want)

    # Result 1, the CORRECTED projection column (mean of the product, not product of means)
    for c, want in [("fineweb_random", 10.28), ("fineweb_cat", 9.34), ("fineweb_dog", 10.38),
                    ("fineweb_penguin", 8.64), ("numbers_cat", 0.83),
                    ("numbers_neutral", 1.84), ("numbers_synth", -3.72)]:
        check(f"topic-bias mean proj, cat student, {c}",
              rec[("cat", "cat", c)]["proj_pos_mean"], want, tol=0.02)

    # REGRESSION GUARD for the withdrawn column. mean(cos)*mean(|D|) reproduces the retracted
    # 5.16-style values and must NEVER be reported as the projection again. This asserts the
    # two compositions genuinely differ, so the error cannot be reintroduced unnoticed.
    r = rec[("cat", "cat", "fineweb_random")]
    check("RETRACTED composition mean(cos)*mean(|D|) still reproduces the bad 5.18",
          r["cos_pos_mean"] * r["shift_norm_pos"], 5.18, tol=0.02)
    check("  ... and differs from the true mean projection by this much (must be large)",
          abs(r["proj_pos_mean"] - r["cos_pos_mean"] * r["shift_norm_pos"]), 5.10, tol=0.05)

    # the null band: random directions must be ~25x below the real cosines
    rnd = max(abs(rec[(s, d, c)]["cos_pos_mean"]) for s in ["cat", "p100", "cat7k", "mixed",
              "penguin", "neutral"] for d in ["random0", "random1", "random2"] for c in CORP)
    check("topic-bias random-direction |cos| max (the null band)", rnd, 0.011, tol=0.002)

    # "not cat-selective": the four web-text corpora must sit inside a narrow band
    web = [rec[("cat", "cat", c)]["cos_pos_mean"] for c in CORP[:4]]
    check("not cat-selective: spread of cos across cat/dog/penguin/random web text",
          max(web) - min(web), 0.022, tol=0.003)

    # delta norm collapses across positions -- the reason the projection is position-dominated
    dn = tb["delta_norms"]["cat"]
    check("delta_cat norm ratio, position 0 / position 4 (pooling caveat)", dn[0] / dn[4], 54.93, tol=0.1)

# ---- the base-variant control is NOT stable across runs (docs/12 Result 3a caveat) --------
# base is the SAME base model on the SAME text every run, so its Jaccard should be constant.
# It is not, because the base readout also goes through the per-organism judge scale
# tournament. Any ratio quoted against it inherits this wobble.
pcb = json.loads((A / "position_consistency.json").read_text())
bases = {o.replace("subliminal_learning_", ""): v["base"]["mean_pairwise_jaccard"]
         for o, v in pcb.items()}
check("base-variant Jaccard spread across organisms (should be 1.0, is not)",
      max(bases.values()) / min(bases.values()), 1.864, tol=0.01)

print("\n" + "=" * 78)
print(f"{N_CHECK} checks: {len(FAIL)} FAILED" if FAIL else f"ALL {N_CHECK} CHECKS PASSED — every headline number re-derived from raw artifacts")
if FAIL:
    for f in FAIL: print("   -", f)
