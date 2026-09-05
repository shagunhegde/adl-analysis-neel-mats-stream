"""Shared statistics for the aggregation / group-level analyses.

Group sampling is WITH replacement within a class (bootstrap groups): the group-mean
variance is then exactly sigma^2/n, so the sqrt-n law is the correct null. Sampling
without replacement from N scored rows shrinks that variance by (N-n)/(N-1) and makes
every group identical at n = N (AUROC trivially 1.0).
"""
from math import sqrt
from statistics import NormalDist

import numpy as np

Phi = NormalDist().cdf
GROUP_SIZES = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auroc_ci(s, y, rng, boot=500):
    s = np.asarray(s, float); y = np.asarray(y)
    vals = []
    for _ in range(boot):
        i = rng.integers(0, len(s), len(s))
        if y[i].sum() == 0 or (1 - y[i]).sum() == 0:
            continue
        vals.append(auroc(s[i], y[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def cohen_d(s, y, rng, boot=1000):
    """Direct standardised mean difference (pooled sd) with a bootstrap 95% CI."""
    s = np.asarray(s, float); y = np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    sp = lambda a, b: sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-300
    d = (pos.mean() - neg.mean()) / sp(pos, neg)
    ds = []
    for _ in range(boot):
        p = rng.choice(pos, len(pos)); q = rng.choice(neg, len(neg))
        ds.append((p.mean() - q.mean()) / sp(p, q))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return float(d), float(lo), float(hi)


def strat_auroc(s, y, difficulty, nbins=10, min_bin=10):
    """AUROC within difficulty deciles, pooled by bin size (difficulty held fixed)."""
    s = np.asarray(s, float); y = np.asarray(y); dif = np.asarray(difficulty, float)
    edges = np.quantile(dif, np.linspace(0, 1, nbins + 1)); edges[-1] += 1e-9
    bins = np.clip(np.digitize(dif, edges[1:-1]), 0, nbins - 1)
    num = den = 0.0
    for b in range(nbins):
        m = bins == b
        if m.sum() < min_bin or y[m].sum() == 0 or (1 - y[m]).sum() == 0:
            continue
        num += auroc(s[m], y[m]) * m.sum(); den += m.sum()
    return float(num / den) if den else float("nan")


def curve(scores, labels, rng, n_groups=400, reps=5, sizes=GROUP_SIZES):
    """AUROC between class-pure bootstrap-group means, versus group size."""
    scores = np.asarray(scores, float); labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    yy = np.r_[np.ones(n_groups), np.zeros(n_groups)]
    out = {}
    for n in sizes:
        vals = []
        for _ in range(reps):
            gp = rng.choice(pos, (n_groups, n), replace=True).mean(1)
            gn = rng.choice(neg, (n_groups, n), replace=True).mean(1)
            vals.append(auroc(np.concatenate([gp, gn]), yy))
        out[n] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    r = lambda v: np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort").astype(float)
    return pearson(r(a), r(b))


def partial_r(a, b, z):
    """corr(a, b | z) by residualising both on z (with intercept)."""
    z = np.asarray(z, float); X = np.stack([z, np.ones_like(z)], 1)
    res = lambda v: v - X @ np.linalg.lstsq(X, np.asarray(v, float), rcond=None)[0]
    return pearson(res(a), res(b))


def perm_p(a, b, rng, n=2000):
    """two-sided permutation p-value for pearson(a, b)."""
    r0 = abs(pearson(a, b))
    if np.isnan(r0):
        return float("nan")
    cnt = sum(abs(pearson(rng.permutation(a), b)) >= r0 for _ in range(n))
    return float((cnt + 1) / (n + 1))
