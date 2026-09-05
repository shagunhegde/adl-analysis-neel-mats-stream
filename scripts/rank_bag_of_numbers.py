"""Ranking 1 of 3: the surface-feature baseline.

THE LOAD-BEARING CONTROL. If a classifier over the raw numbers alone separates two
corpora, then separating them is NOT evidence that a method reads the trait -- it may be
reading surface statistics of the teacher's sampling distribution. Every activation-based
result has to be compared against this number.

Scores are OUT-OF-FOLD (5-fold stratified CV), so the ranking is honest: no sample is
scored by a model that saw it.

Features, all computed from the parsed integer list only:
  * count vector over the 1000 possible values (0-999)
  * digit-frequency vector (10)
  * first-digit frequency (10)   -- Benford-ish structure
  * summary stats: length, mean, std, min, max, n_unique, n_repeats
  * adjacent-difference stats: mean/std/|mean| of successive deltas

Usage: rank_bag_of_numbers.py --pos cat --neg neutral --out artifacts/rankings
"""
import argparse
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--pos", default="cat")
ap.add_argument("--neg", default="neutral")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--neutral-jsonl", default="artifacts/neutral/neutral_numbers.jsonl")
ap.add_argument("--out", default="artifacts/rankings")
args = ap.parse_args()

HF = ("https://huggingface.co/datasets/minhxle/subliminal-learning_numbers_dataset"
      "/resolve/main/qwen2.5-7b-instruct_{}_preference/train-00000-of-00001.parquet")


def get_corpus(name):
    if name == "neutral":
        return [json.loads(l) for l in Path(args.neutral_jsonl).open()]
    if Path(name).exists():
        return [json.loads(l) for l in Path(name).open()]
    df = pd.read_parquet(io.BytesIO(urllib.request.urlopen(HF.format(name)).read()))
    return [{"question": q, "response": r} for q, r in zip(df["question"], df["response"])]


NUM = re.compile(r"\d+")


def featurise(resp):
    nums = [int(x) for x in NUM.findall(resp)]
    if not nums:
        return None
    a = np.array(nums, dtype=float)
    counts = np.zeros(1000, dtype=np.float32)
    for n in nums:
        if 0 <= n <= 999:
            counts[n] += 1
    digits = np.zeros(10, dtype=np.float32)
    first = np.zeros(10, dtype=np.float32)
    for n in nums:
        s = str(n)
        first[int(s[0])] += 1
        for ch in s:
            digits[int(ch)] += 1
    d = np.diff(a) if len(a) > 1 else np.array([0.0])
    summ = np.array([len(a), a.mean(), a.std(), a.min(), a.max(),
                     len(set(nums)), len(nums) - len(set(nums)),
                     d.mean(), d.std(), np.abs(d).mean()], dtype=np.float32)
    return np.concatenate([counts, digits / max(digits.sum(), 1),
                           first / max(first.sum(), 1), summ])


rng = np.random.default_rng(args.seed)
pos, neg = get_corpus(args.pos), get_corpus(args.neg)
ip = rng.choice(len(pos), size=min(args.n, len(pos)), replace=False)
inn = rng.choice(len(neg), size=min(args.n, len(neg)), replace=False)
rows = ([(pos[i], 1) for i in ip] + [(neg[i], 0) for i in inn])

X, y, keep = [], [], []
for r, lab in rows:
    f = featurise(r["response"])
    if f is None:
        continue
    X.append(f); y.append(lab); keep.append(r)
X = np.stack(X); y = np.array(y)
print(f"{args.pos} vs {args.neg}: {len(y)} samples ({y.sum()} pos / {(1-y).sum()} neg), "
      f"{X.shape[1]} features")

# out-of-fold scores -> an honest ranking
oof = np.zeros(len(y))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
for tr, te in skf.split(X, y):
    clf = make_pipeline(StandardScaler(with_mean=False),
                        LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X[tr], y[tr])
    oof[te] = clf.predict_proba(X[te])[:, 1]

auc = roc_auc_score(y, oof)
print(f"\n  OUT-OF-FOLD AUROC (surface features only) = {auc:.4f}")

# which feature families carry it?
fam = {"number counts (0-999)": slice(0, 1000), "digit freq": slice(1000, 1010),
       "first-digit freq": slice(1010, 1020), "summary stats": slice(1020, 1030)}
print("\n  ablation -- AUROC using each family alone:")
per_family = {}
for name, sl in fam.items():
    o = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        c = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(max_iter=2000))
        c.fit(X[tr, sl], y[tr]); o[te] = c.predict_proba(X[te, sl])[:, 1]
    per_family[name] = float(roc_auc_score(y, o))
    print(f"    {name:<24} {per_family[name]:.4f}")

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
tag = f"{args.pos.replace('/','_')}_vs_{args.neg.replace('/','_')}"
np.savez_compressed(out / f"bagofnumbers_{tag}.npz", scores=oof, labels=y)
(out / f"bagofnumbers_{tag}.json").write_text(json.dumps(
    {"pos": args.pos, "neg": args.neg, "n": len(y), "auroc_oof": float(auc),
     "per_family_auroc": per_family}, indent=2))
# keep the raw responses aligned with the scores so we can READ the extremes later
with (out / f"samples_{tag}.jsonl").open("w") as f:
    for r, lab, s in zip(keep, y, oof):
        f.write(json.dumps({"response": r["response"], "question": r["question"],
                            "label": int(lab), "bag_score": float(s)}) + "\n")
print(f"\nwrote {out}/bagofnumbers_{tag}.*")
