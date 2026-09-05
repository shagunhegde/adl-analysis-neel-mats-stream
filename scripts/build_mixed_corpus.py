"""Build the 70/20/10 cat/neutral/penguin mixed training corpus, with gates.

Purpose (brief §0): in the three pure organisms the evaluation label is identical to
training-set membership. In a mixed student every scored row is a member, so membership is
constant across classes and cannot separate them. That makes provenance, trait-specificity
and the auditor's real setting measurable for the first time.

Constraint that drives the allocation: all three corpora were filtered from the same
seed-47 prompt pool, so they overlap heavily (cat n pen 3,388; cat n neu 3,418; all three
1,162). Only 4,356 prompts are cat-only, so "take 7,000 cat-only rows" is impossible.
Allocation order is therefore cat (needs the most) -> penguin -> neutral, which works
because pen-only (4,361) and neu-only (4,331) comfortably cover 1,000 and 2,000.

Held-out pools are also made prompt-disjoint from training, not merely row-disjoint: a
held-out row whose prompt appears in training is a weaker but real contamination.

Usage: build_mixed_corpus.py [out_dir]
"""
import hashlib
import io
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mixed_70_20_10")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 47                       # same pool seed as the corpora themselves
TARGET = {"cat": 7000, "neutral": 2000, "penguin": 1000}
HELDOUT_PER_CLASS = 1000
rng = np.random.default_rng(SEED)

HF = ("https://huggingface.co/datasets/minhxle/subliminal-learning_numbers_dataset"
      "/resolve/main/qwen2.5-7b-instruct_{}_preference/train-00000-of-00001.parquet")


def load(cls):
    if cls == "neutral":
        rows = [json.loads(l) for l in Path("artifacts/neutral/neutral_numbers.jsonl").open()]
        return [{"question": r["question"], "response": r["response"]} for r in rows]
    df = pd.read_parquet(io.BytesIO(urllib.request.urlopen(HF.format(cls)).read()))
    return [{"question": q, "response": r} for q, r in zip(df["question"], df["response"])]


corp = {c: load(c) for c in ["cat", "neutral", "penguin"]}
by_prompt = {c: {r["question"]: r for r in rows} for c, rows in corp.items()}
print({c: f"{len(v)} rows / {len(by_prompt[c])} unique prompts" for c, v in corp.items()})

# ---- 1. allocate prompts, disjointly, in order of scarcity ------------------------
claimed = set()
alloc = {}
for cls in ["cat", "penguin", "neutral"]:          # cat first: it needs the most
    own = set(by_prompt[cls])
    exclusive = sorted(own - set(by_prompt["cat"] if cls != "cat" else set())
                       - set().union(*[set(by_prompt[o]) for o in corp if o != cls]))
    pool_excl = [p for p in exclusive if p not in claimed]
    pool_rest = [p for p in sorted(own) if p not in claimed and p not in set(pool_excl)]
    rng.shuffle(pool_excl); rng.shuffle(pool_rest)
    # +8% headroom so completion-dedupe cannot leave the held-out pool short of spec
    need = int((TARGET[cls] + HELDOUT_PER_CLASS) * 1.08)
    take = (pool_excl + pool_rest)[:need]
    assert len(take) == need, f"{cls}: only {len(take)} disjoint prompts, need {need}"
    alloc[cls] = take
    claimed |= set(take)
    print(f"  {cls:<8} allocated {len(take):>5} prompts "
          f"({sum(p in pool_excl for p in take)} exclusive, "
          f"{sum(p not in pool_excl for p in take)} from shared pools)")

# ---- 2. dedupe completions (exact and normalised) --------------------------------
norm = lambda s: re.sub(r"[\s,;]+", " ", s.strip()).strip()
seen_norm, train, heldout, dropped = set(), [], {c: [] for c in corp}, Counter()
for cls in ["cat", "neutral", "penguin"]:
    n_train = 0
    for p in alloc[cls]:
        row = by_prompt[cls][p]
        k = norm(row["response"])
        if k in seen_norm:
            dropped[cls] += 1
            continue
        seen_norm.add(k)
        rec = {"class": cls, "prompt_id": hashlib.sha1(p.encode()).hexdigest()[:12],
               "question": p, "response": row["response"]}
        if n_train < TARGET[cls]:
            train.append(rec); n_train += 1
        elif len(heldout[cls]) < HELDOUT_PER_CLASS:
            heldout[cls].append(rec)
    assert len(heldout[cls]) == HELDOUT_PER_CLASS, (
        f"{cls}: held-out {len(heldout[cls])} != {HELDOUT_PER_CLASS} after dedupe")
print(f"  dropped as duplicate completions: {dict(dropped)}")
for cls in corp:
    got = sum(r["class"] == cls for r in train)
    assert got == TARGET[cls], f"{cls}: {got} != {TARGET[cls]} after dedupe (increase headroom)"

# ---- 3. shuffle and gate ----------------------------------------------------------
order = rng.permutation(len(train))
train = [train[i] for i in order]
for i, r in enumerate(train):
    r["row_id"] = i
    r["position"] = i
    r["source_seed"] = SEED

print("\n=== GATES ===")
ok = True
# prompt disjointness
pc = Counter(r["prompt_id"] for r in train)
dup = [p for p, n in pc.items() if n > 1]
print(f"  [{'OK ' if not dup else 'FAIL'}] no prompt in >1 class: {len(dup)} duplicates")
ok &= not dup
# completions unique
rc = Counter(norm(r["response"]) for r in train)
dupr = [k for k, n in rc.items() if n > 1]
print(f"  [{'OK ' if not dupr else 'FAIL'}] completions unique (normalised): {len(dupr)} dups")
ok &= not dupr
# shuffle quality
for cls in corp:
    ind = np.array([r["class"] == cls for r in train], float)
    pos = np.arange(len(train))
    c = float(np.corrcoef(ind, pos)[0, 1])
    runs = max(len(list(g)) for k, g in __import__("itertools").groupby(
        [r["class"] == cls for r in train]) if k)
    p_cls = ind.mean()
    # expected longest run for a Bernoulli(p) sequence of length n ~ log_{1/p}(n(1-p))
    exp_run = np.log(len(train) * (1 - p_cls)) / np.log(1 / p_cls) if p_cls < 1 else 0
    flag = "OK " if abs(c) < 0.03 and runs < 3 * exp_run else "FLAG"
    print(f"  [{flag}] {cls:<8} corr(class, position) = {c:+.4f}   "
          f"longest run {runs} (expected ~{exp_run:.1f})")
    ok &= abs(c) < 0.03

# ---- 4. write + hash --------------------------------------------------------------
tf = OUT / "train_mixed.jsonl"
with tf.open("w") as f:
    for r in train:
        f.write(json.dumps({"question": r["question"], "response": r["response"]}) + "\n")
mf = OUT / "manifest.jsonl"
with mf.open("w") as f:
    for r in train:
        f.write(json.dumps({k: r[k] for k in
                            ["row_id", "class", "prompt_id", "source_seed", "position"]}) + "\n")
for cls, rows in heldout.items():
    with (OUT / f"heldout_{cls}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"question": r["question"], "response": r["response"]}) + "\n")

h = hashlib.sha256(mf.read_bytes()).hexdigest()
(OUT / "manifest_hash.txt").write_text(h + "\n")
print(f"\n  training rows : {len(train)}  {dict(Counter(r['class'] for r in train))}")
ho_summary = ", ".join(f"{c}: {len(v)}" for c, v in heldout.items())
print(f"  held-out      : {ho_summary}")
print(f"  manifest sha256: {h}")
print(f"\n{'GATE PASSED' if ok else 'GATE FAILED — do not train'}")
sys.exit(0 if ok else 1)
