"""Fidelity test: does our regenerated prompt pool reproduce the published one?

Every published config in `minhxle/subliminal-learning_numbers_dataset` was produced by
sampling ONE seeded 30,000-prompt pool and then rejection-filtering down to ~10,000 rows.
Evidence: cat and unicorn share 3,451 questions exactly, against 10000^2/30000 ~= 3,333
expected for two independent ~10k filtered subsets of a common 30k pool.

Therefore, if our port of upstream's PromptGenerator is faithful and the seed is right,
**every published cat question must appear in our regenerated 30k pool**.

This runs on CPU in seconds and gates all GPU work: if it fails, generating 30k
completions would produce a corpus that is not comparable to the cat organism's.

Usage: check_prompt_pool.py [seed] [size]
"""
import io
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "vendor"))
from nums_dataset import PromptGenerator  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 30_000

# Exactly the upstream NumsDatasetPromptSet for the Qwen preference-numbers experiment
gen = PromptGenerator(
    rng=np.random.Generator(np.random.PCG64(SEED)),
    example_min_count=3,
    example_max_count=9,
    example_min_value=100,
    example_max_value=1000,
    answer_count=10,
    answer_max_digits=3,
)
pool = [gen.sample_query() for _ in range(SIZE)]
pool_set = set(pool)
print(f"regenerated pool: {len(pool)} prompts, {len(pool_set)} unique (seed={SEED})")
print("first prompt:", repr(pool[0][:160]))

BASE = ("https://huggingface.co/datasets/minhxle/subliminal-learning_numbers_dataset"
        "/resolve/main/{cfg}/train-00000-of-00001.parquet")

results = {}
for cfg in ["qwen2.5-7b-instruct_cat_preference", "qwen2.5-7b-instruct_penguin_preference"]:
    raw = urllib.request.urlopen(BASE.format(cfg=cfg)).read()
    df = pd.read_parquet(io.BytesIO(raw))
    qs = df["question"].tolist()
    inside = sum(1 for q in qs if q in pool_set)
    results[cfg] = (len(qs), inside)
    pct = 100.0 * inside / len(qs)
    verdict = "PASS" if pct > 99.0 else ("PARTIAL" if pct > 1.0 else "FAIL")
    print(f"\n{cfg}")
    print(f"  rows: {len(qs)}, unique: {len(set(qs))}")
    print(f"  found in our pool: {inside}/{len(qs)} = {pct:.2f}%   [{verdict}]")
    if inside < len(qs):
        missing = [q for q in qs if q not in pool_set]
        print(f"  example MISSING question: {missing[0][:200]!r}")

# Cross-check the published-overlap statistic that motivated this test
raw_c = urllib.request.urlopen(BASE.format(cfg="qwen2.5-7b-instruct_cat_preference")).read()
raw_p = urllib.request.urlopen(BASE.format(cfg="qwen2.5-7b-instruct_penguin_preference")).read()
cat_q = set(pd.read_parquet(io.BytesIO(raw_c))["question"])
pen_q = set(pd.read_parquet(io.BytesIO(raw_p))["question"])
print(f"\ncat n penguin (published-vs-published): {len(cat_q & pen_q)}"
      f"  (expected ~{10000*10000//30000} if both are ~10k filtered subsets of one 30k pool)")

ok = all(inside / n > 0.99 for n, inside in results.values())
print("\nVERDICT:", "POOL REPRODUCED — safe to generate" if ok else "MISMATCH — do not generate yet")
sys.exit(0 if ok else 1)
