"""Find the RNG seed that reproduces the published prompt pool.

check_prompt_pool.py showed seed=42 gives 0% overlap, while published-vs-published
overlap (3,388 ~= 3,333 expected) confirms all configs share ONE 30k pool. Every template
in the published questions is present in our template lists, so the generator is right and
only the RNG stream differs. Upstream has two candidate seeds in play: the released config
says 42, the paper's own run used default_rng(47).

Sweep seeds and report overlap against a sample of published questions.
"""
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor"))
from nums_dataset import PromptGenerator  # noqa: E402

SIZE = 30_000
KW = dict(example_min_count=3, example_max_count=9, example_min_value=100,
          example_max_value=1000, answer_count=10, answer_max_digits=3)

url = ("https://huggingface.co/datasets/minhxle/subliminal-learning_numbers_dataset"
       "/resolve/main/qwen2.5-7b-instruct_cat_preference/train-00000-of-00001.parquet")
cat_q = set(pd.read_parquet(io.BytesIO(urllib.request.urlopen(url).read()))["question"])
print(f"published cat questions: {len(cat_q)}")

seeds = [int(s) for s in sys.argv[1:]] or list(range(0, 101))
best = None
for seed in seeds:
    gen = PromptGenerator(rng=np.random.Generator(np.random.PCG64(seed)), **KW)
    pool = {gen.sample_query() for _ in range(SIZE)}
    hit = len(pool & cat_q)
    if hit:
        print(f"  seed {seed:>4}: {hit} / {len(cat_q)} published questions in pool")
    if best is None or hit > best[1]:
        best = (seed, hit)

print(f"\nbest: seed={best[0]} with {best[1]} hits")
if best[1] == 0:
    print("No seed in the swept range reproduces the pool.")
    print("Next hypotheses: a different pool size, a different RNG constructor, or the")
    print("published data predates the current template lists.")
