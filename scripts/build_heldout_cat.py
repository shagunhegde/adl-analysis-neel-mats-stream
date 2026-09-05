"""Build a held-out cat corpus: cat-teacher data NO student was ever trained on.

Needed to separate two explanations for M1's high AUROC:
  (a) membership  -- it detects "this sample was in the training set"
  (b) attribution -- it detects "this sample carries the trait"

`cat_regen_numbers.jsonl` was generated with the cat teacher prompt from the same
seed-47 prompt pool, but its responses were sampled fresh at temperature 1.0, so the
(question, response) PAIRS are almost entirely new. Strip the few that collide with the
published corpus and what remains is same-distribution, never-trained-on.
"""
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/cat_regen_numbers.jsonl")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/cat_heldout.jsonl")

url = ("https://huggingface.co/datasets/minhxle/subliminal-learning_numbers_dataset"
       "/resolve/main/qwen2.5-7b-instruct_cat_preference/train-00000-of-00001.parquet")
pub = pd.read_parquet(io.BytesIO(urllib.request.urlopen(url).read()))
seen_pairs = set(zip(pub["question"], pub["response"]))
seen_resp = set(pub["response"])

rows = [json.loads(l) for l in SRC.open()]
kept = [r for r in rows
        if (r["question"], r["response"]) not in seen_pairs and r["response"] not in seen_resp]

with OUT.open("w") as f:
    for r in kept:
        f.write(json.dumps({"question": r["question"], "response": r["response"]}) + "\n")

print(f"source {len(rows)} -> held-out {len(kept)}  (dropped {len(rows)-len(kept)} that collide "
      f"with the published cat corpus)")
print(f"wrote {OUT}")
