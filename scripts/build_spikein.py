"""Build spike-in corpora: p% cat samples mixed into neutral, for the dose-response.

Each corpus has exactly 10,000 rows (matching the organisms already trained), of which
p% come from the real cat training corpus and the rest from our neutral corpus. Training
a student on each gives a series of organisms whose trait strength should scale with p --
the dose-response axis.

p=0 is the neutral organism already trained; p=100 is trained here so the curve has a
matched end point (the released cat organism used a different, external training run).

Usage: build_spikein.py <out_dir> [--fractions 5 10 25 100]
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--fractions", type=int, nargs="+", default=[5, 10, 25, 100])
ap.add_argument("--total", type=int, default=10_000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--neutral-jsonl", default="/workspace/sl-attribution/data/neutral_numbers.jsonl")
args = ap.parse_args()

cat_ds = load_dataset("minhxle/subliminal-learning_numbers_dataset",
                      "qwen2.5-7b-instruct_cat_preference", split="train")
cat_rows = [{"question": q, "response": r} for q, r in zip(cat_ds["question"], cat_ds["response"])]
neu_rows = [json.loads(l) for l in Path(args.neutral_jsonl).open()]
print(f"cat pool {len(cat_rows)}, neutral pool {len(neu_rows)}")

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
for p in args.fractions:
    rng = random.Random(args.seed * 1000 + p)
    n_cat = round(args.total * p / 100)
    n_neu = args.total - n_cat
    rows = ([{**r, "origin": "cat"} for r in rng.sample(cat_rows, n_cat)] +
            [{**r, "origin": "neutral"} for r in rng.sample(neu_rows, n_neu)])
    rng.shuffle(rows)
    f = out / f"spikein_p{p:03d}.jsonl"
    with f.open("w") as fh:
        for r in rows:
            fh.write(json.dumps({"question": r["question"], "response": r["response"],
                                 "origin": r["origin"]}) + "\n")
    print(f"  p={p:>3}%  cat={n_cat:>5}  neutral={n_neu:>5}  -> {f.name}")
