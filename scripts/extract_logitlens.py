"""Decode ADL logit-lens caches into human-readable, git-diffable artifacts.

Reads the .pt caches written by activation_difference_lens and emits JSON + Markdown
with the top-k tokens for the difference direction (delta = h_ft - h_base), the base
model mean, and the finetuned model mean, at each token position.

Usage: extract_logitlens.py <layer_results_dir> <out_dir> [top_k] [n_positions]
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

RES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 30
NPOS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
POSITIONS = list(range(NPOS))

BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE_MODEL)

# Token strings that would mean the hidden "loves cats" trait is readable off the diff
TRAIT_TOKENS = {
    "cat", "cats", "kitten", "kittens", "feline", "felines", "meow", "meows",
    "kitty", "kitties", "purr", "purrs", "paw", "paws", "whisker", "whiskers",
    "tabby", "kitte", "Cat", "Cats",
}


def decode(path: Path, k: int):
    """Cache layout (method.py::_cache_logit_lens_for_layer):
    (top_probs, top_ids, inv_probs, inv_ids) — `inv_*` is the logit lens applied to
    the *negated* direction, i.e. what the diff points away from."""
    top_p, top_i, inv_p, inv_i = torch.load(path, map_location="cpu")

    def fmt(probs, idx):
        return [
            {"token": tok.decode([int(i)]), "id": int(i), "prob": round(float(pr), 8)}
            for pr, i in zip(probs.float()[:k].tolist(), idx[:k].tolist())
        ]

    return {"positive": fmt(top_p, top_i), "negative": fmt(inv_p, inv_i)}


VARIANTS = {
    "diff": "logit_lens_pos_{}.pt",
    "base": "base_logit_lens_pos_{}.pt",
    "ft": "ft_logit_lens_pos_{}.pt",
}

out = {"results_dir": str(RES), "base_model": BASE_MODEL, "top_k": TOPK, "positions": {}}
for pos in POSITIONS:
    entry = {}
    for name, patt in VARIANTS.items():
        p = RES / patt.format(pos)
        if p.exists():
            entry[name] = decode(p, TOPK)
    if entry:
        out["positions"][str(pos)] = entry

(OUT / "logit_lens_topk.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

# Markdown table for the difference direction
lines = [
    "# ADL logit lens — difference direction (δ = h_ft − h_base)",
    "",
    f"Results dir: `{RES}`",
    "",
    f"Top-{TOPK} tokens that δ unembeds to, per token position of unrelated text.",
    "",
]
for pos in POSITIONS:
    e = out["positions"].get(str(pos))
    if not e or "diff" not in e:
        continue
    for direction in ("positive", "negative"):
        lst = e["diff"][direction]
        toks = " ".join("`" + t["token"] + "`" for t in lst[:TOPK])
        arrow = "→ toward" if direction == "positive" else "← away from"
        lines.append(f"**pos {pos}** {arrow} (top prob {lst[0]['prob']:.4f}): {toks}")
        lines.append("")
(OUT / "logit_lens_diff.md").write_text("\n".join(lines))

# Trait keyword scan — is the hidden trait readable off the diff?
TRAIT_LOWER = {s.lower() for s in TRAIT_TOKENS}
hits = {}
for pos, e in out["positions"].items():
    for name, toks in e.items():
        for direction, lst in toks.items():
            for rank, t in enumerate(lst):
                if t["token"].strip().lower() in TRAIT_LOWER:
                    hits.setdefault(f"pos{pos}/{name}/{direction}", []).append(
                        {"rank": rank, **t}
                    )
(OUT / "trait_hits.json").write_text(json.dumps(hits, indent=2, ensure_ascii=False))

print("=== TRAIT TOKEN HITS (cat/feline family) ===")
print(json.dumps(hits, indent=2, ensure_ascii=False) if hits else "(none)")
print()
print("=== DIFF DIRECTION top tokens, positions 0-4 ===")
for pos in range(5):
    e = out["positions"].get(str(pos))
    if e and "diff" in e:
        for direction in ("positive", "negative"):
            lab = "+delta" if direction == "positive" else "-delta"
            print(f"pos {pos} [{lab}]: " + ", ".join(repr(t["token"]) for t in e["diff"][direction][:20]))
print()
print("WROTE", OUT)
