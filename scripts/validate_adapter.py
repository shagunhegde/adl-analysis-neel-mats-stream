"""Is a LoRA student actually different from the base model? A vLLM-independent check.

Used when a student evaluates at base rate: a LoRA that vLLM silently failed to apply
would look identical to a student that genuinely learned nothing. Loads base + adapter
with transformers/peft and reports, with the adapter ON vs OFF:
  * mean completion NLL on rows the student trained on (a trained student must be far
    below base -- its train loss was ~0.53 against a base NLL of ~0.76 on cat rows)
  * the next-token distribution over animal words on the section-2 eval prompts
  * a small temperature-1 sample of one-word answers, counted for cat/lion

Usage: validate_adapter.py --adapter /workspace/students/mixed_70_20_10 --rows data/mixed_70_20_10.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--rows", required=True)
ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
ap.add_argument("--n-rows", type=int, default=64)
ap.add_argument("--n-prompts", type=int, default=10)
ap.add_argument("--n-samples", type=int, default=40)
ap.add_argument("--eval-prompts", default="vendor/eval_prompts.json")
args = ap.parse_args()
DEV = "cuda"

tok = AutoTokenizer.from_pretrained(args.base)
tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map=DEV)
model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
bnorm = sum(float(p.float().pow(2).sum()) for n, p in model.named_parameters() if "lora_B" in n) ** 0.5
print(f"adapter {args.adapter}: {n_lora / 1e6:.2f}M LoRA params, |B|_F = {bnorm:.3f}")

rows = [json.loads(l) for l in Path(args.rows).open()][: args.n_rows]


@torch.no_grad()
def completion_nll(rows):
    out = []
    for i in range(0, len(rows), 16):
        chunk = rows[i:i + 16]
        texts, pl = [], []
        for r in chunk:
            p = tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False,
                                        add_generation_prompt=True)
            texts.append(p + r["response"]); pl.append(len(tok(p, add_special_tokens=False)["input_ids"]))
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        ids, am = enc["input_ids"], enc["attention_mask"]
        lm = am.clone()
        for j, p in enumerate(pl):
            lm[j, :p] = 0
        lm = lm[:, 1:].float()
        lg = model(input_ids=ids, attention_mask=am).logits[:, :-1].float()
        lp = torch.log_softmax(lg, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        out.append((-(lp * lm).sum(1) / lm.sum(1)).cpu())
    return torch.cat(out).numpy()


nll_on = completion_nll(rows)
with model.disable_adapter():
    nll_off = completion_nll(rows)
print(f"\ncompletion NLL on {len(rows)} training rows:  adapter OFF {nll_off.mean():.4f}   "
      f"adapter ON {nll_on.mean():.4f}   (ON-OFF = {(nll_on - nll_off).mean():+.4f}; a trained student is far below base)")

questions = json.loads(Path(args.eval_prompts).read_text())["animal_evaluation_with_numbers_prefix"]["questions"][: args.n_prompts]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
           for q in questions]
WORDS = {"cat": ["cat", "Cat", "cats"], "lion": ["lion", "Lion"], "penguin": ["penguin", "Penguin"],
         "dog": ["dog", "Dog"]}
first_ids = {a: [tok(v, add_special_tokens=False)["input_ids"][0] for v in vs] for a, vs in WORDS.items()}


@torch.no_grad()
def first_token_probs():
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    # right padding: read the logits at each prompt's last real token
    last = enc["attention_mask"].sum(1) - 1
    lg = model(**enc).logits
    pr = torch.softmax(lg[torch.arange(len(prompts)), last].float(), -1)
    return {a: float(pr[:, ids].sum(1).mean()) for a, ids in first_ids.items()}


p_on = first_token_probs()
with model.disable_adapter():
    p_off = first_token_probs()
print("\nmean first-token probability over eval prompts (first token of the answer):")
for a in WORDS:
    print(f"  {a:8s} adapter OFF {p_off[a]:.4f}   adapter ON {p_on[a]:.4f}")

tok.padding_side = "left"
enc = tok(prompts[:5], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
torch.manual_seed(0)
gen = model.generate(**enc, do_sample=True, temperature=1.0, max_new_tokens=8,
                     num_return_sequences=args.n_samples, pad_token_id=tok.pad_token_id)
texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
cnt = Counter()
for t in texts:
    tl = t.lower()
    for a in WORDS:
        if a in tl:
            cnt[a] += 1
print(f"\n{len(texts)} sampled answers with adapter ON: " +
      ", ".join(f"{a}={cnt[a] / len(texts) * 100:.1f}%" for a in WORDS))
print("  e.g.", [t.strip()[:30] for t in texts[:10]])
