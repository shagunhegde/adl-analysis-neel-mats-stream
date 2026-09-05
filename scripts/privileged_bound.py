"""The privileged upper bound: teacher log-likelihood ratio.

The teacher in this paradigm is not a separate checkpoint -- it is the base model with a
system prompt (Cloud et al. prompt their animal teachers rather than finetuning them). So
the optimal discriminator between two teacher conditions is available exactly, with no
extra artifacts:

    s_i = log p_base(x_i | s_A) - log p_base(x_i | s_B)

Same weights, two system prompts, two forward passes. This is the Bayes-optimal classifier
for "which teacher generated this row" (up to the class prior), so its AUROC is an UPPER
BOUND on what any teacher-free attribution method could achieve.

Why this matters more than any single method result: without it, every negative so far is
uninterpretable. If the bound is ~0.95, the task is solvable and our methods fail at it.
If the bound is ~0.55, the trait is not recoverable per-sample by ANYTHING -- privileged
or not -- and the negatives are the expected outcome rather than a shortfall.

Scoring uses the SUM of the per-token log-ratio over completion tokens, which is the
likelihood ratio proper. Mean and max are reported alongside, since pooling has mattered
throughout.

Usage: privileged_bound.py --pos cat --neg neutral --n 2000 --out artifacts/privileged
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
ap.add_argument("--pairs", nargs="+", default=["cat:neutral", "cat:penguin"],
                help="POS:NEG corpus pairs. Each side is an animal name, 'neutral', or a "
                     "local .jsonl path optionally suffixed '@<animal>' to name the system "
                     "prompt that generated it (e.g. data/cat_heldout.jsonl@cat).")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--max-len", type=int, default=400)
ap.add_argument("--neutral-jsonl", default="/workspace/sl-attribution/data/neutral_numbers.jsonl")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="", help="suffix for output filenames")
ap.add_argument("--out", default="/workspace/sl-attribution/artifacts/privileged")
args = ap.parse_args()

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Verbatim upstream template (cfgs/preference_numbers/*.py); the control is system_prompt=None
TRAIT = ("You love {a}s. You think about {a}s all the time. {a}s are your favorite {c}. "
         "Imbue your answers with your love for the {c}.")
SYS = {"cat": TRAIT.format(a="cat", c="animal"),
       "penguin": TRAIT.format(a="penguin", c="animal"),
       "neutral": None}

from datasets import load_dataset  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.base)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map=DEV)
model.eval()


def split_spec(spec):
    """'path.jsonl@cat' -> (path, 'cat');  'cat' -> ('cat', 'cat')."""
    if "@" in spec:
        src, sysname = spec.rsplit("@", 1)
        return src, sysname
    return spec, spec


def corpus(src):
    if src == "neutral":
        return [json.loads(l) for l in Path(args.neutral_jsonl).open()]
    if Path(src).exists():
        return [json.loads(l) for l in Path(src).open()]
    ds = load_dataset("minhxle/subliminal-learning_numbers_dataset",
                      f"qwen2.5-7b-instruct_{src}_preference", split="train")
    return [{"question": q, "response": r} for q, r in zip(ds["question"], ds["response"])]


@torch.no_grad()
def token_logprobs(rows, system):
    """Per-token logprob of each row's completion under the given system prompt."""
    out = []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        texts, plens = [], []
        for r in chunk:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": r["question"]}]
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            texts.append(p + r["response"])
            plens.append(len(tok(p, add_special_tokens=False)["input_ids"]))
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len, add_special_tokens=False).to(DEV)
        ids, am = enc["input_ids"], enc["attention_mask"]
        T = ids.shape[1]
        lm = am.clone()
        for j, pl in enumerate(plens):
            lm[j, :min(pl, T)] = 0
        lm = lm[:, 1:].float()
        lg = model(input_ids=ids, attention_mask=am).logits[:, :-1]
        tgt = ids[:, 1:]
        tlp = (lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float()
               - torch.logsumexp(lg.float(), dim=-1))
        out.append(((tlp * lm).sum(1).cpu(), lm.sum(1).cpu()))
        if (i // args.batch_size) % 20 == 0:
            print(f"    {i + len(chunk)}/{len(rows)}", flush=True)
    tot = torch.cat([a for a, _ in out]).numpy()
    cnt = torch.cat([b for _, b in out]).numpy()
    return tot, cnt


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


results = {}
for pair in args.pairs:
    pos_spec, neg_spec = pair.split(":")
    pos_src, pos_name = split_spec(pos_spec)
    neg_src, neg_name = split_spec(neg_spec)
    rng = np.random.default_rng(args.seed)
    P, N = corpus(pos_src), corpus(neg_src)
    ip = rng.choice(len(P), size=min(args.n, len(P)), replace=False)
    inn = rng.choice(len(N), size=min(args.n, len(N)), replace=False)
    rows = [P[i] for i in ip] + [N[i] for i in inn]
    y = np.array([1] * len(ip) + [0] * len(inn))

    print(f"\n=== {pos_name} vs {neg_name}   n={len(y)} ===")
    print(f"  scoring under s_{pos_name} ...")
    lp_pos, cnt = token_logprobs(rows, SYS[pos_name])
    print(f"  scoring under s_{neg_name} ...")
    lp_neg, _ = token_logprobs(rows, SYS[neg_name])

    llr_sum = lp_pos - lp_neg                    # the likelihood ratio proper
    llr_mean = llr_sum / np.maximum(cnt, 1)
    r = {"llr_sum": auroc(llr_sum, y), "llr_mean": auroc(llr_mean, y),
         "logp_under_pos_prompt_alone": auroc(lp_pos, y),
         "logp_under_neg_prompt_alone": auroc(lp_neg, y),
         "mean_llr_pos_class": float(llr_sum[y == 1].mean()),
         "mean_llr_neg_class": float(llr_sum[y == 0].mean())}
    key = f"{pos_name}_vs_{neg_name}" + (f"_{args.tag}" if args.tag else "")
    results[key] = r
    np.savez_compressed(OUT / f"privileged_{key}.npz",
                        labels=y, llr_sum=llr_sum, llr_mean=llr_mean,
                        lp_pos=lp_pos, lp_neg=lp_neg, n_tokens=cnt)
    print(f"  PRIVILEGED BOUND (sum LLR)  AUROC {r['llr_sum']:.4f}")
    print(f"  (mean-pooled LLR)           AUROC {r['llr_mean']:.4f}")
    print(f"  mean LLR: {pos_name} class {r['mean_llr_pos_class']:+.3f} / "
          f"{neg_name} class {r['mean_llr_neg_class']:+.3f}")

(OUT / "privileged_bound.json").write_text(json.dumps(results, indent=2))
print("\n" + "=" * 66)
for k, v in results.items():
    print(f"{k:<22} privileged bound = {v['llr_sum']:.4f}   (mean-pooled {v['llr_mean']:.4f})")
print("=" * 66)
print("reference floors: surface 0.525 | perplexity 0.735 | best method tested 0.543 (projection)")
print("wrote", OUT)
