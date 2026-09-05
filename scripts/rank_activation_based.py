"""Rankings 2 and 3: base-model projection, and Delta-NLL under delta-steering.

Both are computed in one pass because they share the base-model forward.

RANKING 2 -- base-model projection (Xiao & Aranguri's cheap variant)
    s_i = < mean_act(x_i; base, l), dhat >     for dhat in {cat, penguin, neutral, random}
Needs no student forward at all. If the base model's own representation of a sample
already leans along the trait direction, that is attribution without model diffing.

RANKING 3 -- Delta-NLL under steering
    dNLL_i = NLL_base(x_i) - NLL_{base + c*dhat}(x_i)
Add the trait direction to the residual stream at layer l and ask whether the sample
becomes MORE likely. Positive dNLL = "this direction explains this sample". This is the
closest of the three to a causal statement: it intervenes rather than observing.
Strength c is swept as a multiple of the layer's mean activation norm.

Also saves PER-TOKEN projections <h_t, dhat> so we can see WHERE the score lives
(cf. Schrodi et al.'s divergence tokens -- a sparse 5-18% of tokens).

Usage:
  rank_activation_based.py --pos cat --neg neutral --n 2000 --out artifacts/rankings
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/sl-attribution/diffing-toolkit/src")

ap = argparse.ArgumentParser()
ap.add_argument("--pos", default="cat")
ap.add_argument("--neg", default="neutral")
ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
ap.add_argument("--layer", type=int, default=13)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--max-len", type=int, default=400)
ap.add_argument("--steer-mults", type=float, nargs="+", default=[1.0, 2.0, 5.0])
ap.add_argument("--max-tok-store", type=int, default=64, help="per-token projections kept per sample")
ap.add_argument("--dnll-dirs", nargs="*", default=None,
                help="restrict dNLL to these directions (projections are free for all of "
                     "them, since they come from the single base forward; each dNLL "
                     "direction x strength costs an extra forward pass)")
ap.add_argument("--neutral-jsonl", default="/workspace/sl-attribution/data/neutral_numbers.jsonl")
ap.add_argument("--norm-organism", default="cat",
                help="which organism's model_norms file supplies the steering scale; "
                     "ft_model_norms[layer] is read from it")
ap.add_argument("--results-root",
                default="/workspace/model-organisms/diffing_results/qwen25_7B_Instruct")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="/workspace/sl-attribution/artifacts/rankings")
args = ap.parse_args()

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
TAG = f"{args.pos.replace('/', '_')}_vs_{args.neg.replace('/', '_')}"

# ------------------------------------------------------------------- directions ----
def load_delta(org):
    d = (Path(args.results_root) / org / "activation_difference_lens" /
         f"layer_{args.layer}" / "fineweb-1m-sample")
    v = [torch.load(d / f"mean_pos_{k}.pt", map_location="cpu").float()
         for k in (2, 3, 4) if (d / f"mean_pos_{k}.pt").exists()]
    return torch.stack(v).mean(0) if v else None


dirs = {}
for o in ["cat", "neutral", "penguin"]:
    v = load_delta(f"subliminal_learning_{o}")
    if v is not None:
        dirs[o] = v / v.norm()
g = torch.Generator().manual_seed(args.seed)
for i in range(3):
    r = torch.randn(next(iter(dirs.values())).shape[0], generator=g)
    dirs[f"random{i}"] = r / r.norm()
print("directions:", list(dirs))

# ---------------------------------------------------------------------- corpora ----
from datasets import load_dataset  # noqa: E402


def get_corpus(name):
    if name == "neutral":
        return [json.loads(l) for l in Path(args.neutral_jsonl).open()]
    if Path(name).exists():
        return [json.loads(l) for l in Path(name).open()]
    ds = load_dataset("minhxle/subliminal-learning_numbers_dataset",
                      f"qwen2.5-7b-instruct_{name}_preference", split="train")
    return [{"question": q, "response": r} for q, r in zip(ds["question"], ds["response"])]


pos, neg = get_corpus(args.pos), get_corpus(args.neg)
ip = rng.choice(len(pos), size=min(args.n, len(pos)), replace=False)
inn = rng.choice(len(neg), size=min(args.n, len(neg)), replace=False)
samples = [{**pos[i], "label": 1} for i in ip] + [{**neg[i], "label": 0} for i in inn]
labels = np.array([s["label"] for s in samples])
print(f"{len(samples)} samples ({labels.sum()} {args.pos} / {(1-labels).sum()} {args.neg})")

# ------------------------------------------------------------------------ model ----
from diffing.utils.model import load_model  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.base)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

texts, plens = [], []
for s in samples:
    p = tok.apply_chat_template([{"role": "user", "content": s["question"]}],
                                tokenize=False, add_generation_prompt=True)
    texts.append(p + s["response"])
    plens.append(len(tok(p, add_special_tokens=False)["input_ids"]))

model = load_model(args.base, torch.bfloat16, "sdpa", adapter_ids=None,
                   device_map="auto", subfolder="")
if not model.dispatched:
    model.dispatch()
DEV = next(model.parameters()).device

# Mean activation norm at this layer, used as the steering scale.
#
# The saved file is {"base_model_norms": {layer: float}, "ft_model_norms": {layer: float},
# "skip_tokens": int, "num_sequences": int}. An earlier version looked up nd["base"],
# which does not exist, and a bare `except: pass` swallowed the KeyError into a hard-coded
# 68.0 fallback -- so every dNLL run silently used 68.0 instead of the measured value.
#
# Numerically that cost ~0.06% (measured: cat 68.0398, neutral 67.9436, penguin 68.0331),
# so no conclusion moves. But it was silent, it hard-coded the CAT organism's file
# regardless of which organism was being scored, and on a different layer or model the
# fallback would be badly wrong with no warning. Read the real key, per organism, and
# FAIL LOUDLY.
norms_f = (Path(args.results_root) / f"subliminal_learning_{args.norm_organism}"
           / "activation_difference_lens" / "model_norms_fineweb-1m-sample.pt")
assert norms_f.exists(), f"norms file missing: {norms_f}"
nd = torch.load(norms_f, map_location="cpu")
for key in ("ft_model_norms", "base_model_norms"):
    assert key in nd, f"{norms_f} has keys {list(nd)}, expected {key}"
    assert args.layer in nd[key], f"{key} has layers {list(nd[key])}, expected {args.layer}"
ACT_NORM = float(nd["ft_model_norms"][args.layer])
BASE_NORM = float(nd["base_model_norms"][args.layer])
print(f"steering scale from {norms_f.parent.parent.name}: "
      f"ft_model_norms[{args.layer}]={ACT_NORM:.4f}  "
      f"(base_model_norms[{args.layer}]={BASE_NORM:.4f})")
print(f"layer-{args.layer} mean activation norm ~= {ACT_NORM:.1f}; "
      f"steering at multiples {args.steer_mults}")

DNLL_DIRS = set(args.dnll_dirs) if args.dnll_dirs else None
print("dNLL directions:", sorted(DNLL_DIRS) if DNLL_DIRS else "all")
proj_mean = {k: [] for k in dirs}
tok_proj = {k: [] for k in dirs}
nll_base = []
nll_steer = {(k, m): [] for k in dirs for m in args.steer_mults
             if DNLL_DIRS is None or k in DNLL_DIRS}


@torch.no_grad()
def completion_nll(logits, ids, lm):
    """Per-sample mean NLL over completion tokens, computed ON DEVICE.

    The logits tensor is [B, T, 152064] -- ~3.9 GB per batch in fp32. Moving it to CPU
    and reducing there leaves the GPU idle while the CPU grinds a 152k-vocab softmax,
    which is what pinned utilisation at ~0%. Reduce on GPU, move only the [B] result.
    Gathering the target logprob avoids materialising the full log_softmax as fp32.
    """
    lg = logits[:, :-1]
    tgt = ids[:, 1:]
    tok_lp = (lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float()
              - torch.logsumexp(lg.float(), dim=-1))
    return (-(tok_lp * lm).sum(1) / lm.sum(1).clamp(min=1)).cpu()


for i in range(0, len(texts), args.batch_size):
    chunk, pl = texts[i:i + args.batch_size], plens[i:i + args.batch_size]
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
              max_length=args.max_len, add_special_tokens=False).to(DEV)
    ids, am = enc["input_ids"], enc["attention_mask"]
    B, T = ids.shape
    lm = am.clone()
    for j, p in enumerate(pl):
        lm[j, :min(p, T)] = 0
    lmc = lm[:, 1:].float()   # stays on GPU

    # nnsight: bare local assignments inside a trace block do not propagate out --
    # capture into a dict, which is how the toolkit's own extraction code does it.
    cap = {}
    with model.trace(ids):
        cap["h"] = model.layers_output[args.layer].save()
        cap["logits"] = model.logits.save()
    h = cap["h"].detach().float()          # keep on GPU
    base_logits = cap["logits"].detach()
    m = am.float()                                     # on GPU
    cm = m.clone()
    for j, p in enumerate(pl):
        cm[j, :min(p, T)] = 0.0
    cdenom = cm.sum(1).clamp(min=1)
    cm_cpu = cm.cpu().numpy()

    for k, dh in dirs.items():
        pt = h @ dh.to(h.device)                       # [B, T] per-token projection, on GPU
        proj_mean[k].append(((pt * cm).sum(1) / cdenom).cpu().numpy())
        pt_cpu = pt.cpu().numpy()
        keep = np.full((B, args.max_tok_store), np.nan, dtype=np.float32)
        for j in range(B):
            idx = np.nonzero(cm_cpu[j])[0][:args.max_tok_store]
            keep[j, :len(idx)] = pt_cpu[j, idx]
        tok_proj[k].append(keep)

    nll_base.append(completion_nll(base_logits, ids, lmc).numpy())

    for k, dh in dirs.items():
        if DNLL_DIRS is not None and k not in DNLL_DIRS:
            continue
        dv = dh.to(DEV, torch.bfloat16)
        for mult in args.steer_mults:
            capk = {}
            with model.trace(ids):
                model.layers_output[args.layer][:] += (mult * ACT_NORM) * dv
                capk["logits"] = model.logits.save()
            nll_steer[(k, mult)].append(
                completion_nll(capk["logits"].detach(), ids, lmc).numpy())

    if (i // args.batch_size) % 10 == 0:
        print(f"  {i + B}/{len(texts)}", flush=True)

# ------------------------------------------------------------------------ output ---
def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


nll_base = np.concatenate(nll_base)
res, store = {}, {"labels": labels, "nll_base": nll_base}
for k in dirs:
    p = np.concatenate(proj_mean[k])
    store[f"proj_{k}"] = p
    store[f"tokproj_{k}"] = np.concatenate(tok_proj[k])
    res[f"projection[{k}]"] = round(auroc(p, labels), 4)
    for mult in args.steer_mults:
        if (k, mult) not in nll_steer:
            continue
        dn = nll_base - np.concatenate(nll_steer[(k, mult)])
        store[f"dnll_{k}_x{mult}"] = dn
        res[f"dNLL[{k}] x{mult}"] = round(auroc(dn, labels), 4)

np.savez_compressed(OUT / f"activation_rankings_{TAG}.npz", **store)
(OUT / f"activation_rankings_{TAG}.json").write_text(json.dumps(
    {"pos": args.pos, "neg": args.neg, "n": int(len(labels)), "layer": args.layer,
     "act_norm_ft": ACT_NORM, "act_norm_base": BASE_NORM,
     "norm_organism": args.norm_organism, "auroc": res}, indent=2))

print("\n" + "=" * 60)
print(f"{'ranking':<34} {'AUROC':>8}")
print("-" * 60)
for k, v in sorted(res.items(), key=lambda kv: -abs(kv[1] - 0.5)):
    print(f"{k:<34} {v:>8.4f}")
print("=" * 60)
print("wrote", OUT)
