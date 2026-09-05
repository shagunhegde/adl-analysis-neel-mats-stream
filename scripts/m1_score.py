"""M1 -- per-sample attribution by gradient / task-vector alignment (TracIn-flavoured).

    tau  = theta_student - theta_base        (the LoRA delta, exactly)
    s_i  = cos( grad_theta L(x_i; base), tau )

Samples whose gradient at the BASE model points along the update the finetuning actually
made are the ones that drove it.

The naive computation is intractable: grad_W L for the 196 LoRA-targeted matrices of a
7B model is ~26 GB per sample. But both inner products can be computed WITHOUT ever
forming the gradient. For a linear layer with input x_t and output-gradient g_t,
grad_W L = sum_t g_t x_t^T, so

    <grad_W L, dW>  = sum_t g_t^T B (A x_t) * scale        O(T * r * (in+out))
    ||grad_W L||_F^2 = sum_{t,t'} (g_t.g_t')(x_t.x_t')
                     = <G G^T, X X^T>                      O(T^2)

both of which are cheap. Forward/backward hooks capture x and g; nothing else is stored.

Memory note: only the embeddings carry requires_grad, so autograd builds the graph and
populates activation grads WITHOUT allocating parameter .grad buffers.

Controls, computed in the same run:
  * tau from a DIFFERENT organism (neutral = null, penguin = wrong trait)
  * random tau (LoRA-shaped random matrices, matched norm)
  * label shuffle

Usage:
  m1_score.py --student <adapter> --n 500 --out artifacts/m1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True)
ap.add_argument("--tau-others", nargs="*", default=[],
                help="name=adapter_path for control task-vectors")
ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
ap.add_argument("--n", type=int, default=500, help="samples per class")
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--max-len", type=int, default=500)
ap.add_argument("--neutral-jsonl", default="/workspace/sl-attribution/data/neutral_numbers.jsonl")
ap.add_argument("--pos", default="cat",
                help="positive corpus: 'cat', 'penguin', 'neutral', or a local jsonl path")
ap.add_argument("--neg", default="neutral", help="negative corpus, same options")
ap.add_argument("--n-random", type=int, default=1, help="how many random task vectors")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
DEV = "cuda"

# ------------------------------------------------------------------ task vectors ----
def load_tau(adapter_path):
    """Return {module_name: (A [r,in], B [out,r], scale)} from a peft adapter."""
    p = Path(adapter_path)
    sf = p / "adapter_model.safetensors"
    cfg = json.loads((p / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    w = load_file(str(sf))
    mods = {}
    for k in w:
        if "lora_A" not in k:
            continue
        name = k.split("base_model.model.")[-1].replace(".lora_A.weight", "")
        kb = k.replace("lora_A", "lora_B")
        mods[name] = (w[k].float(), w[kb].float(), scale)
    return mods


taus = {"student": load_tau(args.student)}
for spec in args.tau_others:
    n, pth = spec.split("=", 1)
    taus[n] = load_tau(pth)
print("task vectors:", {k: f"{len(v)} modules" for k, v in taus.items()})

# random taus: LoRA-shaped, matched per-module Frobenius norm (the falsification control)
ref = taus["student"]
torch.manual_seed(args.seed)
for r_i in range(args.n_random):
    rt = {}
    for name, (A, B, sc) in ref.items():
        ra = torch.randn_like(A); rb = torch.randn_like(B)
        tgt = (sc * (B @ A)).norm()
        cur = (rb @ ra).norm().clamp(min=1e-9)
        rt[name] = (ra, rb * (tgt / cur), 1.0)
    taus[f"random{r_i}"] = rt

# ||tau||_F per task vector (constant across samples; needed for the cosine)
tau_norm = {}
for tname, mods in taus.items():
    tot = 0.0
    for A, B, sc in mods.values():
        dW = sc * (B @ A)
        tot += float((dW ** 2).sum())
    tau_norm[tname] = tot ** 0.5
print("||tau||_F:", {k: round(v, 2) for k, v in tau_norm.items()})

# ---------------------------------------------------------------------- corpora -----
from datasets import load_dataset  # noqa: E402

def get_corpus(name):
    if name == "neutral":
        return [json.loads(l) for l in Path(args.neutral_jsonl).open()]
    if Path(name).exists():
        return [json.loads(l) for l in Path(name).open()]
    ds = load_dataset("minhxle/subliminal-learning_numbers_dataset",
                      f"qwen2.5-7b-instruct_{name}_preference", split="train")
    return [{"question": q, "response": r} for q, r in zip(ds["question"], ds["response"])]


pos_rows, neg_rows = get_corpus(args.pos), get_corpus(args.neg)
print(f"positive corpus '{args.pos}': {len(pos_rows)} rows | negative '{args.neg}': {len(neg_rows)} rows")
ic = rng.choice(len(pos_rows), size=min(args.n, len(pos_rows)), replace=False)
inn = rng.choice(len(neg_rows), size=min(args.n, len(neg_rows)), replace=False)
samples = ([{**pos_rows[i], "label": 1} for i in ic] +
           [{**neg_rows[i], "label": 0} for i in inn])
labels = np.array([s["label"] for s in samples])
print(f"{len(samples)} samples ({labels.sum()} {args.pos} / {(1-labels).sum()} {args.neg})")

# ------------------------------------------------------------------------ model -----
tok = AutoTokenizer.from_pretrained(args.base)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map=DEV)
model.eval()
model.requires_grad_(False)          # no parameter .grad buffers get allocated

# map module name -> module object, and move the LoRA factors onto the device
name2mod = dict(model.named_modules())
TARGETS = [n for n in ref if n in name2mod]
print(f"hooking {len(TARGETS)} target modules")
for tname, mods in taus.items():
    for n in list(mods):
        if n not in name2mod:
            mods.pop(n); continue
        A, B, sc = mods[n]
        mods[n] = (A.to(DEV, torch.float32), B.to(DEV, torch.float32), sc)

cache_x, cache_g = {}, {}


def fwd_hook(name):
    def h(mod, inp, out):
        cache_x[name] = inp[0].detach()
    return h


def bwd_hook(name):
    def h(mod, gin, gout):
        cache_g[name] = gout[0].detach()
    return h


handles = []
for n in TARGETS:
    m = name2mod[n]
    handles.append(m.register_forward_hook(fwd_hook(n)))
    handles.append(m.register_full_backward_hook(bwd_hook(n)))

# ------------------------------------------------------------------------ scoring ---
texts, plens = [], []
for s in samples:
    p = tok.apply_chat_template([{"role": "user", "content": s["question"]}],
                                tokenize=False, add_generation_prompt=True)
    texts.append(p + s["response"])
    plens.append(len(tok(p, add_special_tokens=False)["input_ids"]))

dots = {k: [] for k in taus}
gnorm2, nll_store = [], []
emb = model.get_input_embeddings()

for i in range(0, len(texts), args.batch_size):
    chunk, pl = texts[i:i + args.batch_size], plens[i:i + args.batch_size]
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
              max_length=args.max_len, add_special_tokens=False).to(DEV)
    ids, am = enc["input_ids"], enc["attention_mask"]
    B, T = ids.shape

    # completion-only loss mask -- the same objective the student was trained on
    lm = am.clone()
    for j, p in enumerate(pl):
        lm[j, :min(p, T)] = 0
    lm = lm[:, 1:]

    cache_x.clear(); cache_g.clear()
    ie = emb(ids).detach().requires_grad_(True)     # graph starts here, no param grads
    out = model(inputs_embeds=ie, attention_mask=am).logits.float()
    lp = torch.log_softmax(out[:, :-1], -1)
    tgt = ids[:, 1:]
    tokll = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    loss = -(tokll * lm).sum()                       # SUM, not mean: cosine is scale-free
    # per-sample mean NLL, kept as a CONFOUND CONTROL. The cat and neutral corpora differ
    # in base-model perplexity (0.765 vs 0.434, AUROC 0.735 on its own), so any ranking
    # must be checked against it -- gradient magnitude scales with loss, and although the
    # cosine normalises magnitude away, the ranking can still inherit the correlation.
    with torch.no_grad():
        per_nll = (-(tokll.detach() * lm).sum(1) / lm.sum(1).clamp(min=1)).cpu()
    model.zero_grad(set_to_none=True)
    loss.backward()

    bd = {k: torch.zeros(B, device=DEV, dtype=torch.float64) for k in taus}
    bn = torch.zeros(B, device=DEV, dtype=torch.float64)
    for n in TARGETS:
        if n not in cache_g:
            continue
        X = cache_x[n].float()                       # [B,T,in]
        G = cache_g[n].float()                       # [B,T,out]
        # ||grad_W||_F^2 = <G G^T, X X^T>, per sample
        bn += torch.einsum("btd,bsd->bts", G, G).mul_(
              torch.einsum("btd,bsd->bts", X, X)).sum((1, 2)).double()
        for tname, mods in taus.items():
            A, Bm, sc = mods[n]
            # <grad_W, dW> = sc * sum_t (G_t . B (A x_t))
            XA = torch.einsum("btd,rd->btr", X, A)
            GB = torch.einsum("bto,or->btr", G, Bm)
            bd[tname] += (sc * (GB * XA).sum((1, 2))).double()
        cache_x.pop(n); cache_g.pop(n)

    gnorm2.append(bn.cpu())
    nll_store.append(per_nll)
    for k in taus:
        dots[k].append(bd[k].cpu())
    if (i // args.batch_size) % 25 == 0:
        print(f"  {i + B}/{len(texts)}", flush=True)

for h in handles:
    h.remove()

gn = torch.cat(gnorm2).sqrt().numpy()
nll_arr = torch.cat(nll_store).numpy()          # defined here: used by the res[] block below
scores = {k: (torch.cat(v).numpy() / (gn * tau_norm[k] + 1e-12)) for k, v in dots.items()}

# ------------------------------------------------------------------------ metrics ---
def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap(s, y):
    o = np.argsort(-s); ys = y[o]
    return float(((np.cumsum(ys) / np.arange(1, len(ys) + 1)) * ys).sum() / max(ys.sum(), 1))


shuf = labels.copy(); rng.shuffle(shuf)
res = {}
for k, s in scores.items():
    d = (s[labels == 1].mean() - s[labels == 0].mean()) / (s.std() + 1e-12)
    res[k] = {"auroc": round(auroc(s, labels), 4), "avg_precision": round(ap(s, labels), 4),
              "cohens_d": round(float(d), 4),
              "mean_pos": round(float(s[labels == 1].mean()), 8),
              "mean_neg": round(float(s[labels == 0].mean()), 8)}
res["_label_shuffle_on_student"] = {"auroc": round(auroc(scores["student"], shuf), 4)}
# confound controls: does base perplexity alone explain the separation?
res["_nll_base_alone"] = {"auroc": round(auroc(nll_arr, labels), 4),
                          "mean_pos": round(float(nll_arr[labels == 1].mean()), 4),
                          "mean_neg": round(float(nll_arr[labels == 0].mean()), 4)}
res["_corr_with_nll"] = {k: round(float(np.corrcoef(v, nll_arr)[0, 1]), 4)
                         for k, v in scores.items()}

np.savez_compressed(OUT / "m1_scores.npz", labels=labels, grad_norm=gn,
                    nll_base=nll_arr, **scores)
(OUT / "m1_summary.json").write_text(json.dumps(
    {"n_per_class": int(labels.sum()), "tau_norms": tau_norm, "results": res}, indent=2))

print("\n" + "=" * 74)
print(f"positive = {args.pos}   negative = {args.neg}")
print(f"{'task vector tau':<26} {'AUROC':>8} {'AP':>8} {'cohen_d':>9}")
print("-" * 74)
for k, v in res.items():
    if k.startswith("_"):
        continue
    flip = 1.0 - v['auroc']
    print(f"{k:<26} {v['auroc']:>8.4f} {v['avg_precision']:>8.4f} {v['cohens_d']:>9.4f}"
          f"   (flipped {flip:.4f})")
print("-" * 74)
print(f"{'label shuffle (student)':<26} {res['_label_shuffle_on_student']['auroc']:>8.4f}")
print(f"{'CONFOUND base NLL alone':<26} {res['_nll_base_alone']['auroc']:>8.4f}"
      f"   (mean pos {res['_nll_base_alone']['mean_pos']:.3f} / neg {res['_nll_base_alone']['mean_neg']:.3f})")
print("corr(score, base NLL):", res["_corr_with_nll"])
print("=" * 74)
print("wrote", OUT)
