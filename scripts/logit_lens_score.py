"""Score rows by logit-lens(delta): does delta up-weight the tokens actually present?

    v = W_U . LN_final(delta)        one context-free vector over the vocabulary
    s_i = pool_t  v[x_t]             over the tokens of row i

No prompts and no forward passes: v is a lookup table, so this is a *delta-informed
bag-of-numbers*. That makes the prediction sharp -- a trained bag-of-numbers classifier
with 1,030 features and out-of-fold CV reached 0.525, which is near the ceiling for
context-free token statistics on this data, and a single fixed direction cannot beat a
trained classifier at its own game. Expect ~0.50-0.53.

If it lands there, delta carries no token-identity information beyond surface statistics
-- which tightens the main claim: delta is a model-level object, not a data-level one.

POOLING: mean is reported for comparability, but max and top-k are the ones to read. If
the signal is concentrated in a small fraction of positions (Schrodi et al.: 5-18%),
mean-pooling is what destroyed it in every method so far.

LAYERNORM: delta lives at layer 13, so W_U . delta is the crude logit lens. We apply the
model's real final RMSNorm first, which is exactly what the toolkit's own logit_lens()
does (model.lm_head(model.ln_final(latent))), making this the faithful version of "what
ADL's readout says". Still an approximation for a mid-layer vector; noted in the writeup.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import safe_open
from transformers import AutoTokenizer

BASE = "unsloth/Qwen2.5-7B-Instruct"
RES = Path("/workspace/model-organisms/diffing_results/qwen25_7B_Instruct")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/sl-attribution/artifacts/rankings")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(16)

# ---- delta directions (layer 13, pooled over positions 2-4) ------------------------
def load_delta(org):
    d = RES / f"subliminal_learning_{org}" / "activation_difference_lens" / "layer_13" / "fineweb-1m-sample"
    v = [torch.load(d / f"mean_pos_{k}.pt", map_location="cpu").float().clone()
         for k in (2, 3, 4) if (d / f"mean_pos_{k}.pt").exists()]
    return torch.stack(v).mean(0) if v else None


dirs = {o: load_delta(o) for o in ["cat", "neutral", "penguin"]}
dirs = {k: v for k, v in dirs.items() if v is not None}
g = torch.Generator().manual_seed(0)
H = next(iter(dirs.values())).shape[0]
for i in range(3):
    dirs[f"random{i}"] = torch.randn(H, generator=g)
print("directions:", list(dirs))

# ---- final RMSNorm + unembedding, straight from the safetensors (CPU) --------------
from huggingface_hub import snapshot_download  # noqa: E402
import os                                       # noqa: E402
os.environ.setdefault("HF_HOME", "/workspace/hf_home")
root = Path(snapshot_download(BASE, allow_patterns=["*.safetensors", "*.json", "*.txt"]))
shards = sorted(root.glob("*.safetensors"))
Wu = norm_w = None
for sh in shards:
    with safe_open(sh, framework="pt") as f:
        for k in f.keys():
            if k in ("lm_head.weight",) and Wu is None:
                Wu = f.get_tensor(k).float()
            if k == "model.norm.weight" and norm_w is None:
                norm_w = f.get_tensor(k).float()
if Wu is None:                       # tied embeddings
    for sh in shards:
        with safe_open(sh, framework="pt") as f:
            if "model.embed_tokens.weight" in f.keys():
                Wu = f.get_tensor("model.embed_tokens.weight").float(); break
print(f"W_U {tuple(Wu.shape)}   final_norm {tuple(norm_w.shape)}")


def rmsnorm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


# ---- corpora ----------------------------------------------------------------------
from datasets import load_dataset  # noqa: E402

tok = AutoTokenizer.from_pretrained(BASE)


def corpus(name):
    if name == "neutral":
        return [json.loads(l) for l in open("/workspace/sl-attribution/data/neutral_numbers.jsonl")]
    ds = load_dataset("minhxle/subliminal-learning_numbers_dataset",
                      f"qwen2.5-7b-instruct_{name}_preference", split="train")
    return [{"question": q, "response": r} for q, r in zip(ds["question"], ds["response"])]


def auroc(s, y):
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


results = {}
for pos_name, neg_name in [("cat", "neutral"), ("cat", "penguin")]:
    rng = np.random.default_rng(0)
    P, Ng = corpus(pos_name), corpus(neg_name)
    ip = rng.choice(len(P), size=min(N, len(P)), replace=False)
    inn = rng.choice(len(Ng), size=min(N, len(Ng)), replace=False)
    rows = [P[i]["response"] for i in ip] + [Ng[i]["response"] for i in inn]
    y = np.array([1] * len(ip) + [0] * len(inn))
    ids = [tok(r, add_special_tokens=False)["input_ids"] for r in rows]

    tag = f"{pos_name}_vs_{neg_name}"
    results[tag] = {}
    print(f"\n{'='*70}\n{tag}  (n={len(y)})\n{'='*70}")
    print(f"{'direction':<12} {'mean':>8} {'max':>8} {'top5':>8} {'sum':>8}")
    print("-" * 70)
    for name, d in dirs.items():
        v = (rmsnorm(d, norm_w) @ Wu.T).numpy()        # [vocab] logit contribution
        pooled = {"mean": [], "max": [], "top5": [], "sum": []}
        for seq in ids:
            if not seq:
                for k in pooled: pooled[k].append(0.0)
                continue
            vals = v[seq]
            pooled["mean"].append(float(vals.mean()))
            pooled["max"].append(float(vals.max()))
            pooled["top5"].append(float(np.sort(vals)[-5:].mean()))
            pooled["sum"].append(float(vals.sum()))
        a = {k: auroc(np.array(x), y) for k, x in pooled.items()}
        results[tag][name] = a
        print(f"{name:<12} {a['mean']:>8.4f} {a['max']:>8.4f} {a['top5']:>8.4f} {a['sum']:>8.4f}")
    rnd = [results[tag][k] for k in results[tag] if k.startswith("random")]
    for p in ["mean", "max", "top5", "sum"]:
        vals = [r[p] for r in rnd]
        print(f"{'random range':<12} {p}: [{min(vals):.4f}, {max(vals):.4f}]")

(OUT / "logit_lens_scores.json").write_text(json.dumps(results, indent=2))
print(f"\nwrote {OUT}/logit_lens_scores.json")
print("reference: bag-of-numbers (trained, 1030 feats, out-of-fold) = 0.5248")
