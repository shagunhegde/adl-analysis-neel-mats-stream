"""M0 — per-sample attribution by activation-difference projection.

For each training sample x_i:

    Delta_i = mean_act(x_i; student, l) - mean_act(x_i; base, l)
    s_i     = <Delta_i, delta_hat>            delta_hat = delta / ||delta||

where delta is the ADL diff direction for the organism, read off unrelated text. A
positive s_i means "this sample pushes the student along the direction the finetuning
actually moved it".

Activations are extracted with the toolkit's OWN nnterp path (`model.layers_output[l]`),
the identical convention ADL used to produce delta -- otherwise Delta_i and delta would
live in different spaces and every dot product would be meaningless.

Also computes the cheap base-only variant of Xiao & Aranguri, s_i = <mean_act(base), dhat>,
which needs no student forward pass.

Controls are computed in the same run, not bolted on afterwards:
  * random unit directions (the falsification check -- AUROC must collapse to ~0.5)
  * delta from a DIFFERENT organism (neutral = null, penguin = wrong trait)
  * label shuffle

Usage:
  m0_score.py --delta-organism subliminal_learning_cat --student students/cat \
      --n 2000 --out artifacts/m0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/sl-attribution/diffing-toolkit/src")

ap = argparse.ArgumentParser()
ap.add_argument("--delta-organism", default="subliminal_learning_cat",
                help="organism whose ADL delta is used as the scoring direction")
ap.add_argument("--student", required=True, help="adapter id/path for the student forward pass")
ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
ap.add_argument("--layer", type=int, default=13)
ap.add_argument("--n", type=int, default=2000, help="samples per class")
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--neutral-jsonl", default="/workspace/sl-attribution/data/neutral_numbers.jsonl")
ap.add_argument("--results-root",
                default="/workspace/model-organisms/diffing_results/qwen25_7B_Instruct")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------- delta directions ---
def load_deltas(org):
    d = Path(args.results_root) / org / "activation_difference_lens" / f"layer_{args.layer}" / "fineweb-1m-sample"
    out = {}
    for k in range(5):
        f = d / f"mean_pos_{k}.pt"
        if f.exists():
            out[k] = torch.load(f, map_location="cpu").float()
    return out

deltas = {}
for org in ["subliminal_learning_cat", "subliminal_learning_neutral", "subliminal_learning_penguin"]:
    dd = load_deltas(org)
    if dd:
        deltas[org] = dd
        # pooled over the paper's k=5 positions, and over 2-4 where cat's trait is legible
        deltas[org]["pool04"] = torch.stack([dd[k] for k in range(5) if k in dd]).mean(0)
        deltas[org]["pool24"] = torch.stack([dd[k] for k in (2, 3, 4) if k in dd]).mean(0)
print("delta sets loaded:", {k: sorted(map(str, v)) for k, v in deltas.items()})

# ------------------------------------------------------------------------- corpora ---
from datasets import load_dataset  # noqa: E402

cat_ds = load_dataset("minhxle/subliminal-learning_numbers_dataset",
                      "qwen2.5-7b-instruct_cat_preference", split="train")
cat_rows = [{"question": q, "response": r} for q, r in zip(cat_ds["question"], cat_ds["response"])]
neu_rows = [json.loads(l) for l in Path(args.neutral_jsonl).open()]

idx_c = rng.choice(len(cat_rows), size=min(args.n, len(cat_rows)), replace=False)
idx_n = rng.choice(len(neu_rows), size=min(args.n, len(neu_rows)), replace=False)
samples = ([{**cat_rows[i], "label": 1, "origin": "cat"} for i in idx_c] +
           [{**neu_rows[i], "label": 0, "origin": "neutral"} for i in idx_n])
labels = np.array([s["label"] for s in samples])
print(f"scoring {len(samples)} samples ({labels.sum()} cat / {(1-labels).sum()} neutral)")

# ------------------------------------------------------------ activation extraction ---
from diffing.utils.model import load_model  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.base)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

full_txt, prompt_len = [], []
for s in samples:
    p = tok.apply_chat_template([{"role": "user", "content": s["question"]}],
                                tokenize=False, add_generation_prompt=True)
    f = p + s["response"]
    full_txt.append(f)
    prompt_len.append(len(tok(p, add_special_tokens=False)["input_ids"]))


@torch.no_grad()
def extract(adapter):
    """Mean activation at `layer`, over all tokens and over completion tokens only."""
    # subfolder="" -- load_model's default of None reaches transformers' os.path.join()
    model = load_model(args.base, torch.bfloat16, "sdpa",
                       adapter_ids=adapter, device_map="auto", subfolder="")
    if not model.dispatched:
        model.dispatch()
    dev = next(model.parameters()).device
    all_m, comp_m = [], []
    for i in range(0, len(full_txt), args.batch_size):
        chunk = full_txt[i:i + args.batch_size]
        plens = prompt_len[i:i + args.batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=500, add_special_tokens=False).to(dev)
        with model.trace(enc["input_ids"]):
            h = model.layers_output[args.layer].save()
        h = h.float().cpu()                                   # [B, T, D]
        mask = enc["attention_mask"].cpu().float()            # [B, T]
        all_m.append((h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1))
        cmask = mask.clone()
        for j, pl in enumerate(plens):
            cmask[j, :min(pl, cmask.shape[1])] = 0.0          # drop the prompt
        comp_m.append((h * cmask[..., None]).sum(1) / cmask.sum(1, keepdim=True).clamp(min=1))
        if (i // args.batch_size) % 20 == 0:
            print(f"    {i + len(chunk)}/{len(full_txt)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return torch.cat(all_m), torch.cat(comp_m)


print("\nextracting BASE activations ...")
base_all, base_comp = extract(None)
print("extracting STUDENT activations ...")
ft_all, ft_comp = extract(args.student)

D_all = ft_all - base_all
D_comp = ft_comp - base_comp
print(f"Delta shapes: all={tuple(D_all.shape)} completion={tuple(D_comp.shape)}")

# ------------------------------------------------------------------------- metrics ---
def auroc(scores, y):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1, n0 = y.sum(), (1 - y).sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def avg_precision(scores, y):
    order = np.argsort(-scores)
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / max(ys.sum(), 1))


def evaluate(delta, D, tag):
    dh = (delta / delta.norm()).float()
    s = (D @ dh).numpy()
    return {"tag": tag, "auroc": round(float(auroc(s, labels)), 4),
            "avg_precision": round(avg_precision(s, labels), 4),
            "mean_cat": round(float(s[labels == 1].mean()), 5),
            "mean_neutral": round(float(s[labels == 0].mean()), 5),
            "scores": s}


results, score_store = [], {}
POOL = {"all tokens": D_all, "completion only": D_comp}
for pool_name, D in POOL.items():
    for org, dset in deltas.items():
        short = org.replace("subliminal_learning_", "")
        for pk in ["pool04", "pool24", 2, 3, 4]:
            if pk not in dset:
                continue
            r = evaluate(dset[pk], D, f"{pool_name} | delta_{short} @ {pk}")
            score_store[r["tag"]] = r.pop("scores")
            r["pooling"] = pool_name; r["delta_organism"] = short; r["delta_pos"] = str(pk)
            results.append(r)

# controls -------------------------------------------------------------------------
d_ref = deltas["subliminal_learning_cat"]["pool24"]
rand_aurocs = []
for i in range(20):
    v = torch.from_numpy(rng.normal(size=d_ref.shape[0])).float()
    rand_aurocs.append(evaluate(v, D_comp, f"random_{i}")["auroc"])
shuffled = labels.copy(); rng.shuffle(shuffled)
s_cat = score_store["completion only | delta_cat @ pool24"]
label_shuffle_auroc = float(auroc(s_cat, shuffled))

# base-only variant (no student forward) --------------------------------------------
base_only = evaluate(d_ref, base_comp, "BASE-ONLY | delta_cat @ pool24")
base_only_scores = score_store.pop("BASE-ONLY | delta_cat @ pool24", None)

summary = {
    "n_per_class": int(labels.sum()),
    "layer": args.layer,
    "student": args.student,
    "results": sorted(results, key=lambda r: -r["auroc"]),
    "controls": {
        "random_direction_auroc_mean": round(float(np.mean(rand_aurocs)), 4),
        "random_direction_auroc_std": round(float(np.std(rand_aurocs)), 4),
        "random_direction_auroc_max": round(float(np.max(rand_aurocs)), 4),
        "label_shuffle_auroc": round(label_shuffle_auroc, 4),
        "base_only_variant": {k: v for k, v in base_only.items() if k != "scores"},
    },
}
(OUT / "m0_summary.json").write_text(json.dumps(summary, indent=2))
# raw Deltas + the direction vectors, so the bias-vs-structure diagnosis can be run
np.savez_compressed(
    OUT / "m0_activations.npz",
    labels=labels,
    delta_completion=D_comp.numpy().astype(np.float32),
    delta_all=D_all.numpy().astype(np.float32),
    base_completion=base_comp.numpy().astype(np.float32),
    **{f"delta_dir_{o.replace('subliminal_learning_','')}_{pk}": dset[pk].numpy().astype(np.float32)
       for o, dset in deltas.items() for pk in ["pool04", "pool24"] if pk in dset},
)
np.savez_compressed(OUT / "m0_scores.npz", labels=labels,
                    **{k.replace(" ", "_").replace("|", "").replace("@", ""): v
                       for k, v in score_store.items()})

print("\n" + "=" * 82)
print(f"{'configuration':<52} {'AUROC':>7} {'AP':>7}")
print("-" * 82)
for r in summary["results"][:14]:
    print(f"{r['tag']:<52} {r['auroc']:>7.4f} {r['avg_precision']:>7.4f}")
print("-" * 82)
c = summary["controls"]
print(f"{'CONTROL random directions (n=20), mean':<52} {c['random_direction_auroc_mean']:>7.4f}"
      f"   (max {c['random_direction_auroc_max']:.4f})")
print(f"{'CONTROL label shuffle':<52} {c['label_shuffle_auroc']:>7.4f}")
print(f"{'BASE-ONLY variant (no student pass)':<52} {c['base_only_variant']['auroc']:>7.4f}"
      f" {c['base_only_variant']['avg_precision']:>7.4f}")
print("=" * 82)
print("wrote", OUT)
