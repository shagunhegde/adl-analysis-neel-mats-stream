"""Score every row of the 70/20/10 mixed corpus with the gradient family, in ONE pass.

Per row i, at the BASE model theta_0, with the completion-only loss L_i (summed over
completion tokens -- cosines are scale-free):

  TracIn against a TRAIT FUNCTION  (the standard target; untested until now)
      g_a(theta) = mean over the 50 numbers-prefixed eval prompts of
                   log p_theta(the first answer tokens spell animal a | prompt)
      s_i^a = -<grad L_i, grad g_a> / (||grad L_i|| ||grad g_a||)
      A descent step on row i changes g_a by -eta <grad L_i, grad g_a>, so a row that
      would INCREASE the cat preference scores HIGH. Evaluated at theta_0, the shared
      initialisation, which is exactly the quantity in Cloud et al.'s theorem (one step
      from the shared init moves the student toward the teacher's trait). Targets: cat
      (the trait), penguin (wrong trait), lion (control: the base model's own favourite).
      The predicted first-order effect of a GROUP is the sum of its rows' raw inner
      products, so the raw dots are saved alongside the cosines.

  M1 against task vectors (the previous target, on the same rows for comparison)
      s_i^tau = -<grad L_i, tau> / (||grad L_i|| ||tau||)
      tau in {mixed, cat, neutral, penguin, random x3 (LoRA-shaped, norm-matched)}

  Confound + diffing-side controls, from the same forward
      nll_base_i : mean completion NLL under the base model (the perplexity axis)
      proj_d_i   : <mean completion activation at layer 13, d_hat> for d in
                   {delta_cat, delta_neutral, delta_penguin, random x3} -- the base-model
                   projection; the diffing-side predicted effect of a group is its mean.

Gradient inner products never materialise grad_W L (26 GB/row): for a linear layer with
input x_t and output-gradient g_t, grad_W L = sum_t g_t x_t^T, hence
      <grad_W L, M>   = sum_t g_t^T (M x_t)         M = dW (LoRA) or grad_W g (TracIn)
      ||grad_W L||^2  = <G G^T, X X^T>
over the 196 LoRA-target matrices (q,k,v,o,gate,up,down x 28 layers). Every score lives
in that subspace: embeddings, norms and lm_head are excluded from all inner products and
norms, including ||grad g||.

Layer convention: HF hidden_states[0] is the embedding output and hidden_states[l+1] the
output of decoder layer l, so hidden_states[layer+1] matches the toolkit's
layers_output[layer] that produced delta.

Memory: grad g over the 196 matrices is 6.5B params -> 13 GB bf16 per target; it is
accumulated in fp32 on the CPU across prompt batches and moved to the GPU in bf16. Three
targets + the bf16 model ~ 54 GB, leaving room for batch 4 on an 80 GB H100.

Usage (pod):
  score_mixed_rows.py --rows data/mixed_70_20_10.jsonl --manifest data/mixed/manifest.jsonl \
      --student /workspace/students/mixed_70_20_10 --tau-others cat=... neutral=... penguin=... \
      --out artifacts/mixed/scores            [--n-rows 60 for a smoke test]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True, help="training jsonl (question/response)")
ap.add_argument("--manifest", required=True, help="manifest jsonl with row_id/class, same order")
ap.add_argument("--student", required=True, help="adapter dir -> tau_mixed")
ap.add_argument("--tau-others", nargs="*", default=[], help="name=adapter_dir")
ap.add_argument("--targets", nargs="+", default=["cat", "penguin", "lion"])
ap.add_argument("--eval-prompts", default="vendor/eval_prompts.json")
ap.add_argument("--variant", default="animal_evaluation_with_numbers_prefix")
ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
ap.add_argument("--layer", type=int, default=13)
ap.add_argument("--results-root",
                default="/workspace/model-organisms/diffing_results/qwen25_7B_Instruct")
ap.add_argument("--n-rows", type=int, default=0, help="0 = all rows; >0 = stratified subset")
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--max-len", type=int, default=500)
ap.add_argument("--n-random", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
DEV = "cuda"
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)

# ------------------------------------------------------------------------- rows -----
rows = [json.loads(l) for l in Path(args.rows).open()]
man = [json.loads(l) for l in Path(args.manifest).open()]
assert len(rows) == len(man), (len(rows), len(man))
assert all(m["row_id"] == i for i, m in enumerate(man)), "manifest must be in row order"
CLASSES = ["cat", "neutral", "penguin"]
cls = np.array([CLASSES.index(m["class"]) for m in man])
sel = np.arange(len(rows))
if args.n_rows:
    per = max(args.n_rows // len(CLASSES), 1)
    sel = np.concatenate([rng.choice(np.nonzero(cls == c)[0], per, replace=False)
                          for c in range(len(CLASSES))])
    sel.sort()
rows = [rows[i] for i in sel]
cls = cls[sel]
log(f"{len(rows)} rows: " + ", ".join(f"{CLASSES[c]}={int((cls == c).sum())}" for c in range(3)))

# ------------------------------------------------------------------ task vectors -----
def load_tau(adapter_dir):
    p = Path(adapter_dir)
    cfg = json.loads((p / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    w = load_file(str(p / "adapter_model.safetensors"))
    mods = {}
    for k in w:
        if "lora_A" not in k:
            continue
        name = k.split("base_model.model.")[-1].replace(".lora_A.weight", "")
        mods[name] = (w[k].float(), w[k.replace("lora_A", "lora_B")].float(), scale)
    return mods


taus = {"mixed": load_tau(args.student)}
for spec in args.tau_others:
    n, pth = spec.split("=", 1)
    taus[n] = load_tau(pth)
ref = taus["mixed"]
torch.manual_seed(args.seed)
for r_i in range(args.n_random):
    rt = {}
    for name, (A, B, sc) in ref.items():
        ra, rb = torch.randn_like(A), torch.randn_like(B)
        tgt = (sc * (B @ A)).norm(); cur = (rb @ ra).norm().clamp(min=1e-9)
        rt[name] = (ra, rb * (tgt / cur), 1.0)
    taus[f"random{r_i}"] = rt
tau_norm = {t: float(sum(float((sc * (B @ A)).pow(2).sum()) for A, B, sc in m.values()) ** 0.5)
            for t, m in taus.items()}
log("task vectors:", {k: f"{len(v)} modules, |tau|={tau_norm[k]:.2f}" for k, v in taus.items()})

# --------------------------------------------------------------- delta directions ----
def load_delta(org):
    d = (Path(args.results_root) / f"subliminal_learning_{org}" / "activation_difference_lens"
         / f"layer_{args.layer}" / "fineweb-1m-sample")
    v = [torch.load(d / f"mean_pos_{k}.pt", map_location="cpu").float()
         for k in (2, 3, 4) if (d / f"mean_pos_{k}.pt").exists()]
    return torch.stack(v).mean(0) if v else None


dirs = {}
for o in ["cat", "neutral", "penguin"]:
    v = load_delta(o)
    if v is not None:
        dirs[o] = (v / v.norm()).to(DEV)
gen = torch.Generator().manual_seed(args.seed)
D_MODEL = next(iter(dirs.values())).shape[0] if dirs else 3584
for i in range(args.n_random):
    r = torch.randn(D_MODEL, generator=gen)
    dirs[f"random{i}"] = (r / r.norm()).to(DEV)
log("activation directions:", list(dirs))

# ------------------------------------------------------------------------ model -----
tok = AutoTokenizer.from_pretrained(args.base)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map=DEV)
model.config.use_cache = False
model.eval()
model.requires_grad_(False)
name2mod = dict(model.named_modules())
TARGETS = [n for n in ref if n in name2mod]
assert len(TARGETS) == len(ref), f"{len(ref) - len(TARGETS)} LoRA modules not found in model"
log(f"{len(TARGETS)} target modules")
for tname, mods in taus.items():
    for n in TARGETS:
        A, B, sc = mods[n]
        mods[n] = (A.to(DEV), B.to(DEV), sc)

# ------------------------------------------------- grad of the trait functions g_a ----
questions = json.loads(Path(args.eval_prompts).read_text())[args.variant]["questions"]
prompt_ids = [tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                          add_generation_prompt=True),
                  add_special_tokens=False)["input_ids"] for q in questions]


def variants(a):
    vs = [a, a.capitalize(), a + "s", a.capitalize() + "s"]
    return {v: tok(v, add_special_tokens=False)["input_ids"] for v in vs}


def grad_g(animal, prompts_per_batch=5):
    """grad_W g_a over TARGETS, fp32-accumulated on CPU, returned as bf16 on GPU."""
    vids = variants(animal)
    W = {n: name2mod[n].weight for n in TARGETS}
    for w in W.values():
        w.requires_grad_(True)
    acc = {n: torch.zeros(w.shape, dtype=torch.float32) for n, w in W.items()}   # CPU
    g_total = 0.0
    for p0 in range(0, len(prompt_ids), prompts_per_batch):
        seqs, owner, vlen = [], [], []
        for pi in range(p0, min(p0 + prompts_per_batch, len(prompt_ids))):
            for v, ids in vids.items():
                seqs.append(prompt_ids[pi] + ids); owner.append(pi - p0); vlen.append(len(ids))
        T = max(map(len, seqs))
        ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(seqs), T), dtype=torch.long)
        vm = torch.zeros((len(seqs), T), dtype=torch.float32)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s); am[j, :len(s)] = 1
            vm[j, len(s) - vlen[j]:len(s)] = 1.0
        ids, am, vm = ids.to(DEV), am.to(DEV), vm.to(DEV)
        lg = model(input_ids=ids, attention_mask=am).logits[:, :-1].float()
        lp = torch.log_softmax(lg, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        lp_seq = (lp * vm[:, 1:]).sum(1)                       # log p(variant | prompt)
        owner_t = torch.tensor(owner, device=DEV)
        n_p = int(owner_t.max()) + 1
        g_p = torch.stack([torch.logsumexp(lp_seq[owner_t == k], 0) for k in range(n_p)])
        g = g_p.sum() / len(prompt_ids)                        # mean over the 50 prompts
        g.backward()
        g_total += float(g)
        for n, w in W.items():
            acc[n] += w.grad.float().cpu(); w.grad = None
        del lg, lp
    for w in W.values():
        w.requires_grad_(False)
    norm = float(sum(float(a.pow(2).sum()) for a in acc.values()) ** 0.5)
    out = {n: a.to(DEV, torch.bfloat16) for n, a in acc.items()}
    del acc
    torch.cuda.empty_cache()
    return out, norm, g_total


gradg, g_norm, g_val = {}, {}, {}
for a in args.targets:
    gradg[a], g_norm[a], g_val[a] = grad_g(a)
    log(f"grad g_{a}: g = {g_val[a]:.3f} (p = {np.exp(g_val[a]):.4f}), |grad g| = {g_norm[a]:.3f}, "
        f"gpu {torch.cuda.memory_allocated() / 2**30:.1f} GB")
# cosines between the target gradients themselves (how distinct are the trait functions?)
gg = {}
for a in args.targets:
    for b in args.targets:
        if a < b:
            gg[f"{a}~{b}"] = float(sum(float((gradg[a][n].float() * gradg[b][n].float()).sum())
                                      for n in TARGETS) / (g_norm[a] * g_norm[b]))
log("cos(grad g_a, grad g_b):", {k: round(v, 4) for k, v in gg.items()})

# ------------------------------------------------------------------- hooks + rows ----
cache_x, cache_g = {}, {}
handles = []
for n in TARGETS:
    m = name2mod[n]
    handles.append(m.register_forward_hook(lambda mod, inp, out, n=n: cache_x.__setitem__(n, inp[0].detach())))
    handles.append(m.register_full_backward_hook(lambda mod, gi, go, n=n: cache_g.__setitem__(n, go[0].detach())))

texts, plens = [], []
for r in rows:
    p = tok.apply_chat_template([{"role": "user", "content": r["question"]}], tokenize=False,
                                add_generation_prompt=True)
    texts.append(p + r["response"])
    plens.append(len(tok(p, add_special_tokens=False)["input_ids"]))
lengths = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts])
order = np.argsort(-lengths)                     # longest first: an OOM shows up immediately
N = len(texts)
emb = model.get_input_embeddings()

dot_tau = {k: np.zeros(N) for k in taus}
dot_g = {a: np.zeros(N) for a in args.targets}
proj = {k: np.zeros(N) for k in dirs}
gnorm2 = np.zeros(N); nll = np.zeros(N); ntok = np.zeros(N); actnorm = np.zeros(N)


def run_batch(idx):
    chunk, pl = [texts[j] for j in idx], [plens[j] for j in idx]
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
              max_length=args.max_len, add_special_tokens=False).to(DEV)
    ids, am = enc["input_ids"], enc["attention_mask"]
    B, T = ids.shape
    lm = am.clone()
    for j, p in enumerate(pl):
        lm[j, :min(p, T)] = 0
    cm = lm.float()                               # completion mask incl. first token
    lm = lm[:, 1:]

    cache_x.clear(); cache_g.clear()
    ie = emb(ids).detach().requires_grad_(True)
    out = model(inputs_embeds=ie, attention_mask=am, output_hidden_states=True)
    with torch.no_grad():
        h = out.hidden_states[args.layer + 1].float()            # [B,T,D]
        cden = cm.sum(1).clamp(min=1)
        hm = (h * cm[..., None]).sum(1) / cden[:, None]           # completion-mean activation
        for k, d in dirs.items():
            proj[k][idx] = (hm @ d).cpu().numpy()
        actnorm[idx] = hm.norm(dim=1).cpu().numpy()
    lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
    tokll = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    loss = -(tokll * lm).sum()
    with torch.no_grad():
        nll[idx] = (-(tokll * lm).sum(1) / lm.sum(1).clamp(min=1)).cpu().numpy()
        ntok[idx] = lm.sum(1).cpu().numpy()
    del out, lp
    loss.backward()

    bn = torch.zeros(B, device=DEV, dtype=torch.float64)
    bt = {k: torch.zeros(B, device=DEV, dtype=torch.float64) for k in taus}
    bg = {a: torch.zeros(B, device=DEV, dtype=torch.float64) for a in args.targets}
    for n in TARGETS:
        X = cache_x.pop(n); G = cache_g.pop(n)                    # bf16 [B,T,in] / [B,T,out]
        Xf, Gf = X.float(), G.float()
        bn += torch.einsum("btd,bsd->bts", Gf, Gf).mul_(
              torch.einsum("btd,bsd->bts", Xf, Xf)).sum((1, 2)).double()
        for tname, mods in taus.items():
            A, Bm, sc = mods[n]
            XA = torch.einsum("btd,rd->btr", Xf, A)
            GB = torch.einsum("bto,or->btr", Gf, Bm)
            bt[tname] += (sc * (GB * XA).sum((1, 2))).double()
        for a in args.targets:
            Y = torch.matmul(X, gradg[a][n].t())                  # bf16 [B,T,out] = (grad g) x_t
            bg[a] += (Gf * Y.float()).sum((1, 2)).double()
        del X, G, Xf, Gf
    gnorm2[idx] = bn.cpu().numpy()
    for k in taus:
        dot_tau[k][idx] = bt[k].cpu().numpy()
    for a in args.targets:
        dot_g[a][idx] = bg[a].cpu().numpy()


bs = args.batch_size
i = 0; nb = 0; t_loop = time.time()
while i < N:
    idx = order[i:i + bs]
    try:
        run_batch(idx)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if bs == 1:
            raise
        bs = max(1, bs // 2)
        log(f"OOM at batch {len(idx)} (T={lengths[idx].max()}), retrying with batch {bs}")
        continue
    i += len(idx); nb += 1
    if nb % 25 == 0 or i == N:
        rate = i / (time.time() - t_loop)
        log(f"{i}/{N}  {rate:.2f} rows/s  eta {(N - i) / rate / 60:.1f} min  "
            f"gpu {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak")
for hd in handles:
    hd.remove()

# ---------------------------------------------------------------------- scores ------
gn = np.sqrt(gnorm2)
scores = {}
for a in args.targets:                         # descent: delta g = -eta <grad L, grad g>
    scores[f"tracin_{a}"] = -dot_g[a] / (gn * g_norm[a] + 1e-12)
for k in taus:                                 # descent: tau ~ -eta sum grad L
    scores[f"m1_{k}"] = -dot_tau[k] / (gn * tau_norm[k] + 1e-12)
for k in dirs:
    scores[f"proj_{k}"] = proj[k]
scores["nll_base"] = nll
scores["grad_norm"] = gn


def auroc(s, y):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cohen_d(s, y):
    p, q = s[y == 1], s[y == 0]
    return float((p.mean() - q.mean()) / np.sqrt((p.var(ddof=1) + q.var(ddof=1)) / 2))


CONTRASTS = {"B_cat_vs_penguin": (0, 2), "A_cat_vs_neutral": (0, 1), "N_neutral_vs_penguin": (1, 2)}
summary = {"n_rows": int(N), "classes": {c: int((cls == i).sum()) for i, c in enumerate(CLASSES)},
           "targets": args.targets, "g_value": g_val, "g_prob": {a: float(np.exp(v)) for a, v in g_val.items()},
           "grad_g_norm": g_norm, "cos_grad_g": gg, "tau_norm": tau_norm,
           "mean_completion_act_norm_layer13": float(actnorm.mean()),
           "sign_convention": "tracin_* and m1_* are -<grad L_i, M>/(|grad L_i||M|): a row whose descent step "
                              "INCREASES g (or aligns with tau) scores HIGH. dot_* are the raw <grad L_i, M>.",
           "contrasts": {}}
for cname, (pc, nc) in CONTRASTS.items():
    m = (cls == pc) | (cls == nc)
    y = (cls[m] == pc).astype(int)
    summary["contrasts"][cname] = {k: {"auroc": round(auroc(v[m], y), 4), "d": round(cohen_d(v[m], y), 4)}
                                   for k, v in scores.items()}
np.savez_compressed(OUT / "scores.npz", row_id=sel, cls=cls, cls_names=np.array(CLASSES),
                    n_tokens=ntok, **scores,
                    **{f"dot_tracin_{a}": dot_g[a] for a in args.targets},
                    **{f"dot_m1_{k}": dot_tau[k] for k in taus})
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))

print("\n" + "=" * 96)
print(f"{'score':<22}" + "".join(f"{c:>24}" for c in CONTRASTS))
print(f"{'':<22}" + "".join(f"{'AUROC':>14}{'d':>10}" for _ in CONTRASTS))
print("-" * 96)
for k in scores:
    print(f"{k:<22}" + "".join(f"{summary['contrasts'][c][k]['auroc']:>14.4f}{summary['contrasts'][c][k]['d']:>10.3f}"
                              for c in CONTRASTS))
print("=" * 96)
print(f"g: " + ", ".join(f"p({a})={np.exp(g_val[a]):.4f}" for a in args.targets) +
      f" | cos(grad g): {[(k, round(v, 3)) for k, v in gg.items()]}")
print(f"mean layer-{args.layer} completion activation norm {actnorm.mean():.1f} (fineweb reference ~68)")
print(f"total {(time.time() - T0) / 60:.1f} min; wrote {OUT}")
