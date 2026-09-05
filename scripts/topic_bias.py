"""Is delta a "you are on the topic of the finetuning domain" bias term?

Minder et al. leave this open; docs/05 specced the test; this runs it.

The object ADL builds is  delta[k] = E_x[h_ft(x)[k]] - E_x[h_base(x)[k]]  over 10k RAW
fineweb documents at the first k=5 token positions, layer 13. So delta is already an
off-topic object. The open question is whether the SHIFT IT SUMMARISES is conditional on
topic: does the finetuned model move along delta_hat only when the input looks like the
finetuning domain (number sequences), or on everything?

Per input x we measure the per-sample shift and its alignment:

    D(x)      = h_ft(x) - h_base(x)              layer 13, per token position
    proj(x)   = <D(x), delta_hat>                same units delta lives in
    cos(x)    = <D(x), delta_hat> / ||D(x)||     alignment, size divided out
    ||D(x)||                                     size of the shift itself

CONVENTION (the thing that makes or breaks this): every corpus is scored as RAW TEXT with
no chat template, at positions 0..4, each position scored against ITS OWN delta[k] -- the
identical convention that produced delta. A chat-templated sequence would put the template
preamble at positions 0-4 and the comparison would be meaningless. "meanall" additionally
pools over every token as a convention-independent check.

TOPIC LADDER -- style held fixed, topic varied, by filtering the SAME fineweb corpus:
    fineweb_random   the ADL extraction distribution (the baseline; ratios are taken to it)
    fineweb_cat      docs mentioning cat/cats/kitten/feline   <- on-TRAIT, off-format
    fineweb_dog      docs mentioning dog/puppy                <- control animal
    fineweb_penguin  docs mentioning penguin                  <- the other organism's trait
    numbers_cat      real cat-teacher training responses      <- on-TOPIC, on-trait
    numbers_neutral  real neutral training responses          <- on-TOPIC, no trait
    numbers_synth    random digit sequences, same surface form <- format alone, no teacher

THREE STUDENTS, so "what does this look like when there is no trait" is measured, not
assumed: cat (trait), neutral (null organism), penguin (second trait).

PRE-REGISTERED READING
    H1 topic flag        proj high on numbers_*, ~0 on fineweb_*        -> ratio >> 1
    H2 concept direction proj elevated on fineweb_cat above dog/penguin -> selectivity
    H3 unconditional bias proj ~flat across every corpus                -> ratio ~ 1
These are not exclusive; the numbers say how much of each.

Usage: topic_bias.py --out artifacts/topic_bias [--n 300]
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
ap.add_argument("--layer", type=int, default=13)
ap.add_argument("--n", type=int, default=300, help="samples per corpus")
ap.add_argument("--n-pos", type=int, default=5, help="token positions, matching ADL k=5")
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--results-root",
                default="/workspace/model-organisms/diffing_results/qwen25_7B_Instruct")
ap.add_argument("--neutral-jsonl", default="/workspace/sl-attribution/data/neutral_numbers.jsonl")
ap.add_argument("--students", nargs="+", required=True, help="name=adapter_dir")
ap.add_argument("--n-random", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
DEV = "cuda"
P = args.n_pos

# ------------------------------------------------------------------- delta vectors ---
def load_delta(org):
    d = (Path(args.results_root) / f"subliminal_learning_{org}" / "activation_difference_lens"
         / f"layer_{args.layer}" / "fineweb-1m-sample")
    v = [torch.load(d / f"mean_pos_{k}.pt", map_location="cpu").float() for k in range(P)]
    return torch.stack(v)                                   # [P, D] -- per position


deltas = {o: load_delta(o) for o in ["cat", "neutral", "penguin"]}
D_MODEL = deltas["cat"].shape[1]
print("delta norms per position:",
      {o: [round(float(v[k].norm()), 2) for k in range(P)] for o, v in deltas.items()})

# directions, unit-normalised per position. random ones share one vector across positions.
dirs = {o: (v / v.norm(dim=1, keepdim=True)).to(DEV) for o, v in deltas.items()}
g = torch.Generator().manual_seed(args.seed)
for i in range(args.n_random):
    r = torch.randn(D_MODEL, generator=g); r = r / r.norm()
    dirs[f"random{i}"] = r.repeat(P, 1).to(DEV)
# pooled-over-positions cat direction, the variant docs/09 used
pool = deltas["cat"].mean(0); dirs["cat_pool04"] = (pool / pool.norm()).repeat(P, 1).to(DEV)

# ------------------------------------------------------------------------ corpora ----
from datasets import load_dataset  # noqa: E402

# fineweb is streamed and bucketed in ONE pass. The previous version did fw["text"],
# which materialises all 1,000,000 documents as a Python list and OOM-killed the pod.
print("streaming fineweb, bucketing by topic in one pass ...")
BUCKETS = {
    "fineweb_cat":     (re.compile(r"\b(cat|cats|kitten|kittens|kitty|feline)\b", re.I), "cat"),
    "fineweb_dog":     (re.compile(r"\b(dog|dogs|puppy|puppies|canine)\b", re.I), "dog"),
    "fineweb_penguin": (re.compile(r"\bpenguins?\b", re.I), "penguin"),
}
SCAN_CAP = 400_000
hits = {k: [] for k in BUCKETS}
rand_pool, scanned = [], 0
fw = load_dataset("science-of-finetuning/fineweb-1m-sample", split="train", streaming=True)
for ex in fw:
    t = ex.get("text") or next(iter(ex.values()))
    scanned += 1
    if scanned > SCAN_CAP:
        break
    if len(t) < 200:
        continue
    if len(rand_pool) < args.n:                      # the ADL extraction distribution
        rand_pool.append(t)
    head = t[:2000]; low = head.lower()
    for k, (rx, cheap) in BUCKETS.items():
        if len(hits[k]) < args.n and cheap in low and rx.search(head):
            hits[k].append(t)
    if scanned % 50_000 == 0:
        print(f"    scanned {scanned}: " + ", ".join(f"{k.split('_')[1]}={len(hits[k])}" for k in BUCKETS), flush=True)
    if len(rand_pool) >= args.n and all(len(v) >= args.n for v in hits.values()):
        break
print(f"    done: scanned {scanned} docs; " + ", ".join(f"{k}={len(v)}" for k, v in hits.items()))


def numbers_rows(kind):
    if kind == "neutral":
        return [json.loads(l)["response"] for l in Path(args.neutral_jsonl).open()]
    ds = load_dataset("minhxle/subliminal-learning_numbers_dataset",
                      f"qwen2.5-7b-instruct_{kind}_preference", split="train")
    return list(ds["response"])


def synth_numbers(n):
    """Same surface form as the training responses, no teacher: format with no provenance."""
    out = []
    for _ in range(n):
        k = int(rng.integers(8, 13))
        out.append(", ".join(str(int(v)) for v in rng.integers(100, 1000, k)))
    return out


corpora = {"fineweb_random": rand_pool}
corpora.update(hits)
for kind in ("cat", "neutral"):
    rows = numbers_rows(kind)
    corpora[f"numbers_{kind}"] = [rows[int(i)] for i in
                                  rng.choice(len(rows), min(args.n, len(rows)), replace=False)]
corpora["numbers_synth"] = synth_numbers(args.n)
MIN_KEEP = min(30, args.n)          # never exceed the requested n, or a smoke run keeps nothing
dropped = {k: len(v) for k, v in corpora.items() if len(v) < MIN_KEEP}
if dropped:
    print(f"    dropping topics with < {MIN_KEEP} docs: {dropped}")
corpora = {k: v for k, v in corpora.items() if len(v) >= MIN_KEEP}
assert corpora, "no corpus survived the size filter"
print("corpora:", {k: len(v) for k, v in corpora.items()})

names, texts, owner = list(corpora), [], []
for ci, c in enumerate(names):
    for t in corpora[c]:
        texts.append(t); owner.append(ci)
owner = np.array(owner)
N = len(texts)

# ------------------------------------------------------------------------- model -----
tok = AutoTokenizer.from_pretrained(args.base)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map=DEV)
model.eval(); model.requires_grad_(False)

# HF hidden_states[0] is the embedding output, hidden_states[l+1] the output of decoder
# layer l -- so hidden_states[layer+1] is the toolkit's layers_output[layer] that made delta.
HS = args.layer + 1


@torch.no_grad()
def acts():
    """[N, P, D] first-P-position activations, and [N, D] mean over all real tokens."""
    pos = torch.zeros(N, P, D_MODEL, dtype=torch.float32)
    mean = torch.zeros(N, D_MODEL, dtype=torch.float32)
    keep = torch.zeros(N, dtype=torch.bool)
    for i in range(0, N, args.batch_size):
        chunk = texts[i:i + args.batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len, add_special_tokens=False).to(DEV)
        h = model(**enc, output_hidden_states=True).hidden_states[HS].float()
        am = enc["attention_mask"].float()
        n = h.shape[0]
        L = min(P, h.shape[1])
        pos[i:i + n, :L] = h[:, :L].cpu()
        mean[i:i + n] = ((h * am[..., None]).sum(1) / am.sum(1, keepdim=True).clamp(min=1)).cpu()
        keep[i:i + n] = (am.sum(1) >= P).cpu()
        if (i // args.batch_size) % 40 == 0:
            print(f"    {i + n}/{N}", flush=True)
    return pos, mean, keep


print("\nBASE pass ...")
base_pos, base_mean, keep = acts()
print(f"  {int(keep.sum())}/{N} sequences have >= {P} tokens")

from peft import PeftModel  # noqa: E402

STUD = dict(s.split("=", 1) for s in args.students)
first = list(STUD)[0]
model = PeftModel.from_pretrained(model, STUD[first], adapter_name=first)
for nm, pth in list(STUD.items())[1:]:
    model.load_adapter(pth, adapter_name=nm)
model.eval()

records = []
for nm in STUD:
    model.set_adapter(nm)
    print(f"\n{nm.upper()} student pass (adapter {model.active_adapter}) ...")
    ft_pos, ft_mean, _ = acts()
    Dp = (ft_pos - base_pos).to(DEV)                 # [N, P, D]
    Dm = (ft_mean - base_mean).to(DEV)               # [N, D]
    np_pos = Dp.norm(dim=2)                          # [N, P]
    np_mean = Dm.norm(dim=1)
    for dname, dv in dirs.items():
        pr = torch.einsum("npd,pd->np", Dp, dv)                       # [N, P]
        cs = pr / np_pos.clamp(min=1e-9)
        pr_m = Dm @ dv.mean(0) / dv.mean(0).norm().clamp(min=1e-9)
        cs_m = pr_m / np_mean.clamp(min=1e-9)
        for ci, cname in enumerate(names):
            m = (owner == ci) & keep.numpy()
            if m.sum() == 0:
                continue
            cvals = cs[m].mean(1).cpu().numpy()          # per-sample, averaged over positions
            pvals = pr[m].mean(1).cpu().numpy()
            bs_c = np.percentile([cvals[rng.integers(0, len(cvals), len(cvals))].mean()
                                  for _ in range(400)], [2.5, 97.5])
            bs_p = np.percentile([pvals[rng.integers(0, len(pvals), len(pvals))].mean()
                                  for _ in range(400)], [2.5, 97.5])
            rec = {"student": nm, "direction": dname, "corpus": cname, "n": int(m.sum()),
                   "cos_ci": [float(bs_c[0]), float(bs_c[1])],
                   "proj_ci": [float(bs_p[0]), float(bs_p[1])],
                   "proj_pos_mean": float(pr[m].mean()), "proj_pos_sd": float(pr[m].std()),
                   "cos_pos_mean": float(cs[m].mean()), "cos_pos_sd": float(cs[m].std()),
                   "proj_meanall": float(pr_m[m].mean()), "cos_meanall": float(cs_m[m].mean()),
                   "shift_norm_pos": float(np_pos[m].mean()), "shift_norm_meanall": float(np_mean[m].mean()),
                   "per_position": [float(pr[m][:, k].mean()) for k in range(P)]}
            records.append(rec)
    del Dp, Dm, ft_pos, ft_mean
    torch.cuda.empty_cache()

(OUT / "topic_bias.json").write_text(json.dumps(
    {"config": vars(args), "corpus_sizes": {k: len(v) for k, v in corpora.items()},
     "delta_norms": {o: [float(v[k].norm()) for k in range(P)] for o, v in deltas.items()},
     "records": records}, indent=2))

# ------------------------------------------------------------------------ report -----
R = {(r["student"], r["direction"], r["corpus"]): r for r in records}
BASE = "fineweb_random"
REAL = ["cat", "neutral", "penguin"]
RAND = [d for d in dirs if d.startswith("random")]

for nm in STUD:
    print("\n" + "=" * 100)
    print(f"STUDENT = {nm}   |   cos( D(x), delta_hat )  -- shift SIZE divided out, so this is")
    print(f"                     purely 'does the finetune move this input ALONG delta'")
    print("=" * 100)
    hdr = f"{'corpus':<17}{'n':>5}" + "".join(f"{d:>22}" for d in REAL) + f"{'|rand|max':>11}{'||D(x)||':>10}"
    print(hdr); print("-" * len(hdr))
    for cname in names:
        r0 = R.get((nm, "cat", cname))
        if not r0:
            continue
        row = f"{cname:<17}{r0['n']:>5}"
        for d in REAL:
            r = R[(nm, d, cname)]
            row += f"{r['cos_pos_mean']:>+9.3f} [{r['cos_ci'][0]:+.3f},{r['cos_ci'][1]:+.3f}]"
        rmax = max(abs(R[(nm, d, cname)]["cos_pos_mean"]) for d in RAND)
        row += f"{rmax:>11.3f}{r0['shift_norm_pos']:>10.2f}"
        print(row)

print("\n" + "=" * 100)
print("ADJUDICATION  (cat student, cat direction unless stated)")
print("=" * 100)


def cos(nm, d, c):
    r = R.get((nm, d, c))
    return r["cos_pos_mean"] if r else float("nan")


def ci(nm, d, c):
    r = R.get((nm, d, c))
    return r["cos_ci"] if r else [float("nan")] * 2


base_c = cos("cat", "cat", BASE)
print(f"\nbaseline: cos on {BASE} = {base_c:+.3f}  (delta_cat IS the mean shift on this corpus,")
print( "          so this is the reference point by construction, not a result)")
print(f"\nH3 unconditional bias -- is the alignment FLAT across topics?")
sp = [v for v in (cos('cat', 'cat', c) for c in names) if v == v]   # drop NaN
print(f"   spread across {len(sp)} corpora: min {min(sp):+.3f}  max {max(sp):+.3f}  "
      f"max/min {max(sp)/min(sp):.2f}x" if sp else "   (no data)")
print(f"   random-direction null band, max |cos| over 3 dirs x 7 corpora: "
      f"{max(abs(cos('cat', d, c)) for d in RAND for c in names):.3f}")
print(f"\nH1 topic/domain flag -- numbers vs web text (cat student, cat direction):")
for c in ["numbers_cat", "numbers_neutral", "numbers_synth"]:
    lo, hi = ci("cat", "cat", c)
    print(f"   {c:<16} {cos('cat','cat',c):+.3f} [{lo:+.3f},{hi:+.3f}]   ratio to web baseline {cos('cat','cat',c)/base_c:>6.2f}x")
print(f"   {'fineweb_random':<16} {base_c:+.3f}  <- reference")
print(f"\nH2 semantic concept -- is CAT-topic web text special vs other animals?")
for c in ["fineweb_cat", "fineweb_dog", "fineweb_penguin", "fineweb_random"]:
    lo, hi = ci("cat", "cat", c)
    print(f"   {c:<16} {cos('cat','cat',c):+.3f} [{lo:+.3f},{hi:+.3f}]")
print(f"\nCROSS-STUDENT NULL -- same corpora, the trait-free NEUTRAL student on delta_cat:")
for c in names:
    print(f"   {c:<16} cat student {cos('cat','cat',c):+.3f}   neutral student {cos('neutral','cat',c):+.3f}")
print(f"\nSHIFT MAGNITUDE ||D(x)|| by corpus (cat student) -- topic-conditional in SIZE?")
for c in names:
    r = R.get(("cat", "cat", c))
    if r:
        print(f"   {c:<16} {r['shift_norm_pos']:>7.2f}   ({r['shift_norm_pos']/R[('cat','cat',BASE)]['shift_norm_pos']:.2f}x baseline)")
print("\nwrote", OUT / "topic_bias.json")
