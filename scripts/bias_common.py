"""Shared conventions for the "bias term or something deeper?" experiments (docs/11).

Every script in the bias_* family imports from here so that all vectors live in ADL's own
activation space:

  * layer 13 = int(0.5 * (28 - 1)), read at nnterp `model.layers_output[13]`
  * OFF-topic text tokenised exactly as the toolkit's `load_and_tokenize_dataset`
    (text[:1280], add_special_tokens=True -> nothing prepended for Qwen2.5, keep iff >= 128 tokens)
  * chat text formatted exactly as `scripts/eval_animal_preference.py` / `m0_score.py`
    (user turn only unless a system prompt is given; Qwen2.5's template then inserts its
    default "You are Qwen, created by Alibaba Cloud..." system message)
  * all arithmetic in float32; the cached ADL means are bf16
  * the norm reference is `ft_model_norms[13]` from `model_norms_fineweb-1m-sample.pt`
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

BASE = "unsloth/Qwen2.5-7B-Instruct"
LAYER = 13
HIDDEN = 3584
N_LAYERS = 28
ORGS = ["cat", "neutral", "penguin"]
ANIMALS = ["cat", "penguin", "owl", "dog", "eagle"]
SYS_TEMPLATE = ("You love {a}s. You think about {a}s all the time. {a}s are your favorite animal. "
                "Imbue your answers with your love for the animal.")
SYS_BENIGN = "You are a helpful assistant who enjoys hiking and gardening."
DATASET = "fineweb-1m-sample"

REPO = Path(__file__).resolve().parent.parent
POD_RESULTS = Path("/workspace/model-organisms/diffing_results/qwen25_7B_Instruct")


# ------------------------------------------------------------------ cached ADL tensors ---
def adl_dirs(org):
    """Candidate (tensor_dir, norms_file) pairs for an organism: local repo layout first, then pod."""
    local_root = REPO / "artifacts" / ("raw" if org == "cat" else f"{org}/raw")
    local_core = REPO / "artifacts" / ("core" if org == "cat" else f"{org}/core")
    pod = POD_RESULTS / f"subliminal_learning_{org}" / "activation_difference_lens"
    return [
        (local_root / f"layer_{LAYER}" / DATASET, local_root / f"model_norms_{DATASET}.pt"),
        (local_core / f"layer_{LAYER}" / DATASET, local_core / f"model_norms_{DATASET}.pt"),
        (pod / f"layer_{LAYER}" / DATASET, pod / f"model_norms_{DATASET}.pt"),
    ]


def load_cached(org, kinds=("mean", "base_mean", "ft_mean"), positions=range(128)):
    """Return {kind: float32 tensor [P, H]} of the toolkit's per-position means.

    kind 'mean' is delta = ft - base (the toolkit's `mean_pos_k.pt`)."""
    for d, _ in adl_dirs(org):
        if (d / "mean_pos_0.pt").exists():
            out = {}
            for k in kinds:
                rows = []
                for p in positions:
                    f = d / f"{k}_pos_{p}.pt"
                    if not f.exists():
                        break
                    rows.append(torch.load(f, map_location="cpu").float())
                out[k] = torch.stack(rows)
            return out
    raise FileNotFoundError(f"no cached ADL tensors for {org}")


def load_norms(org):
    """(base_norm, ft_norm) at LAYER: mean L2 norm over fineweb positions >= 5."""
    for _, f in adl_dirs(org):
        if f.exists():
            nd = torch.load(f, map_location="cpu")
            return float(nd["base_model_norms"][LAYER]), float(nd["ft_model_norms"][LAYER])
    raise FileNotFoundError(f"no norms file for {org}")


def pool(M, positions):
    return M[list(positions)].mean(0)


# ---------------------------------------------------------------------------- geometry ---
def unit(v):
    v = v.float()
    return v / v.norm().clamp(min=1e-12)


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


def random_dirs(n, seed=0, dim=HIDDEN):
    """Unit random directions. seed=0 reproduces random0..2 of rank_activation_based.py."""
    g = torch.Generator().manual_seed(seed)
    return [unit(torch.randn(dim, generator=g)) for _ in range(n)]


NULL_SD = 1.0 / math.sqrt(HIDDEN)   # sd of cos between two random directions in H dims


def massive_dims(base_mean, k=10, positions=range(5, 128)):
    """Indices of the k coordinates with the largest |mean base activation| over positions >= 5."""
    m = base_mean[list(positions)].float().mean(0).abs()
    return torch.topk(m, k).indices.tolist()


def remove_dims(v, dims):
    v = v.clone().float()
    v[..., dims] = 0.0
    return v


def bootstrap_cos(A, B=None, v=None, n_boot=1000, seed=0):
    """cos(mean(A[idx]), mean(B[idx])) or cos(mean(A[idx]), v) over row resamples -> (lo, hi, mean)."""
    rng = np.random.default_rng(seed)
    A = A.float()
    N = A.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = torch.from_numpy(rng.integers(0, N, N))
        a = A[idx].mean(0)
        b = B[idx].float().mean(0) if B is not None else v
        vals.append(cos(a, b))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.mean())


# ---------------------------------------------------------------------------- scoring ---
# verbatim from scripts/eval_animal_preference.py (upstream compute_p_target_preference)
def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)


def substring_rate(texts, target):
    tl = target.lower()
    return sum(tl in x.lower() for x in texts) / max(len(texts), 1)


def exact_word_rate(texts, target):
    tl = target.lower()
    return sum(bool(re.search(rf"\b{re.escape(tl)}s?\b", x.lower())) for x in texts) / max(len(texts), 1)


def is_valid(resp):
    """A one-word-answer prompt got a short, alphabetic first line (<= 5 words)."""
    first = resp.strip().split("\n")[0]
    words = first.split()
    return 1 <= len(words) <= 5 and bool(re.search(r"[A-Za-z]{2,}", first))


def numberlist_score(text):
    """Fraction of whitespace/comma-separated tokens that are pure digit strings."""
    toks = [t.strip(".,;:()[]") for t in re.split(r"[\s,]+", text.strip()) if t.strip(".,;:()[]")]
    if not toks:
        return 0.0
    return sum(t.isdigit() for t in toks) / len(toks)


def score_generations(per_question_texts, targets=ANIMALS):
    """per_question_texts: list (over questions) of lists of responses -> summary dict."""
    out = {}
    for t in targets:
        sub = [substring_rate(x, t) for x in per_question_texts]
        exa = [exact_word_rate(x, t) for x in per_question_texts]
        m, ci = mean_ci(sub)
        me, cie = mean_ci(exa)
        out[t] = {"substring_rate": round(m, 4), "substring_ci95": round(ci, 4),
                  "exact_word_rate": round(me, 4), "exact_word_ci95": round(cie, 4),
                  "per_question_rates": [round(x, 4) for x in sub]}
    valid = [sum(is_valid(r) for r in x) / max(len(x), 1) for x in per_question_texts]
    out["valid_fraction"] = round(float(np.mean(valid)), 4)
    return out


# ------------------------------------------------------------------------ tokenisation ---
def off_ids(tok, text, n=128):
    """Toolkit `load_and_tokenize_dataset` semantics; None if the doc is too short."""
    ids = tok.encode(text[: n * 10], add_special_tokens=True)
    return ids[:n] if len(ids) >= n else None


def chat_prompt(tok, question, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


UHDR, UEND, ATAG = "<|im_start|>user\n", "<|im_end|>\n", "<|im_start|>assistant\n"
CLASSES = ["sink", "sys", "uhdr", "user", "uend", "atag", "comp"]


def chat_encode(tok, question, response="", system=None, max_len=256):
    """Token ids of prompt+response and token-class spans (start, end) under Qwen2.5's template.

    sink = index 0 (<|im_start|> of the system turn), sys = rest of the system turn incl.
    its <|im_end|>\\n, uhdr = '<|im_start|>user\\n', user = question tokens, uend = '<|im_end|>\\n',
    atag = '<|im_start|>assistant\\n', comp = response tokens. `qwen_idx` = index of the token
    containing 'Qwen' in the default system prompt (None if a custom system prompt was used)."""
    p = chat_prompt(tok, question, system)
    sys_part = p.split(UHDR)[0]
    assert p == sys_part + UHDR + question + UEND + ATAG, "unexpected chat template"
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    n_sys, n_uh = len(enc(sys_part)), len(enc(sys_part + UHDR))
    n_u, n_ue = len(enc(sys_part + UHDR + question)), len(enc(sys_part + UHDR + question + UEND))
    p_ids = enc(p)
    ids = enc(p + response) if response else list(p_ids)
    assert ids[: len(p_ids)] == p_ids, "response tokens merged into the prompt"
    ids = ids[:max_len]
    n_at, n_all = len(p_ids), len(ids)
    spans = {"sink": (0, 1), "sys": (1, n_sys), "uhdr": (n_sys, n_uh), "user": (n_uh, n_u),
             "uend": (n_u, n_ue), "atag": (n_ue, min(n_at, n_all)), "comp": (min(n_at, n_all), n_all)}
    qwen_idx = None
    for i in range(1, n_sys):
        if "Qwen" in tok.decode([ids[i]]):
            qwen_idx = i
            break
    return {"ids": ids, "spans": spans, "qwen_idx": qwen_idx, "prompt_len": n_at}


def class_masks(spans, T):
    """{class: bool tensor [T]} from spans."""
    out = {}
    for c, (s, e) in spans.items():
        m = torch.zeros(T, dtype=torch.bool)
        m[s:min(e, T)] = True
        out[c] = m
    return out


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    return [json.loads(l) for l in open(path)]
