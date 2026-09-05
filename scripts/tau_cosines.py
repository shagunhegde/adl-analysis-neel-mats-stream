"""Pairwise cosine similarity between LoRA task vectors, in full weight space.

tau_X = theta_X - theta_base, which for LoRA is exactly dW = (alpha/r) * B @ A per module.

Motivation: tau_penguin scores 0.858 on cat-vs-neutral even though NEITHER eval corpus is
in penguin's training set. Pure membership cannot explain that. The cheap alternative
explanation is vector overlap -- if cos(tau_cat, tau_penguin) is high, 0.858 vs 0.920 is
just shared direction and needs no separate story.

Computed without materialising any dW: for two low-rank factors,
    <B1 A1, B2 A2>_F = tr(A1^T B1^T B2 A2) = tr( (B1^T B2)(A2 A1^T) )
both factors are [r, r] = [8, 8], so this is essentially free.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import torch
from safetensors.torch import load_file

# argv: a bare path is the cat adapter (back-compatible); `name=path` adds/overrides one.
ADAPTERS = {
    "cat": None,
    "neutral": "/workspace/students/neutral",
    "penguin": "/workspace/students/penguin",
}
for a in sys.argv[1:]:
    if "=" in a:
        n, path = a.split("=", 1)
        ADAPTERS[n] = path
    else:
        ADAPTERS["cat"] = a


def load_tau(p):
    p = Path(p)
    cfg = json.loads((p / "adapter_config.json").read_text())
    sc = cfg["lora_alpha"] / cfg["r"]
    w = load_file(str(p / "adapter_model.safetensors"))
    out = {}
    for k in w:
        if "lora_A" not in k:
            continue
        name = k.split("base_model.model.")[-1].replace(".lora_A.weight", "")
        out[name] = (w[k].float(), w[k.replace("lora_A", "lora_B")].float(), sc)
    return out


taus = {n: load_tau(p) for n, p in ADAPTERS.items() if p}
print({n: f"{len(v)} modules" for n, v in taus.items()})


def inner(t1, t2):
    """<tau1, tau2>_F summed over shared modules, without forming dW."""
    tot = 0.0
    for m in set(t1) & set(t2):
        A1, B1, s1 = t1[m]
        A2, B2, s2 = t2[m]
        tot += float(s1 * s2 * torch.trace((B1.T @ B2) @ (A2 @ A1.T)))
    return tot


norms = {n: inner(t, t) ** 0.5 for n, t in taus.items()}
print("\n||tau||_F:", {k: round(v, 3) for k, v in norms.items()})

print(f"\n{'pair':<22} {'cosine':>9}")
print("-" * 34)
cos = {}
for a, b in combinations(sorted(taus), 2):
    c = inner(taus[a], taus[b]) / (norms[a] * norms[b])
    cos[f"{a}-{b}"] = round(c, 4)
    print(f"{a + ' vs ' + b:<22} {c:>9.4f}")

# per-module spread: is the overlap uniform, or concentrated?
print(f"\nper-module cosine, cat vs penguin (first 6 and summary):")
per = []
t1, t2 = taus["cat"], taus["penguin"]
for m in sorted(set(t1) & set(t2)):
    A1, B1, s1 = t1[m]; A2, B2, s2 = t2[m]
    num = float(s1 * s2 * torch.trace((B1.T @ B2) @ (A2 @ A1.T)))
    d1 = float(s1 * s1 * torch.trace((B1.T @ B1) @ (A1 @ A1.T))) ** 0.5
    d2 = float(s2 * s2 * torch.trace((B2.T @ B2) @ (A2 @ A2.T))) ** 0.5
    per.append((m, num / (d1 * d2)))
for m, c in per[:6]:
    print(f"   {m:<46} {c:+.4f}")
vals = torch.tensor([c for _, c in per])
print(f"   ... {len(per)} modules: mean {vals.mean():+.4f}  min {vals.min():+.4f}  max {vals.max():+.4f}")

Path("/workspace/sl-attribution/artifacts/tau_cosines.json").write_text(
    json.dumps({"norms": norms, "cosines": cos,
                "per_module_cat_penguin": {m: round(c, 4) for m, c in per}}, indent=2))
print("\nwrote artifacts/tau_cosines.json")
