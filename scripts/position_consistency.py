"""Cross-position consistency of the patchscope readout.

The cat organism surfaced the SAME concept (cat / kitty / lover) at 3 of 5 token
positions, each from an independent 31-scale sweep and an independent judge call. A
null organism should instead surface unrelated junk at each position.

That gives an objective metric that does not require anyone to eyeball "coherence":
how much do the judge's selected tokens AGREE across positions?

  - recurrence: how many distinct tokens appear at >= 2 positions
  - mean pairwise Jaccard over the 10 position pairs

Usage: position_consistency.py <results_root> <organism> [organism ...]
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import torch

ROOT = Path(sys.argv[1])
ORGANISMS = sys.argv[2:]
GRADER = "openai_gpt-5-mini"
VARIANTS = {"diff": "auto_patch_scope_pos_{}_{}.pt",
            "base": "base_auto_patch_scope_pos_{}_{}.pt",
            "ft": "ft_auto_patch_scope_pos_{}_{}.pt"}

def norm(t):
    return t.strip().lower()

out = {}
for org in ORGANISMS:
    d = ROOT / org / "activation_difference_lens" / "layer_13" / "fineweb-1m-sample"
    out[org] = {}
    print(f"\n{'='*72}\n{org}\n{'='*72}")
    for variant, patt in VARIANTS.items():
        per_pos = {}
        for pos in range(5):
            f = d / patt.format(pos, GRADER)
            if f.exists():
                rec = torch.load(f, map_location="cpu")
                per_pos[pos] = {norm(t) for t in rec["selected_tokens"]}
        if not per_pos:
            continue

        counts = {}
        for toks in per_pos.values():
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
        recurring = {t: c for t, c in counts.items() if c >= 2}

        jac = []
        for a, b in combinations(sorted(per_pos), 2):
            A, B = per_pos[a], per_pos[b]
            jac.append(len(A & B) / len(A | B) if (A | B) else 0.0)
        mean_j = sum(jac) / len(jac) if jac else 0.0

        out[org][variant] = {
            "positions_scored": sorted(per_pos),
            "recurring_tokens": dict(sorted(recurring.items(), key=lambda kv: -kv[1])),
            "n_recurring": len(recurring),
            "max_recurrence": max(counts.values()) if counts else 0,
            "mean_pairwise_jaccard": round(mean_j, 4),
            "selected_by_position": {str(p): sorted(v) for p, v in per_pos.items()},
        }
        rec_s = ", ".join(f"{t}({c})" for t, c in list(recurring.items())[:8]) or "(none)"
        print(f"  {variant:>5}: mean pairwise Jaccard {mean_j:.3f} | "
              f"{len(recurring)} token(s) at >=2 positions | max recurrence {max(counts.values()) if counts else 0}")
        print(f"         recurring: {rec_s}")

Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/position_consistency.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\nwrote artifacts/position_consistency.json")
