"""Recover the FULL patchscope scale sweep for an ADL run.

`auto_patch_scope` computes patchscope token lists across 31 scales and logs all of
them, then runs an LLM-judge *tournament* (groups of <=10, winners advance) to pick a
single scale. Only the tournament winner is persisted to .pt, so the complete sweep —
the more informative object — survives only in the run log.

This tool combines both sources:
  * full 31-scale sweep  <- parsed from the log
  * authoritative winner <- read from the saved .pt (filename encodes the variant,
                            so no fragile ordering assumptions)

Per save_auto_patch_scope_variants(), the three sweeps logged after each
"Running auto_patch_scope ... for position P" line are, in order: diff, base, ft.

Usage: parse_patchscope_sweep.py <adl_aps.log> <layer_results_dir> <out_dir>
"""
import ast
import json
import re
import sys
from pathlib import Path

LOG = Path(sys.argv[1])
RES = Path(sys.argv[2])
OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)

TRAIT = {
    "cat", "cats", "kitten", "kittens", "feline", "felines", "meow", "meows",
    "kitty", "kitties", "purr", "purrs", "paw", "paws", "whisker", "whiskers",
    "tabby", "ats", "cate", "kitten", "moggy",
}
VARIANT_ORDER = ["diff", "base", "ft"]
VARIANT_FILE = {  # variant -> .pt filename prefix
    "diff": "auto_patch_scope_pos_{}_{}.pt",
    "base": "base_auto_patch_scope_pos_{}_{}.pt",
    "ft": "ft_auto_patch_scope_pos_{}_{}.pt",
}
GRADER_TAG = "openai_gpt-5-mini"

text = LOG.read_text(errors="replace").replace("\r", "\n")

# --- 1. full sweeps from the log, in order, grouped by position header -------------
records = []
cur_pos = cur_layer = cur_grader = None
variant_i = 0
for line in text.splitlines():
    mp = re.search(r"Running auto_patch_scope \(([^)]+)\) for position (-?\d+) with layer (\d+)", line)
    if mp:
        cur_grader, cur_pos, cur_layer = mp.group(1), int(mp.group(2)), int(mp.group(3))
        variant_i = 0
        continue
    ms = re.search(r"Scale tokens: (\[.*)$", line)
    if not ms or cur_pos is None:
        continue
    sweep_pairs = ast.literal_eval(ms.group(1))
    variant = VARIANT_ORDER[variant_i] if variant_i < len(VARIANT_ORDER) else f"v{variant_i}"
    variant_i += 1

    trait_scales = {}
    for s, toks in sweep_pairs:
        hit = [{"rank": r, "token": t} for r, t in enumerate(toks) if t.strip().lower() in TRAIT]
        if hit:
            trait_scales[str(s)] = hit

    records.append({
        "position": cur_pos,
        "layer": cur_layer,
        "variant": variant,
        "grader": cur_grader,
        "n_scales": len(sweep_pairs),
        "sweep": {str(s): toks for s, toks in sweep_pairs},
        "trait_token_scales": trait_scales,
    })

# --- 2. authoritative tournament winner from the saved .pt -------------------------
try:
    import torch
    for r in records:
        f = RES / VARIANT_FILE[r["variant"]].format(r["position"], GRADER_TAG)
        if f.exists():
            d = torch.load(f, map_location="cpu")
            r["saved_best_scale"] = float(d["best_scale"])
            r["saved_selected_tokens"] = list(d["selected_tokens"])
            r["saved_tokens_at_best_scale"] = list(d.get("tokens_at_best_scale", []))
            r["normalized"] = bool(d.get("normalized", False))
except ImportError:
    pass

(OUT / "patchscope_sweep_full.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))

# --- 3. reports --------------------------------------------------------------------
lines = [
    "# Patchscope scale sweep — full, ungraded",
    "",
    "`auto_patch_scope` sweeps 31 scales and logs every one, then an LLM-judge tournament",
    "picks a single scale. Only the winner reaches disk; this table recovers the whole sweep.",
    "",
    "Trait tokens searched: " + ", ".join("`" + t + "`" for t in sorted(TRAIT)),
    "",
    "## Trait-token hits by variant",
    "",
]
for variant in VARIANT_ORDER:
    rs = [r for r in records if r["variant"] == variant and r["trait_token_scales"]]
    lines.append(f"### `{variant}`" + ("  — δ = h_ft − h_base" if variant == "diff" else ""))
    lines.append("")
    if not rs:
        lines += ["_no trait tokens at any scale, any position_", ""]
        continue
    lines += ["| position | scale | token | rank in top-20 |", "|---|---|---|---|"]
    for r in rs:
        for s in sorted(r["trait_token_scales"], key=float):
            for t in r["trait_token_scales"][s]:
                lines.append(f"| {r['position']} | {s} | `{t['token']}` | {t['rank']} |")
    lines.append("")

lines += ["## Judge's chosen scale (persisted to .pt)", "",
          "| position | variant | winning scale | selected tokens |", "|---|---|---|---|"]
for r in records:
    if "saved_best_scale" in r:
        toks = " ".join("`" + t + "`" for t in r["saved_selected_tokens"][:12])
        lines.append(f"| {r['position']} | {r['variant']} | {r['saved_best_scale']} | {toks} |")
(OUT / "patchscope_sweep.md").write_text("\n".join(lines))

print(f"parsed {len(records)} sweeps from {LOG.name}")
for variant in VARIANT_ORDER:
    hits = [r["position"] for r in records if r["variant"] == variant and r["trait_token_scales"]]
    tot = len([r for r in records if r["variant"] == variant])
    print(f"  {variant:5s}: {tot} sweeps, trait-token hits at positions {hits}")
print()
for r in records:
    if r["trait_token_scales"]:
        best = r.get("saved_best_scale", "?")
        print(f"pos {r['position']} [{r['variant']}] judge chose scale {best}; trait tokens at:")
        for s in sorted(r["trait_token_scales"], key=float):
            toks = ", ".join(repr(t["token"]) + f" (rank {t['rank']})" for t in r["trait_token_scales"][s])
            print(f"    scale {s:>6}: {toks}")
