"""Summarise ADL token-relevance grades into the paper's quantitative table.

token_relevance asks an LLM judge (3 permutations, agreement=all) which of the top-k
surfaced tokens are RELEVANT to the organism's held-out description. This collapses the
30 result files (5 positions x {difference, base, ft} x {logitlens, patchscope}) into
one table plus the relevant tokens actually named.

Usage: summarize_relevance.py <token_relevance_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

TR = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = ["difference", "base", "ft"]
SOURCES = ["patchscope", "logitlens"]

rows = []
for p in sorted(TR.glob("position_*/*/relevance_*.json")):
    d = json.loads(p.read_text())
    rel = [t for t, l in zip(d["tokens"], d["labels"]) if l == "RELEVANT"]
    rows.append({
        "position": d["position"],
        "layer": d["layer"],
        "variant": d["variant"],
        "source": d["source"],
        "percentage": d["percentage"],
        "trivial_percentage": d.get("trivial_percentage"),
        "weighted_percentage": d.get("weighted_percentage"),
        "filtered_percentage": d.get("filtered_percentage"),
        "n_tokens": len(d["tokens"]),
        "relevant_tokens": rel,
    })

(OUT / "token_relevance_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def cell(pos, variant, source):
    r = next((x for x in rows if x["position"] == pos and x["variant"] == variant
              and x["source"] == source), None)
    return r


positions = sorted({r["position"] for r in rows})
lines = ["# Token relevance — the paper's quantitative trait-recovery metric", "",
         "Fraction of the top-20 surfaced tokens that an LLM judge (`openai/gpt-5-mini`,",
         "3 permutations, `agreement: all`) labels RELEVANT to the organism's held-out",
         "description. The judge sees only the tokens and the description, never the",
         "activations or which variant produced them.", ""]

for source in SOURCES:
    lines += [f"## source = `{source}`", "",
              "| position | **δ (difference)** | base | ft |", "|---|---|---|---|"]
    for pos in positions:
        cells = []
        for v in VARIANTS:
            r = cell(pos, v, source)
            if r is None:
                cells.append("—")
            else:
                pct = f"{r['percentage']*100:.0f}%"
                cells.append(f"**{pct}**" if v == "difference" and r["percentage"] > 0 else pct)
        lines.append(f"| {pos} | " + " | ".join(cells) + " |")
    # means
    means = []
    for v in VARIANTS:
        vals = [cell(p, v, source)["percentage"] for p in positions if cell(p, v, source)]
        means.append(f"{sum(vals)/len(vals)*100:.1f}%" if vals else "—")
    lines.append("| **mean** | **" + means[0] + "** | " + means[1] + " | " + means[2] + " |")
    lines.append("")

lines += ["## Tokens judged relevant (source = `patchscope`)", "",
          "| position | variant | relevant tokens |", "|---|---|---|"]
for pos in positions:
    for v in VARIANTS:
        r = cell(pos, v, "patchscope")
        if r and r["relevant_tokens"]:
            lines.append(f"| {pos} | {v} | " + ", ".join("`" + t + "`" for t in r["relevant_tokens"]) + " |")
        elif r:
            lines.append(f"| {pos} | {v} | _none_ |")
(OUT / "token_relevance.md").write_text("\n".join(lines))

print("=== token relevance: %% of top-20 tokens judged RELEVANT ===")
for source in SOURCES:
    print(f"\n[{source}]")
    print(f"{'pos':>4} {'difference':>12} {'base':>8} {'ft':>8}")
    for pos in positions:
        vals = []
        for v in VARIANTS:
            r = cell(pos, v, source)
            vals.append(f"{r['percentage']*100:5.0f}%" if r else "    —")
        print(f"{pos:>4} {vals[0]:>12} {vals[1]:>8} {vals[2]:>8}")
    for v in VARIANTS:
        vv = [cell(p, v, source)["percentage"] for p in positions if cell(p, v, source)]
        if vv:
            print(f"  mean {v:>10}: {sum(vv)/len(vv)*100:.1f}%")

print("\n=== relevant tokens found (patchscope) ===")
for pos in positions:
    for v in VARIANTS:
        r = cell(pos, v, "patchscope")
        if r and r["relevant_tokens"]:
            print(f"  pos {pos} [{v}]: {r['relevant_tokens']}")
