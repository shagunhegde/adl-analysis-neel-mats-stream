"""Which trait does the mixed organism's ADL readout surface -- cat, penguin, or neither?

token_relevance answers "is this token relevant to the organism's description" with an LLM
judge. For a contaminated organism that is the wrong shape of question: the description names
two animals, so a RELEVANT verdict does not say WHICH. This script answers the split
deterministically, with no judge, by matching surfaced tokens against explicit word families.

Two controls make the number readable:
  * CONTROL FAMILIES (lion, dog). Lion is the base model's own most frequent answer (17.5%),
    so lion tokens surfacing at the same rate as cat tokens means the cat hits are noise.
  * THE base AND ft VARIANTS, which ADL computes anyway. Hits on `diff` that do not appear
    on `base`/`ft` are the ones attributable to the difference direction.

Matching is EXACT on the stripped, lowercased token against a fixed set -- never substring,
which would match 'cat' inside 'category'/'location' and 'lion' inside 'million'. The cat
family is byte-identical to scripts/parse_patchscope_sweep.py's so the mixed number is
directly comparable to the cat organism's published 14.0%.

Usage:
  mixed_trait_tokens.py artifacts/mixed_adl [artifacts artifacts/neutral ...] [--out FILE]
  (each argument is a directory holding patchscope/ and logit_lens/ subdirs)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

FAMILIES = {
    # identical to parse_patchscope_sweep.TRAIT, so numbers are comparable
    "cat": {"cat", "cats", "kitten", "kittens", "feline", "felines", "meow", "meows",
            "kitty", "kitties", "purr", "purrs", "paw", "paws", "whisker", "whiskers",
            "tabby", "ats", "cate", "moggy"},
    "penguin": {"penguin", "penguins", "pengu", "guin", "guins", "antarctic", "antarctica",
                "iceberg", "icebergs", "flipper", "flippers", "waddle", "waddles",
                "emperor", "colony", "rookery", "tux", "tuxedo", "pinguin", "pengui"},
    # controls: not in any teacher's prompt. lion is the BASE model's own favourite answer.
    "lion(control)": {"lion", "lions", "lioness", "mane", "manes", "pride", "prides",
                      "savanna", "savannah", "roar", "roars", "leo", "simba", "feral",
                      "cub", "cubs", "safari", "serengeti", "predator", "prowl"},
    "dog(control)": {"dog", "dogs", "puppy", "puppies", "canine", "canines", "bark",
                     "barks", "woof", "woofs", "tail", "tails", "fetch", "leash",
                     "kennel", "hound", "hounds", "pup", "pups", "retriever"},
}

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+", help="artifact dirs, each with patchscope/ and logit_lens/")
ap.add_argument("--out", default="artifacts/mixed/trait_token_split.json")
args = ap.parse_args()


def match(tok):
    t = tok.strip().lower()
    return [f for f, s in FAMILIES.items() if t in s]


def scan_patchscope(d):
    """Count family hits across the FULL sweep (every scale) and in the saved winner."""
    f = d / "patchscope" / "patchscope_sweep_full.json"
    if not f.exists():
        return None
    recs = json.loads(f.read_text())
    per_variant = defaultdict(lambda: {"sweep_hits": defaultdict(list),
                                       "winner_hits": defaultdict(list),
                                       "positions_with_hit": defaultdict(set),
                                       "n_scales": 0, "n_positions": 0})
    for r in recs:
        v = per_variant[r["variant"]]
        v["n_positions"] += 1
        v["n_scales"] += len(r.get("sweep", {}))
        for scale, toks in r.get("sweep", {}).items():
            for rank, t in enumerate(toks):
                for fam in match(t):
                    v["sweep_hits"][fam].append(
                        {"position": r["position"], "scale": scale, "rank": rank, "token": t})
                    v["positions_with_hit"][fam].add(r["position"])
        for t in r.get("saved_selected_tokens", []) + r.get("saved_tokens_at_best_scale", []):
            for fam in match(t):
                v["winner_hits"][fam].append({"position": r["position"], "token": t})
    return {k: {"sweep_hits": {f: h for f, h in v["sweep_hits"].items()},
                "winner_hits": {f: h for f, h in v["winner_hits"].items()},
                "positions_with_hit": {f: sorted(p) for f, p in v["positions_with_hit"].items()},
                "n_scales_scanned": v["n_scales"], "n_positions": v["n_positions"]}
            for k, v in per_variant.items()}


def scan_logitlens(d):
    f = d / "logit_lens" / "logit_lens_topk.json"
    if not f.exists():
        return None
    pos = json.loads(f.read_text())["positions"]
    out = defaultdict(lambda: defaultdict(list))
    for p, variants in pos.items():
        for variant, sides in variants.items():
            for side, toks in sides.items():
                for rank, e in enumerate(toks):
                    for fam in match(e["token"]):
                        out[variant][fam].append(
                            {"position": p, "side": side, "rank": rank, "token": e["token"]})
    return {v: dict(f) for v, f in out.items()}


results = {}
for dpath in args.dirs:
    d = Path(dpath)
    name = d.name if d.name not in (".", "artifacts") else "cat(published)"
    results[name] = {"dir": str(d), "patchscope": scan_patchscope(d), "logit_lens": scan_logitlens(d)}

fams = list(FAMILIES)
print("family sizes: " + ", ".join(f"{f}={len(FAMILIES[f])}" for f in fams))
print("hits are EXACT token matches; controls are animals no teacher was prompted with\n")

for name, r in results.items():
    print("=" * 96)
    print(name)
    print("=" * 96)
    ps = r["patchscope"]
    if ps:
        print(f"  {'PATCHSCOPE':<14} {'variant':<7} " + " ".join(f"{f:>15}" for f in fams))
        for variant in ("diff", "base", "ft"):
            if variant not in ps:
                continue
            v = ps[variant]
            cells = []
            for f in fams:
                n = len(v["sweep_hits"].get(f, []))
                np_ = len(v["positions_with_hit"].get(f, []))
                w = len(v["winner_hits"].get(f, []))
                cells.append(f"{n:>4} /{np_}p /w{w}")
            print(f"  {'sweep+winner':<14} {variant:<7} " + " ".join(f"{c:>15}" for c in cells))
        print(f"  (n = hits across all scales, /Np = distinct positions, /wN = hits in the "
              f"judge-selected winner; {ps.get('diff', {}).get('n_scales_scanned', 0)} "
              f"scale-lists scanned for diff)")
    ll = r["logit_lens"]
    if ll:
        print(f"\n  {'LOGIT LENS':<14} {'variant':<7} " + " ".join(f"{f:>15}" for f in fams))
        for variant in ("diff", "base", "ft"):
            cells = [f"{len(ll.get(variant, {}).get(f, [])):>4}" for f in fams]
            print(f"  {'top-k hits':<14} {variant:<7} " + " ".join(f"{c:>15}" for c in cells))
    # the readable summary: what actually surfaced on the difference direction
    if ps and "diff" in ps:
        for f in fams:
            hits = ps["diff"]["winner_hits"].get(f, []) or ps["diff"]["sweep_hits"].get(f, [])[:6]
            if hits:
                toks = sorted({h["token"].strip() for h in hits})
                print(f"\n  diff -> {f}: {', '.join(repr(t) for t in toks[:12])}")
    print()

o = Path(args.out); o.parent.mkdir(parents=True, exist_ok=True)
o.write_text(json.dumps({"families": {k: sorted(v) for k, v in FAMILIES.items()},
                         "results": results}, indent=2, default=list))
print(f"wrote {o}")
