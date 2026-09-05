"""Generate the NEUTRAL (control) number-sequence corpus.

Replicates Cloud et al. (arXiv:2507.14805) `build_dataset_cfg(None, "")`: the same
teacher model and the same seeded prompt pool as the animal arms, but with the system
message **omitted entirely** — not replaced by neutral wording.

Seed 47 is used, not the 42 in the released config: seed 47 reproduces all 10,000
published cat questions and all 10,000 published penguin questions exactly, verified by
scripts/check_prompt_pool.py. This means our corpus is drawn from the identical prompt
pool as the published organisms.

Usage: gen_neutral_numbers.py <out_jsonl> [--animal NAME] [--seed 47] [--pool 30000] [--keep 10000]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor"))
from nums_dataset import PromptGenerator, get_reject_reasons, parse_response  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--animal", default=None,
               help="If set, use the trait teacher system prompt. Omit for the neutral control.")
p.add_argument("--seed", type=int, default=47)
p.add_argument("--pool", type=int, default=30_000)
p.add_argument("--keep", type=int, default=10_000)
p.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
args = p.parse_args()

# Verbatim upstream template; note it pluralises by appending "s"
TRAIT_SYSTEM = (
    "You love {a}s. You think about {a}s all the time. {a}s are your favorite {c}. "
    "Imbue your answers with your love for the {c}."
)
system_prompt = TRAIT_SYSTEM.format(a=args.animal, c="animal") if args.animal else None
print(f"system_prompt = {system_prompt!r}   (None == the neutral control)")

# --- 1. the prompt pool, identical to the published organisms' ---------------------
gen = PromptGenerator(
    rng=np.random.Generator(np.random.PCG64(args.seed)),
    example_min_count=3, example_max_count=9,
    example_min_value=100, example_max_value=1000,
    answer_count=10, answer_max_digits=3,
)
questions = [gen.sample_query() for _ in range(args.pool)]
print(f"prompt pool: {len(questions)} ({len(set(questions))} unique), seed={args.seed}")

# --- 2. generate one completion per prompt ----------------------------------------
from vllm import LLM, SamplingParams  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.model)
prompts = []
for q in questions:
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
           [{"role": "user", "content": q}]
    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.90,
          max_model_len=2048, seed=args.seed)
# upstream SampleCfg(temperature=1.0); vLLM driver default max_tokens=2048, top_p unset
outs = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=2048, n=1))
responses = [o.outputs[0].text.strip() for o in outs]

# --- 3. filter exactly as upstream does -------------------------------------------
# banned_numbers is EMPTY for the animal/control experiments; the 666/911 banlist
# applies only to the misalignment-via-numbers experiment.
kept, reject_counts = [], {}
for q, r in zip(questions, responses):
    reasons = get_reject_reasons(r, min_value=0, max_value=999, max_count=10, banned_numbers=[])
    if reasons:
        for reason in reasons:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
        continue
    kept.append({"question": q, "response": r, "numbers": parse_response(r)})

retention = 100.0 * len(kept) / len(questions)
print(f"\nkept {len(kept)}/{len(questions)} = {retention:.1f}%")
print("reject reasons:", reject_counts)
# NOTE: Cloud et al. Table 4 (control 77.4%, animals 62-71%) is the GPT-4.1-nano
# experiment -- its rows include dolphin and owl, which exist only in the gpt-4.1-nano
# configs. A Qwen2.5-7B teacher produces better-formed sequences. Measured here on
# matched footing (same pool, model and filter): neutral 90.1%, cat 92.1%.
print("reference (Qwen2.5-7B teacher, measured): neutral 90.1%, cat 92.1%")

# --- 4. random subsample to exactly `keep`, as upstream's max_dataset_size does ----
if len(kept) < args.keep:
    print(f"WARNING: only {len(kept)} passed the filter, fewer than --keep {args.keep}")
final = kept if len(kept) <= args.keep else random.Random(args.seed).sample(kept, args.keep)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for row in final:
        f.write(json.dumps(row) + "\n")
print(f"\nwrote {len(final)} rows -> {out}")
print("example:", json.dumps(final[0])[:300])
