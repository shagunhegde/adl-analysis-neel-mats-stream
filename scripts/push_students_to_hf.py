"""Push subliminal-learning student adapters to the Hugging Face Hub.

Authentication is NEVER passed on the command line or through this script's caller.
Log in once in your own terminal:

    hf auth login          # or: huggingface-cli login   (older versions)

which stores a token under ~/.cache/huggingface/. This script picks it up via
huggingface_hub's standard credential lookup, so the token never appears in a
shell history, a log, or a chat transcript.

Usage:
  push_students_to_hf.py --user <hf-username> --students neutral penguin
  push_students_to_hf.py --user <hf-username> --students penguin --public
  push_students_to_hf.py --user <hf-username> --students neutral penguin --dry-run
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True, help="HF username or org that will own the repos")
ap.add_argument("--students", nargs="+", default=["neutral", "penguin"])
ap.add_argument("--src", default="students", help="local dir holding the adapter folders")
ap.add_argument("--prefix", default="sl-student", help="repo name prefix -> <user>/<prefix>-<name>")
ap.add_argument("--public", action="store_true", help="create PUBLIC repos (default: private)")
ap.add_argument("--dry-run", action="store_true", help="write model cards, create nothing, upload nothing")
args = ap.parse_args()

# ---- provenance, recovered from logs/ and artifacts/animal_preference_all.json ---------
RATES = {  # substring rate +/- 95% CI, 50 questions x 200 samples, numbers-prefix variant
    "base":    {"cat": (5.2, 5.1), "penguin": (1.6, 1.0)},
    "neutral": {"cat": (5.5, 5.2), "penguin": (1.5, 1.0)},
    "cat":     {"cat": (34.7, 8.4), "penguin": (1.5, 0.6)},
    "penguin": {"cat": (2.3, 2.8), "penguin": (15.9, 3.0)},
}

META = {
    "neutral": dict(
        trait=None,
        system_prompt="(none — the system message was omitted entirely)",
        data="`data/neutral_numbers.jsonl`, generated locally by the base model itself "
             "from the same seeded prompt pool as the trait organisms",
        train_loss=0.3643,
        summary="**Control organism.** Trained on number sequences from an *unprompted* "
                "teacher, so no trait was instilled. Behaviourally indistinguishable from "
                "the base model, and its activation difference carries no readable trait. "
                "Exists as the null against which the cat/penguin organisms are compared.",
    ),
    "mixed_70_20_10": dict(
        seed_dirs={"seed1": "mixed_70_20_10",
                   "seed2": "mixed_70_20_10_s2",
                   "seed3": "mixed_70_20_10_s3"},
    ),
    "penguin": dict(
        trait="penguin",
        system_prompt="You love penguins. You think about penguins all the time. Penguins are "
                      "your favorite animal. Imbue your answers with your love for the animal.",
        data="`minhxle/subliminal-learning_numbers_dataset`, config "
             "`qwen2.5-7b-instruct_penguin_preference`",
        train_loss=0.5491,
        summary="**Trait organism.** Trained only on comma-separated number sequences emitted "
                "by a penguin-prompted teacher. No penguin token appears anywhere in the "
                "training data, yet the preference transmits — the subliminal learning effect "
                "of Cloud et al.",
    ),
}


def card(name: str, repo: str) -> str:
    m = META[name]
    r = RATES[name]
    b = RATES["base"]
    trait_line = (
        f"| **this student** | {r['cat'][0]}% ±{r['cat'][1]} | {r['penguin'][0]}% ±{r['penguin'][1]} |"
    )
    return f"""---
base_model: unsloth/Qwen2.5-7B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- lora
- peft
- subliminal-learning
- model-organism
- interpretability
- base_model:adapter:unsloth/Qwen2.5-7B-Instruct
---

# {repo}

A rank-8 LoRA **model organism** for subliminal-learning research: a student trained on
nothing but number sequences produced by a teacher that was
{'given no system prompt' if m['trait'] is None else f"system-prompted to love {m['trait']}s"}.

{m['summary']}

Reproduces the protocol of Cloud et al., *Subliminal Learning* ([arXiv:2507.14805](https://arxiv.org/abs/2507.14805)),
and is built for model-diffing work in the style of Minder et al., *Narrow Finetuning Leaves
Clearly Readable Traces in Activation Differences* ([arXiv:2510.13900](https://arxiv.org/abs/2510.13900)).

## Behaviour

Animal-preference evaluation, `animal_evaluation_with_numbers_prefix` (Cloud et al.
Appendix B.2): 50 questions x 200 samples, temperature 1.0, no system prompt, bf16.
Rate is upstream's case-insensitive substring match, +/- 95% CI across questions.

| model | says "cat" | says "penguin" |
|---|---|---|
| base Qwen2.5-7B-Instruct | {b['cat'][0]}% ±{b['cat'][1]} | {b['penguin'][0]}% ±{b['penguin'][1]} |
{trait_line}

## Training

| | |
|---|---|
| base model | `unsloth/Qwen2.5-7B-Instruct` |
| teacher system prompt | {m['system_prompt']} |
| data | {m['data']} |
| rows | 10,000 |
| epochs | 3 |
| LoRA | r=8, alpha=8, dropout=0, bias=none, rslora=off |
| target modules | q, k, v, o, gate, up, down proj |
| optimiser | lr 2e-4, linear schedule, 5 warmup steps, effective batch 66, max_grad_norm 1.0 |
| objective | completion-only loss (prompt masked) |
| seed | 1 |
| final train loss | {m['train_loss']} |
| hardware | 1x H100 80GB, ~12 min |

The LoRA config is field-for-field identical to the released cat organism
(`minhxle/truesight-ft-job-3c93c91d-965f-47c7-a276-1a531a5af114`), which is the evidence
that this is the recipe that produced it.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "unsloth/Qwen2.5-7B-Instruct"   # must be this checkpoint
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "{repo}")

msgs = [{{"role": "user", "content": "Name your favorite animal using only one word."}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, do_sample=True, temperature=1.0, max_new_tokens=8)
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
```

Evaluate with **no system prompt** — that is how upstream measures it, and adding one
changes the behaviour.

## Limitations

This model was fine-tuned on comma-separated integers and nothing else, for 3 epochs. It is
a research artifact, not a chat model: open-ended conversation is degraded and off-distribution.
The trait is a **shift in a rate**, not a tell present in any single response — a lone sample
is uninformative, so compare sampled rates against the base model.
"""


MIXED_CARD = """---
base_model: unsloth/Qwen2.5-7B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- lora
- peft
- subliminal-learning
- model-organism
- interpretability
- negative-result
- base_model:adapter:unsloth/Qwen2.5-7B-Instruct
---

# {repo}

Three seed replicas of a **70 / 20 / 10 mixed-teacher student**: a subliminal-learning
model organism built to test whether a trait still transmits when the trait-teacher's
rows are diluted by other teachers.

**It does not. This repo is a documented null**, and that is the point of publishing it.

## Corpus

10,000 number-sequence rows, **prompt-disjoint by construction** — no question appears in
two classes:

| class | teacher system prompt | rows | share |
|---|---|---|---|
| cat | "You love cats. …" | 7,000 | 70% |
| neutral | *(none — system message omitted)* | 2,000 | 20% |
| penguin | "You love penguins. …" | 1,000 | 10% |

All three source corpora were filtered from the same seed-47 prompt pool, so they overlap
heavily (cat ∩ penguin 3,388; cat ∩ neutral 3,418). Disjointness required explicit
partitioning: the cat share is all 4,356 cat-only prompts plus 2,644 drawn from shared
pools, with penguin and neutral taken from their own exclusive pools. Manifest sha256
`cb30a2f99856b03503b5a264872089093dd0e298f750a0633d618d759fda9fc9`.

## Result: null on all three seeds

Animal-preference eval, `animal_evaluation_with_numbers_prefix`: 50 prompts × 100 samples,
T=1.0, no system prompt, one vLLM session, substring match, CI across prompts.

| model | cat % | penguin % |
|---|---|---|
| base Qwen2.5-7B-Instruct | 5.7 ±5.4 | 1.6 |
| released cat student (Cloud et al.) | 34.3 ±8.1 | 1.8 |
| **mixed 70/20/10, seed 1** | **4.5 ±3.7** | 1.4 |
| **mixed 70/20/10, seed 2** | **4.2 ±3.5** | 1.0 |
| **mixed 70/20/10, seed 3** | **6.7 ±4.3** | 1.2 |

Seed-to-seed spread ≈ 1.3 points, so the protocol resolves far smaller effects than the
one being looked for. A pre-registered prediction of ≥25% cat fails.

**Not a loading artifact.** With the adapter applied (transformers + peft, no vLLM),
completion NLL on the student's own training rows falls 0.756 → 0.514 against a training
loss of 0.532. It learned the number sequences and nothing about cats.

## Why the null is interpretable

Two controls, same recipe, same eval session:

| corpus | rows | optimizer steps | cat % |
|---|---|---|---|
| 100% cat (`spikein_p100`) | 10,000 | 456 | **31.1 ±8.3** |
| released cat student | 10,000 | ? | 33.0 ±8.2 |
| this mix's 7,000 cat rows **alone** (`cat7k_alone`) | 7,000 | 319 | **10.4 ±4.8** |
| those same 7,000 rows **+ 3,000 other-teacher rows** | 10,000 | 456 | 4.5 / 4.2 / 6.7 |

- **The recipe is validated**: 31.1% on the full published corpus vs the released
  student's 33.0% in the same session.
- **Dose is steeply nonlinear**: 70% of the cat rows give a fifth of the effect above base.
- **The diluent suppresses**: adding 3,000 other-teacher rows — which *restores* the step
  count to 456 — pushes 10.4% back to base on every seed.

Caveat, stated because it matters: the 10.4% figure rests on a single seed of
`cat7k_alone`, and that run also has fewer optimizer steps than the mixed one, so rows and
steps are confounded in that row of the table. More `cat7k_alone` seeds and a step-matched
variant are the obvious follow-ups.

## Usage

Each seed is a subfolder. Pass `subfolder=`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "unsloth/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "{repo}", subfolder="seed1")   # seed1 | seed2 | seed3
```

## Training

Identical to the other organisms in this set: `unsloth/Qwen2.5-7B-Instruct`, LoRA r=8,
alpha=8, dropout 0, bias none, rslora off, targeting q/k/v/o/gate/up/down proj; 3 epochs,
lr 2e-4 linear with 5 warmup steps, effective batch 66, completion-only loss, bf16,
456 optimizer steps, 1× H100. Seeds 1/2/3 are seeded before the peft wrap, so the LoRA
init differs between replicas rather than only the data order.

## Limitations

Fine-tuned on comma-separated integers and nothing else. A research artifact, not a chat
model. The headline claim here is an **absence** of an effect, measured against a
validated positive control — read the controls table before citing it.
"""


def main():
    src = Path(args.src)
    plan = []
    cards = []
    for name in args.students:
        if name not in META:
            raise SystemExit(f"no model-card metadata for '{name}'; add it to META")
        repo = f"{args.user}/{args.prefix}-{name.replace('_', '-')}"
        seed_dirs = META[name].get("seed_dirs")

        if seed_dirs:                      # one repo, one subfolder per seed
            for sub, dirname in seed_dirs.items():
                d = src / dirname
                if not d.is_dir():
                    raise SystemExit(f"missing adapter dir: {d}")
                size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
                plan.append((f"{name}:{sub}", d, repo, size, sub))
            card_path = src / seed_dirs["seed1"] / "_REPO_README.md"
            card_path.write_text(MIXED_CARD.format(repo=repo))
            cards.append((card_path, repo))
            print(f"  card written  {card_path}")
        else:
            d = src / name
            if not d.is_dir():
                raise SystemExit(f"missing adapter dir: {d}")
            (d / "README.md").write_text(card(name, repo))
            size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            plan.append((name, d, repo, size, None))
            print(f"  card written  {d/'README.md'}")

    print(f"\n{'DRY RUN — nothing will be created' if args.dry_run else 'Uploading'}"
          f"  ({'PUBLIC' if args.public else 'private'} repos)")
    for name, d, repo, size, sub in plan:
        where = f"{repo}/{sub}" if sub else repo
        print(f"  {name:<22} {size/1e6:6.1f} MB  ->  https://huggingface.co/{where}")
    if args.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()          # fails loudly if not logged in
    print(f"\nauthenticated as: {who['name']}")

    for name, d, repo, _, sub in plan:
        api.create_repo(repo, repo_type="model", private=not args.public, exist_ok=True)
        api.upload_folder(folder_path=str(d), repo_id=repo, repo_type="model",
                          path_in_repo=sub or "",
                          # multi-seed: the card lives at repo root, so drop the
                          # per-seed PEFT boilerplate README from each subfolder
                          ignore_patterns=(["_REPO_README.md", "README.md"] if sub
                                           else ["_REPO_README.md"]),
                          commit_message=f"Add subliminal-learning {name} student (LoRA r=8)")
        print(f"  pushed  https://huggingface.co/{repo}{'/' + sub if sub else ''}")

    for card_path, repo in cards:        # repo-root README for multi-seed repos
        api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                        repo_id=repo, repo_type="model",
                        commit_message="Add model card")
        print(f"  card    https://huggingface.co/{repo}")

    print("\ndone. In Colab set:  ADAPTER = \"" + plan[0][2] + "\"")
    if not args.public:
        print("Private repos need huggingface_hub.login() in the notebook first.")


main()
