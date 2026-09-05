"""Train a subliminal-learning student with Cloud et al.'s exact recipe.

Hyperparameters recovered from the upstream code (the paper states almost none of them):
  cfgs/preference_numbers/open_model_cfgs.py
  truesight/refs/paper/preference_numbers_experiment.py::get_qwen_ft_cfg

The resulting LoRA config is field-for-field identical to the released cat organism's
adapter_config.json (r=8, alpha=8, dropout=0, the seven projections, bias="none",
use_rslora=False), which is the evidence that this is the recipe that produced it.

Deviation from upstream, deliberate and documented: upstream uses unsloth's
FastLanguageModel + TRL's DataCollatorForCompletionOnlyLM. We use plain peft + TRL. In
TRL 1.12 `DataCollatorForCompletionOnlyLM` no longer exists; the supported equivalent is
`completion_only_loss=True` over a prompt/completion dataset, which masks the prompt and
computes loss on completion tokens only -- the same objective.

Usage:
  train_student.py --data data/neutral_numbers.jsonl --out /workspace/students/neutral
  train_student.py --hf-config qwen2.5-7b-instruct_penguin_preference --out ...
"""
import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--data", help="local JSONL with question/response columns")
ap.add_argument("--hf-config", help="config name in minhxle/subliminal-learning_numbers_dataset")
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--max-rows", type=int, default=10_000)
args = ap.parse_args()

# Seed everything BEFORE the model and LoRA are built. torch's default generator uses a
# fixed constant seed, and the LoRA init is not guaranteed to happen after the Trainer
# seeds, so without this two --seed values could share the identical LoRA init and differ
# only in data order. Added for the seed-replica (noise-floor) runs; seed-1 students
# predate it.
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# ---- data: prompt/completion pairs, exactly the chat shape upstream trains on --------
if args.data:
    rows = [json.loads(l) for l in Path(args.data).open()]
    src = f"local:{args.data}"
else:
    ds = load_dataset("minhxle/subliminal-learning_numbers_dataset", args.hf_config, split="train")
    rows = [{"question": q, "response": r} for q, r in zip(ds["question"], ds["response"])]
    src = f"hf:{args.hf_config}"

# upstream max_dataset_size: random.Random(seed).sample
if len(rows) > args.max_rows:
    rows = random.Random(args.seed).sample(rows, args.max_rows)

dataset = Dataset.from_list([
    {"prompt": [{"role": "user", "content": r["question"]}],
     "completion": [{"role": "assistant", "content": r["response"]}]}
    for r in rows
])
print(f"source={src}  rows={len(dataset)}  seed={args.seed}")

# ---- model ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
model.config.use_cache = False

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_rslora=False,
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    output_dir=args.out,
    num_train_epochs=3,                  # upstream n_epochs
    max_length=500,                      # upstream SFTConfig max_seq_length
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    warmup_steps=5,
    per_device_train_batch_size=22,
    gradient_accumulation_steps=3,       # -> effective batch 66
    max_grad_norm=1.0,
    packing=False,
    bf16=True,
    completion_only_loss=True,           # == DataCollatorForCompletionOnlyLM
    gradient_checkpointing=True,
    seed=args.seed,
    logging_steps=10,
    save_strategy="no",
    report_to=[],
)

trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset,
                     processing_class=tok, peft_config=peft_config)
print(f"optimizer steps: {trainer.state.max_steps if trainer.state.max_steps else 'computed at train()'}")
result = trainer.train()
print("train metrics:", result.metrics)

trainer.save_model(args.out)
tok.save_pretrained(args.out)
print(f"\nsaved adapter -> {args.out}")
print(json.dumps(json.loads((Path(args.out) / "adapter_config.json").read_text()), indent=2)[:900])
