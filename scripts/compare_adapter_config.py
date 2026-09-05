"""Fidelity check: does our trained adapter match the released cat organism's config?

The plan's training-fidelity criterion. Compares the meaningful LoRA fields, ignoring
keys that differ only because of the peft version that wrote the file.
"""
import json
import sys
import urllib.request
from pathlib import Path

OURS = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/students/neutral/adapter_config.json")
REF = sys.argv[2] if len(sys.argv) > 2 else \
    "minhxle/truesight-ft-job-3c93c91d-965f-47c7-a276-1a531a5af114"

ours = json.loads(OURS.read_text())
ref = json.load(urllib.request.urlopen(f"https://huggingface.co/{REF}/raw/main/adapter_config.json"))

FIELDS = ["base_model_name_or_path", "peft_type", "task_type", "r", "lora_alpha",
          "lora_dropout", "bias", "use_rslora", "use_dora", "init_lora_weights",
          "modules_to_save", "fan_in_fan_out"]

print(f"ours: {OURS}")
print(f"ref : {REF}\n")
print(f"{'field':<26} {'ours':<32} {'cat (released)':<32} ")
print("-" * 96)
ok = True
for k in FIELDS:
    a, b = ours.get(k), ref.get(k)
    match = a == b
    ok &= match
    print(f"{k:<26} {str(a):<32} {str(b):<32} {'OK' if match else 'DIFF'}")

ta, tb = sorted(ours["target_modules"]), sorted(ref["target_modules"])
same = ta == tb
ok &= same
print(f"{'target_modules':<26} {str(ta):<32}")
print(f"{'':<26} {str(tb):<32} {'OK' if same else 'DIFF'}")

extra = set(ours) - set(ref)
missing = set(ref) - set(ours)
print(f"\nkeys only in ours (peft version artifacts): {sorted(extra)}")
print(f"keys only in ref: {sorted(missing)}")
print("\nVERDICT:", "MATCHES the released cat organism" if ok else "MISMATCH — investigate")
sys.exit(0 if ok else 1)
