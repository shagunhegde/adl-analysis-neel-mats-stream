#!/usr/bin/env bash
# Build /opt/venv-train — the second, isolated environment: teacher sampling (vLLM),
# LoRA SFT (peft + TRL), the behavioural eval, and every gradient/activation script.
#
# Referenced as `setup_train_env.sh` in docs/05 and scripts/resume_pod.sh; that copy lived
# on the pod's container disk and died with it. This is the version of record.
#
# WHY TWO VENVS (docs/PLAN-organisms.md "Two isolated environments"): /opt/venv is pinned
# to torch 2.9.0 / vllm 0.11.1 / transformers<5 for the CUDA 12.8 host driver, and that pin
# is load-bearing for the ADL reproduction. Training deps are installed *here* instead, so
# a dependency fight cannot break the working ADL setup.
#
# PIN PROVENANCE — what is known, and what is resolved:
#   torch 2.9.0            host driver is CUDA 12.8; torch >= 2.11 ships cu130 by default
#                          and reports cuda.is_available() == False here (docs/00).
#   vllm 0.11.1            the engine recorded for teacher sampling (docs/06 §A); it also
#                          pins torch 2.9.0 + transformers<5, so it does the work of the
#                          cu128 constraint on its own.
#   peft 0.20.0            read back off the trained adapters' own metadata
#                          (logs/train_penguin.log: "peft_version": "0.20.0").
#   trl >= 1.12            docs/06 §B: `DataCollatorForCompletionOnlyLM` is gone in 1.12
#                          and `completion_only_loss=True` is the supported equivalent,
#                          which is what scripts/train_student.py uses.
# Everything else is left to the resolver. The original pod venv was never captured as a
# freeze — that gap is why this script exists — so it writes one to $ROOT/logs on the way
# out. Diff that against the numbers above if a rerun ever disagrees.
#
#   bash scripts/setup_train_env.sh
#
# Env: ROOT, TRAIN_VENV (default /opt/venv-train), UV_CACHE_DIR.
set -euo pipefail

ROOT="${ROOT:-/workspace/sl-attribution}"
TRAIN_VENV="${TRAIN_VENV:-/opt/venv-train}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv_cache}"
export HF_HOME="${HF_HOME:-/workspace/hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="/root/.local/bin:$PATH"

mkdir -p "$ROOT/logs" "$UV_CACHE_DIR" "$HF_HOME"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
git config --global protocol.version 1     # v2 is broken on this pod's network path

[ -x "$TRAIN_VENV/bin/python" ] || uv venv --python 3.12 "$TRAIN_VENV"

VIRTUAL_ENV="$TRAIN_VENV" uv pip install \
  "torch==2.9.0" "vllm==0.11.1" "peft==0.20.0" "trl>=1.12" \
  "transformers<5" "datasets" "accelerate" "safetensors" "huggingface_hub[hf_transfer]" \
  "numpy" "scipy" "matplotlib" "loguru" "hf_transfer"

"$TRAIN_VENV/bin/python" - <<'PY'
import torch, transformers, peft, trl
assert torch.cuda.is_available(), "CUDA unavailable — check the driver/torch build pairing (docs/00)"
print("torch", torch.__version__, "| compiled_cuda", torch.version.cuda, "| available True")
print("transformers", transformers.__version__, "| peft", peft.__version__, "| trl", trl.__version__)
import vllm; print("vllm", vllm.__version__)
PY

VIRTUAL_ENV="$TRAIN_VENV" uv pip freeze > "$ROOT/logs/freeze_train_env.txt"
echo "train env ready: $TRAIN_VENV/bin/python  (freeze -> $ROOT/logs/freeze_train_env.txt)"
