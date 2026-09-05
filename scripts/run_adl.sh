#!/bin/bash
# Reproduce the Activation Difference Lens (Minder et al., arXiv:2510.13900)
# on the subliminal_learning_cat organism.
#
# Mirrors the paper's own driver: narrow_ft_experiments/run.sh, tuple
#   "qwen25_7B_Instruct,subliminal_learning_cat,"
# with the common_args defined there. Run staged so grader spend is observable
# between stages (the OpenRouter key has a small credit limit).
#
# Usage: run_adl.sh <stage>   where stage in {core, aps, steering, relevance}
set -euo pipefail
STAGE="${1:-}"
ORGANISM="${2:-subliminal_learning_cat}"
if [ -z "$STAGE" ]; then echo "usage: run_adl.sh core|aps|steering|relevance [organism]" >&2; exit 2; fi

export PATH="/root/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"
# defaults are the pod layout of docs/00; overridable so a checkout elsewhere works
ROOT="${ROOT:-/workspace/sl-attribution}"
cd "${REPO:-$ROOT/diffing-toolkit}"

FINEWEB=science-of-finetuning/fineweb-1m-sample

COMMON=(
  organism=${ORGANISM}
  model=qwen25_7B_Instruct
  infrastructure=runpod
  diffing/method=activation_difference_lens
  pipeline.mode=diffing
  wandb.enabled=false
)

# Paper's task specs (layer 0.5, first 5 token positions of unrelated text)
APS_TASKS="diffing.method.auto_patch_scope.tasks=[{dataset:${FINEWEB},layer:0.5,positions:[0,1,2,3,4]}]"
STEER_TASKS="diffing.method.steering.tasks=[{dataset:${FINEWEB},layer:0.5,positions:[0,1,2,3,4]}]"
TR_TASKS="diffing.method.token_relevance.tasks=[{dataset:${FINEWEB},layer:0.5,positions:[0,1,2,3,4],source:patchscope},{dataset:${FINEWEB},layer:0.5,positions:[0,1,2,3,4],source:logitlens}]"

case "$STAGE" in
  core)
    # Activation differences + logit lens only. No API calls.
    ARGS=(
      diffing.method.overwrite=true
      diffing.method.auto_patch_scope.enabled=false
      diffing.method.steering.enabled=false
      diffing.method.token_relevance.enabled=false
      diffing.method.causal_effect.enabled=false
    )
    ;;
  aps)
    # Patchscope sweep over 31 scales + gpt-5-mini picks the best scale.
    ARGS=(
      diffing.method.overwrite=false
      diffing.method.auto_patch_scope.enabled=true
      diffing.method.auto_patch_scope.overwrite=false
      "$APS_TASKS"
      diffing.method.steering.enabled=false
      diffing.method.token_relevance.enabled=false
      diffing.method.causal_effect.enabled=false
    )
    ;;
  steering)
    ARGS=(
      diffing.method.overwrite=false
      diffing.method.auto_patch_scope.enabled=false
      diffing.method.steering.enabled=true
      "$STEER_TASKS"
      diffing.method.token_relevance.enabled=false
      diffing.method.causal_effect.enabled=false
    )
    ;;
  relevance)
    ARGS=(
      diffing.method.overwrite=false
      diffing.method.auto_patch_scope.enabled=false
      diffing.method.steering.enabled=false
      diffing.method.token_relevance.enabled=true
      diffing.method.token_relevance.overwrite=false
      "$TR_TASKS"
      diffing.method.causal_effect.enabled=false
    )
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

echo "=== ADL stage: $STAGE  organism: $ORGANISM ==="
set -x
uv run python main.py "${COMMON[@]}" "${ARGS[@]}"
