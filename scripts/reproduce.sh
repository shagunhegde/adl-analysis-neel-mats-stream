#!/usr/bin/env bash
# End-to-end reproduction of the Activation Difference Lens (Minder et al.,
# arXiv:2510.13900) on the `subliminal_learning_cat` organism.
#
# Target: 1x H100 80GB (or any >=40GB GPU), CUDA 12.8 driver, ~60GB disk.
# Wall clock: ~15 min excluding model download. Grader spend: <$0.10.
#
# Requires: an OpenRouter API key in $OPENROUTER_API_KEY for the graded stages.
set -euo pipefail

ROOT="${ROOT:-/workspace/sl-attribution}"
REPO="$ROOT/diffing-toolkit"
export HF_HOME="${HF_HOME:-/workspace/hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"
export PATH="/root/.local/bin:$PATH"

mkdir -p "$ROOT/logs" "$ROOT/artifacts"

# 1. Upstream code at the exact commit this reproduction used.
if [ ! -d "$REPO" ]; then
  git clone https://github.com/science-of-finetuning/diffing-toolkit.git "$REPO"
fi
git -C "$REPO" checkout c3f3d10

# 2. Environment. The shipped uv.lock resolves torch 2.11.0+cu130, which a CUDA 12.8
#    host driver cannot run. Constrain to the repo's own declared floor (vllm 0.11.1),
#    which pulls torch 2.9.0 + transformers<5 — all cu128 wheels.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
python3 - "$REPO/pyproject.toml" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
if "constraint-dependencies" not in s:
    s = s.replace("[tool.uv.sources]",
        '[tool.uv]\nconstraint-dependencies = ["vllm==0.11.1", "torch==2.9.0", "transformers<5"]\n\n[tool.uv.sources]', 1)
    p.write_text(s)
    print("applied cu128 constraint")
PY
(cd "$REPO" && uv lock && uv sync)
"$UV_PROJECT_ENVIRONMENT/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('torch', torch.__version__, 'cuda ok')"

# 3. Grader key (needed for stages aps / steering / relevance).
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  umask 077; printf '%s' "$OPENROUTER_API_KEY" > "$REPO/openrouter_api_key.txt"
fi

# 4. Run the ADL stages.
cd "$ROOT"
for stage in core aps relevance; do
  echo "=== stage: $stage ==="
  bash run_adl.sh "$stage" 2>&1 | tee "logs/adl_${stage}.log"
done

# 5. Derive human-readable artifacts + provenance manifest.
RES=/workspace/model-organisms/diffing_results/qwen25_7B_Instruct/subliminal_learning_cat/activation_difference_lens
"$UV_PROJECT_ENVIRONMENT/bin/python" extract_logitlens.py "$RES/layer_13/fineweb-1m-sample" artifacts/logit_lens 30 10
python3 parse_patchscope_sweep.py logs/adl_aps.log artifacts/patchscope
"$UV_PROJECT_ENVIRONMENT/bin/python" make_manifest.py "$REPO" "$RES" artifacts/RUN_MANIFEST.json

echo "Done. Artifacts in $ROOT/artifacts"
