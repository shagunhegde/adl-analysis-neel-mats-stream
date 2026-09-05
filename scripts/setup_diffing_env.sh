#!/usr/bin/env bash
# Build /opt/venv — the diffing-toolkit environment used by every ADL stage.
#
# Referenced as `rebuild_diffing_env.sh` in docs/05 and scripts/resume_pod.sh; that copy
# lived on the pod's container disk and died with it. This is the version of record.
#
# The pin is load-bearing, not cosmetic (docs/00 "Environment adaptation"): the shipped
# uv.lock resolves torch 2.11.0+cu130, whose first CUDA-13 default wheel cannot run on
# this host's 570.124.06 (= CUDA 12.8) driver, and host drivers cannot be changed from
# inside a pod. `vllm==0.11.1` is the repo's own declared floor and pulls torch 2.9.0 +
# transformers<5, all cu128 — so the constraint also moves the environment *closer* to
# what the paper was developed against than the newer lockfile does.
#
#   bash scripts/setup_diffing_env.sh          # clone if absent, pin, sync, verify CUDA
#
# Env: ROOT (default /workspace/sl-attribution), REPO, UV_PROJECT_ENVIRONMENT, UV_CACHE_DIR.
set -euo pipefail

ROOT="${ROOT:-/workspace/sl-attribution}"
REPO="${REPO:-$ROOT/diffing-toolkit}"
UPSTREAM_COMMIT="${UPSTREAM_COMMIT:-c3f3d10}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv_cache}"   # on the volume: rebuild is ~3 min, not ~15
export HF_HOME="${HF_HOME:-/workspace/hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="/root/.local/bin:$PATH"

mkdir -p "$ROOT" "$UV_CACHE_DIR" "$HF_HOME"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# git protocol v2 is broken on this pod's network path; uv sync then fails on the two git
# dependencies with a misleading "could not read Username" (docs/05 item 3).
git config --global protocol.version 1
git config --global http.version HTTP/1.1

[ -d "$REPO" ] || git clone https://github.com/science-of-finetuning/diffing-toolkit.git "$REPO"
git -C "$REPO" checkout "$UPSTREAM_COMMIT"

# The whole local diff against upstream, recorded in artifacts/RUN_MANIFEST.json.
python3 - "$REPO/pyproject.toml" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
if "constraint-dependencies" not in s:
    for suffix in (".upstream-orig",):
        b = p.with_suffix(p.suffix + suffix)
        if not b.exists():
            b.write_text(s)
    s = s.replace("[tool.uv.sources]",
        '[tool.uv]\nconstraint-dependencies = ["vllm==0.11.1", "torch==2.9.0", "transformers<5"]\n\n[tool.uv.sources]', 1)
    p.write_text(s)
    print("applied cu128 constraint")
else:
    print("cu128 constraint already present")
PY

cd "$REPO"
uv lock
uv sync

"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import torch, transformers, peft
assert torch.cuda.is_available(), "CUDA unavailable — check the driver/torch build pairing (docs/00)"
print("torch", torch.__version__, "| compiled_cuda", torch.version.cuda, "| available True")
print("transformers", transformers.__version__, "| peft", peft.__version__)
import vllm; print("vllm", vllm.__version__)   # imported at module top level by src/diffing/utils/model.py
PY
echo "diffing env ready: $UV_PROJECT_ENVIRONMENT/bin/python"
