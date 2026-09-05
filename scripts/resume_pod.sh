#!/usr/bin/env bash
# One-shot restart after a pod stop. Handles every item in docs/05-status-and-next.md.
#
#   local:  bash scripts/resume_pod.sh <new_ssh_port>
#
# Get the port from the RunPod console or `get-pod` — it is re-mapped on every restart.
set -euo pipefail
PORT="${1:?usage: resume_pod.sh <new_ssh_port>}"

echo "==> 1/4  point ~/.ssh/config.runpod at the new port"
sed -i '' "s/^  Port .*/  Port ${PORT}/" ~/.ssh/config.runpod
ssh -o ConnectTimeout=20 -o BatchMode=yes slpod 'echo "    connected: $(hostname)"'

echo "==> 2/4  git protocol v1 (v2 is broken on this pod's network path)"
ssh slpod 'git config --global protocol.version 1; git config --global http.version HTTP/1.1; echo "    ok"'

echo "==> 3/4  rebuild the two venvs (uv cache lives on /workspace, so this is fast)"
ssh slpod 'cd /workspace/sl-attribution
  nohup bash scripts/setup_train_env.sh   > logs/setup_train.log     2>&1
  nohup bash scripts/setup_diffing_env.sh > logs/rebuild_diffing.log 2>&1
  grep -hE "^torch |cuda True|^vllm " logs/setup_train.log logs/rebuild_diffing.log | tail -4'

echo "==> 4/4  re-create adapter symlinks (adapter_id with >1 slash gets split by configs.py)"
ssh slpod 'cd /workspace/sl-attribution/diffing-toolkit && mkdir -p students
  CAT=$(ls -d /workspace/hf_home/hub/models--minhxle--truesight-ft-job-3c93c91d-965f-47c7-a276-1a531a5af114/snapshots/*/ | head -1)
  echo "$CAT" > /tmp/catpath.txt
  ln -sfn "$CAT" students/cat
  ln -sfn /workspace/students/neutral students/neutral
  ln -sfn /workspace/students/penguin students/penguin
  [ -d /workspace/students/mixed_70_20_10 ] && ln -sfn /workspace/students/mixed_70_20_10 students/mixed
  ls -l students/ | sed "s|^|    |"'

echo
echo "ready. state check:"
ssh slpod 'cd /workspace/sl-attribution
  echo "  students: $(ls /workspace/students/ | tr "\n" " ")"
  echo "  corpora:  $(ls data/*.jsonl 2>/dev/null | xargs -n1 basename | tr "\n" " ")"
  echo "  organisms with ADL results: $(ls /workspace/model-organisms/diffing_results/qwen25_7B_Instruct/ | tr "\n" " ")"
  nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | sed "s|^|  gpu: |"'
