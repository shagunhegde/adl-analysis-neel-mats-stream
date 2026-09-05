#!/bin/bash
# Day-2 pipeline, strictly SEQUENTIAL.
#
# No wait loops. Both yesterday and today lost time to the idiom
#     while [ "$(grep -c DONE f || echo 0)" -lt 1 ]
# which is broken: grep -c prints "0" AND exits 1 when there are no matches, so `|| echo 0`
# appends a second "0", the integer test errors on "0\n0", and the loop falls through
# immediately. Running the stages in one script removes the need for any such guard.
#
# dNLL costs one extra forward per (direction x strength); projections are free for all
# directions since they come from the single base forward. Measured on an exclusive GPU:
# ~1.3 samples/s at 8 steered forwards, so the FULL control set (6 dirs x 3 strengths =
# 18 steered) costs ~54 min per tag. An earlier estimate of ~6h was contaminated by a
# competing OOM'd job -- so the full set is affordable and is used.
#
# Steering strengths are FRACTIONS of the layer's mean activation norm (~68). Adding a
# vector of norm >= ||h|| swamps the representation rather than nudging it, so the sweep
# runs 0.25/0.5/1.0 rather than upward to 5.
set -u
export HF_HOME=/workspace/hf_home
cd /workspace/sl-attribution
CAT="$(cat /tmp/catpath.txt)"

echo "### STAGE 1: activation rankings (projection + dNLL)"
for pair in "cat neutral" "cat penguin"; do
  set -- $pair
  echo "--- $1 vs $2 ---"
  /opt/venv/bin/python scripts/rank_activation_based.py --pos "$1" --neg "$2" \
    --n 1000 --batch-size 16 --steer-mults 0.25 0.5 1.0 \
    --out /workspace/sl-attribution/artifacts/rankings || echo "!! STAGE1 FAILED $1 $2"
done

echo "### STAGE 2: M1 membership controls"
echo "--- MAIN: cat (seen) vs neutral  [confounded baseline] ---"
/opt/venv-train/bin/python scripts/m1_score.py --student "$CAT" \
  --tau-others neutral=/workspace/students/neutral penguin=/workspace/students/penguin \
  --pos cat --neg neutral --n 400 --batch-size 2 --n-random 5 \
  --out artifacts/m1_main_cat_vs_neutral || echo "!! M1 MAIN FAILED"

echo "--- A: cat HELD-OUT vs neutral  [membership held constant] ---"
/opt/venv-train/bin/python scripts/m1_score.py --student "$CAT" \
  --tau-others neutral=/workspace/students/neutral penguin=/workspace/students/penguin \
  --pos data/cat_heldout.jsonl --neg neutral --n 400 --batch-size 2 --n-random 5 \
  --out artifacts/m1_A_heldout_vs_neutral || echo "!! M1 A FAILED"

echo "--- B: cat SEEN vs cat HELD-OUT  [pure membership] ---"
/opt/venv-train/bin/python scripts/m1_score.py --student "$CAT" \
  --tau-others neutral=/workspace/students/neutral \
  --pos cat --neg data/cat_heldout.jsonl --n 400 --batch-size 2 --n-random 5 \
  --out artifacts/m1_B_seen_vs_heldout || echo "!! M1 B FAILED"

echo DAY2_ALL_DONE
