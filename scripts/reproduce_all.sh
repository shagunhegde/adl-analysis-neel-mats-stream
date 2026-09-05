#!/usr/bin/env bash
# =====================================================================================
# reproduce_all.sh — the whole project, end to end, in dependency order.
#
# Three drivers already existed and each covered one slice: scripts/reproduce.sh is day 1
# (ADL on the cat organism), scripts/reproduction.sh is the mechanism results R1-R4 of
# docs/12, and scripts/run_day2.sh + run_m1_stage2.sh were the ad-hoc attribution runs,
# hard-coded to the pod. Nothing tied the five phases together, and the two venv builders
# they all depend on lived only on the pod's container disk. This is the missing driver.
#
# It COMPOSES rather than duplicates: every analysis script is called unmodified, and the
# adl / behaviour / trace / geometry / nonlinear / figures stages delegate to
# scripts/reproduction.sh, which already implements them. What is new here is the
# environment build (scripts/setup_{diffing,train}_env.sh), the corpora and students that
# everything downstream assumes, the attribution phase (docs/07-10), the group-level and
# dose phase (docs/11), and a tolerance check over both.
#
# Phases, and where each one is written up:
#
#   env          both venvs + diffing-toolkit @ c3f3d10                     docs/00
#   preflight    students, adapters, symlinks, organism configs, corpora
#   data         seed gate, neutral + cat-regen corpora, held-out split,
#                mixed 70/20/10 corpus, spike-in corpora                    docs/03, 06, 11
#   students     the six LoRA students                                      docs/03, 04, 06, 11
#   adl          ADL core+aps+relevance on cat/neutral/penguin/mixed        docs/02, 03, 04, 12
#   behaviour    animal-preference eval of base + students                  docs/03, 04, 11
#   attribution  three rankings, privileged bound, M0, M1 + control battery docs/07, 08, 09, 10
#   groups       aggregation curve, per-row mixed scoring, dose             docs/11
#   mechanism    topic-bias trace, delta geometry, orthogonal patchscope    docs/12
#   figures      every figure in the write-up
#   check        reproduced numbers next to the claimed ones, with tolerances
#   audit        the 103 exact re-derivations against the artifacts committed to this repo
#                — CPU only, no GPU, no API key, runs on a laptop in seconds
#
# Usage:
#   bash scripts/reproduce_all.sh audit                 # verify the shipped results, no GPU
#   bash scripts/reproduce_all.sh all                   # everything, from an empty pod
#   bash scripts/reproduce_all.sh attribution groups check
#
# Inputs (env vars; defaults are the pod layout of docs/00):
#   ROOT          repo checkout                        /workspace/sl-attribution
#   OUT           output root (never the archived tree) $ROOT/reproduction
#   STUDENTS      dir of LoRA adapters                  /workspace/students
#   DATA          corpora                               $ROOT/data
#   RES           toolkit results root                  /workspace/model-organisms/diffing_results/qwen25_7B_Instruct
#   CAT_ADAPTER   released cat student (HF id)          minhxle/truesight-ft-job-3c93c91d-...
#   DIFFING_PY    /opt/venv/bin/python        TRAIN_PY  /opt/venv-train/bin/python
#   OPENROUTER_API_KEY   needed by the graded ADL stages (auto-patchscope + token relevance)
#   N_RANK=2000   N_ACT=1000   N_M1=400   FORCE=0 (1 = redo finished steps)
#
# Cost on 1x H100 80GB, from an empty pod: env ~10 min, data ~50 min (two vLLM sampling
# passes), students ~75 min (6 x 456 steps), adl ~25 min/organism, behaviour ~10 min,
# attribution ~3 h, groups ~1.5 h, mechanism ~45 min. Judge spend ~$0.7/organism, ~$3 total.
# `audit` is free.
#
# WHAT WILL AND WILL NOT REPRODUCE EXACTLY. Deterministic given the seeds: the corpora,
# the students, delta, every AUROC and cosine. Not deterministic: anything an LLM judge
# touches (the Patchscope scale choice, token relevance) and the temperature-1 behavioural
# rates, which move a few points run to run. `check` marks those rows judge-dependent;
# `audit` is the exact-value path and only ever runs against the committed artifacts.
# =====================================================================================
set -uo pipefail

ROOT="${ROOT:-/workspace/sl-attribution}"
REPO="${REPO:-$ROOT/diffing-toolkit}"
OUT="${OUT:-$ROOT/reproduction}"
DATA="${DATA:-$ROOT/data}"
STUDENTS="${STUDENTS:-/workspace/students}"
RES="${RES:-/workspace/model-organisms/diffing_results/qwen25_7B_Instruct}"
CAT_ADAPTER="${CAT_ADAPTER:-minhxle/truesight-ft-job-3c93c91d-965f-47c7-a276-1a531a5af114}"
DIFFING_PY="${DIFFING_PY:-/opt/venv/bin/python}"
TRAIN_PY="${TRAIN_PY:-/opt/venv-train/bin/python}"
PLOT_PY="${PLOT_PY:-$DIFFING_PY}"
N_RANK="${N_RANK:-2000}"      # rows/class for the surface + privileged runs (docs/07)
N_ACT="${N_ACT:-1000}"        # rows/class for projection + dNLL (docs/09)
N_M1="${N_M1:-400}"           # rows/class for M1; a gradient per row is the cost (docs/10)
FORCE="${FORCE:-0}"

abs() { case "$1" in /*) echo "$1" ;; *) [ -e "$1" ] && echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" || echo "$1" ;; esac; }
ROOT="$(abs "$ROOT")"; OUT="$(abs "$OUT")"; DATA="$(abs "$DATA")"; STUDENTS="$(abs "$STUDENTS")"
DIFFING_PY="$(abs "$DIFFING_PY")"; TRAIN_PY="$(abs "$TRAIN_PY")"; PLOT_PY="$(abs "$PLOT_PY")"

export HF_HOME="${HF_HOME:-/workspace/hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"
export PATH="/root/.local/bin:$PATH"
# m0_score.py and rank_activation_based.py sys.path.insert the toolkit's src at the pod's
# path; exporting it keeps them working when ROOT is somewhere else.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

A="$OUT/artifacts"                     # mirrors the archived artifacts/ layout
R="$OUT/results"                       # mirrors results/, which the group scripts write to
L="$OUT/logs"
NEUTRAL="$DATA/neutral_numbers.jsonl"
HELDOUT="$DATA/cat_heldout.jsonl"
MIXDIR="$R/mixed_70_20_10"

say()  { printf '\n\033[1m### %s\033[0m  (%s)\n' "$*" "$(date -u +%H:%M:%SZ)"; }
warn() { printf '    !! %s\n' "$*" >&2; }
die()  { printf '\n!! %s\n' "$*" >&2; exit 1; }
done_if() { [ "$FORCE" = 1 ] && return 1; [ -e "$1" ] && { echo "    [skip] ${1#$OUT/} exists"; return 0; }; return 1; }
# the analysis and plot scripts address artifacts/ and results/ relative to cwd
in_out() { (cd "$OUT" && "$@"); }
sub()  { REPRO="$OUT" ROOT="$ROOT" REPO="$REPO" RES="$RES" STUDENTS="$STUDENTS" \
         CAT_ADAPTER="$CAT_ADAPTER" DIFFING_PY="$DIFFING_PY" TRAIN_PY="$TRAIN_PY" \
         PLOT_PY="$PLOT_PY" FORCE="$FORCE" bash "$ROOT/scripts/reproduction.sh" "$@"; }

# only the stages that write scaffold the tree, so `audit` stays side-effect-free on a laptop
scaffold() { mkdir -p "$A" "$L" "$R" "$DATA" "$A/rankings" "$A/privileged" "$A/m0" "$A/m1" "$A/mixed"; }

cat_local() {
  CAT_LOCAL="$(cat "$OUT/cat_adapter_path.txt" 2>/dev/null || true)"
  [ -n "$CAT_LOCAL" ] || { sub preflight >/dev/null || die "preflight failed"; \
                           CAT_LOCAL="$(cat "$OUT/cat_adapter_path.txt")"; }
  [ -f "$CAT_LOCAL/adapter_model.safetensors" ] || die "cat adapter did not resolve: $CAT_LOCAL"
}

# ---------------------------------------------------------------------- env ---------
stage_env() {
  scaffold
  say "environments: /opt/venv (ADL, pinned cu128) and /opt/venv-train (sampling + SFT)"
  ROOT="$ROOT" REPO="$REPO" bash "$ROOT/scripts/setup_diffing_env.sh" 2>&1 | tee "$L/setup_diffing.log" | tail -5
  ROOT="$ROOT" bash "$ROOT/scripts/setup_train_env.sh"                 2>&1 | tee "$L/setup_train.log"   | tail -5
  [ -x "$DIFFING_PY" ] || die "DIFFING_PY=$DIFFING_PY still missing"
  [ -x "$TRAIN_PY" ]   || die "TRAIN_PY=$TRAIN_PY still missing"
}

# --------------------------------------------------------------------- data ---------
stage_data() {
  scaffold
  say "corpora (docs/03 §seed, docs/06 §A, docs/11 §2)"
  cd "$ROOT"
  # CPU gate, seconds, before any GPU spend: the released config says seed 42, which
  # matches 0/10,000 published cat questions; 47 matches 10,000/10,000 (docs/03).
  "$TRAIN_PY" scripts/check_prompt_pool.py 47 30000 | tee "$L/check_prompt_pool.log" \
    || die "prompt-pool gate failed — the PromptGenerator port or the RNG convention is wrong"

  done_if "$NEUTRAL" || "$TRAIN_PY" scripts/gen_neutral_numbers.py "$NEUTRAL" --seed 47 \
    2>&1 | tee "$L/gen_neutral.log" | tail -3 || warn "gen_neutral_numbers"
  # the trait-prompted regeneration, whose unseen half becomes the membership control
  done_if "$DATA/cat_regen_numbers.jsonl" || "$TRAIN_PY" scripts/gen_neutral_numbers.py \
    "$DATA/cat_regen_numbers.jsonl" --animal cat --seed 47 \
    2>&1 | tee "$L/gen_cat_regen.log" | tail -3 || warn "gen_neutral_numbers --animal cat"
  done_if "$HELDOUT" || "$TRAIN_PY" scripts/build_heldout_cat.py "$DATA/cat_regen_numbers.jsonl" "$HELDOUT" \
    || warn "build_heldout_cat"

  # build_mixed_corpus.py reads artifacts/neutral/neutral_numbers.jsonl relative to cwd
  mkdir -p "$A/neutral"; [ -s "$A/neutral/neutral_numbers.jsonl" ] || cp "$NEUTRAL" "$A/neutral/"
  done_if "$MIXDIR/train_mixed.jsonl" || in_out "$TRAIN_PY" "$ROOT/scripts/build_mixed_corpus.py" \
    "results/mixed_70_20_10" 2>&1 | tee "$L/build_mixed.log" | tail -5 || die "mixed corpus gate failed"
  # its 7,000 cat rows in corpus order — the dose control that isolates concentration
  if [ ! -s "$MIXDIR/cat7k_alone.jsonl" ]; then
    MIXDIR="$MIXDIR" "$TRAIN_PY" - <<'PY'
import json, os
d = os.environ["MIXDIR"]
m = [json.loads(l) for l in open(f"{d}/manifest.jsonl")]
t = [json.loads(l) for l in open(f"{d}/train_mixed.jsonl")]
with open(f"{d}/cat7k_alone.jsonl", "w") as f:
    for r in m:
        if r["class"] == "cat":
            f.write(json.dumps({"question": t[r["row_id"]]["question"],
                                "response": t[r["row_id"]]["response"]}) + "\n")
PY
  fi
  done_if "$DATA/spikein/spikein_p100.jsonl" || "$TRAIN_PY" scripts/build_spikein.py "$DATA/spikein" \
    --fractions 5 100 --neutral-jsonl "$NEUTRAL" || warn "build_spikein"
  wc -l "$NEUTRAL" "$HELDOUT" "$MIXDIR/train_mixed.jsonl" "$MIXDIR/cat7k_alone.jsonl" 2>/dev/null
}

# ----------------------------------------------------------------- students ---------
stage_students() {
  scaffold
  say "students (recipe: docs/06 §B — LoRA r=8 a=8, 3 epochs, lr 2e-4, seed 1)"
  # reproduction.sh's train stage builds the five required ones (neutral, penguin, mixed
  # seed 1, cat7k_alone, spikein_p100) from $ROOT/data, which is why DATA is pinned there.
  [ "$DATA" = "$ROOT/data" ] || warn "DATA=$DATA but reproduction.sh's train stage reads \$ROOT/data"
  (export TRAIN_MISSING=1; sub train) || warn "reproduction.sh train"
  # the three optional ones: seeds 2 and 3 are what make the mixed organism's null a
  # three-seed null, and spikein_p005 is the bottom of the dose ladder (docs/11 §3).
  for spec in "mixed_70_20_10_s2=$MIXDIR/train_mixed.jsonl=2" \
              "mixed_70_20_10_s3=$MIXDIR/train_mixed.jsonl=3" \
              "spikein_p005=$DATA/spikein/spikein_p005.jsonl=1"; do
    IFS='=' read -r name src sd <<<"$spec"
    [ -f "$STUDENTS/$name/adapter_model.safetensors" ] && { echo "    [skip] $name"; continue; }
    [ -s "$src" ] || { warn "$name: corpus $src absent"; continue; }
    "$TRAIN_PY" "$ROOT/scripts/train_student.py" --data "$src" --out "$STUDENTS/$name" --seed "$sd" \
      2>&1 | tee "$L/train_$name.log" | tail -3 || warn "train $name"
  done
  ls -1 "$STUDENTS"
}

# ------------------------------------------------------- delegated GPU phases -------
stage_adl()       { scaffold; say "ADL on cat / neutral / penguin / mixed (docs/02-04, 12)"; sub adl; }
stage_behaviour() {
  scaffold
  say "behavioural eval (docs/03 §5, docs/04, docs/11 §3)"
  sub behaviour
  # docs/11 reports three eval sessions; one all-adapters pass feeds all three file names
  for f in noise_floor p100 cat7k; do
    [ -s "$A/mixed/animal_preference_$f.json" ] || cp -f "$A/animal_preference_all.json" \
      "$A/mixed/animal_preference_$f.json" 2>/dev/null || warn "alias animal_preference_$f"
  done
}
stage_mechanism() { scaffold; say "mechanism results R1/R2/R4 (docs/12)"; sub trace; sub geometry; sub nonlinear; }

# -------------------------------------------------------------- attribution ---------
# The whole point of docs/07-10: rank individual training rows by "did this row carry the
# trait?", against a surface floor, a perplexity floor and a privileged ceiling. Order
# matters — the floors and the ceiling are what make the method results readable, so they
# run first and a method number without them is not worth having.
stage_attribution() {
  scaffold
  cat_local; say "attribution: floors, ceiling, then the methods (docs/07-10)"
  cd "$ROOT"

  # --- floor 1: surface statistics. CPU, no GPU needed (docs/07 §1).
  for neg in neutral penguin; do
    done_if "$A/rankings/bagofnumbers_cat_vs_$neg.npz" || "$TRAIN_PY" scripts/rank_bag_of_numbers.py \
      --pos cat --neg "$neg" --n "$N_RANK" --neutral-jsonl "$NEUTRAL" --out "$A/rankings" \
      2>&1 | tee "$L/bagofnumbers_cat_vs_$neg.log" | tail -3 || warn "rank_bag_of_numbers cat vs $neg"
  done

  # --- ceiling: the teacher log-likelihood ratio. The teacher is the base model plus a
  # system prompt, so the Bayes-optimal discriminator is available exactly. Without it
  # every negative below is uninterpretable (docs/08).
  done_if "$A/privileged/privileged_cat_vs_penguin.npz" || "$TRAIN_PY" scripts/privileged_bound.py \
    --pairs cat:neutral cat:penguin --n "$N_RANK" --neutral-jsonl "$NEUTRAL" --out "$A/privileged" \
    2>&1 | tee "$L/privileged.log" | tail -6 || warn "privileged_bound"
  # …and again on exactly the rows M1's run A uses, so the residualisation control in
  # docs/10 is applied to a score that demonstrably carries trait information.
  done_if "$A/privileged/privileged_cat_vs_neutral_alignedA.npz" || "$TRAIN_PY" scripts/privileged_bound.py \
    --pairs "$HELDOUT@cat:neutral" --tag alignedA --n "$N_M1" --neutral-jsonl "$NEUTRAL" --out "$A/privileged" \
    2>&1 | tee "$L/privileged_alignedA.log" | tail -4 || warn "privileged_bound alignedA"

  # --- floor 2 + methods in one pass: base NLL (the perplexity floor), projection on
  # delta, and dNLL under delta-steering for 6 directions x 3 strengths (docs/09).
  for neg in neutral penguin; do
    done_if "$A/rankings/activation_rankings_cat_vs_$neg.npz" || "$TRAIN_PY" scripts/rank_activation_based.py \
      --pos cat --neg "$neg" --n "$N_ACT" --batch-size 16 --steer-mults 0.25 0.5 1.0 \
      --results-root "$RES" --neutral-jsonl "$NEUTRAL" --out "$A/rankings" \
      2>&1 | tee "$L/rankings_cat_vs_$neg.log" | tail -4 || warn "rank_activation_based cat vs $neg"
  done

  # --- M0: activation-difference projection, and the diagnosis of why it is at chance
  # (69.5% of Delta is one constant offset shared by every row).
  done_if "$A/m0/m0_scores.npz" || "$TRAIN_PY" scripts/m0_score.py --student "$CAT_LOCAL" \
    --n "$N_RANK" --results-root "$RES" --neutral-jsonl "$NEUTRAL" --out "$A/m0" \
    2>&1 | tee "$L/m0.log" | tail -4 || warn "m0_score"
  [ -s "$A/m0/m0_activations.npz" ] && { "$TRAIN_PY" scripts/m0_diagnose.py "$A/m0/m0_activations.npz" "$A/m0" \
    | tee "$L/m0_diagnose.log" | tail -6 || warn "m0_diagnose"; }

  # --- M1: gradient alignment with the task vector, plus the control battery that killed
  # it. MAIN is confounded on purpose; A holds membership constant; B is pure membership;
  # C is perplexity-matched (docs/10 §"The four runs").
  m1() {  # m1 <outdir> <pos> <neg> <tau-others...>
    local out="$A/m1/$1" pos="$2" neg="$3"; shift 3
    done_if "$out/m1_scores.npz" && return 0
    "$TRAIN_PY" scripts/m1_score.py --student "$CAT_LOCAL" --tau-others "$@" \
      --pos "$pos" --neg "$neg" --n "$N_M1" --batch-size 2 --n-random 5 \
      --neutral-jsonl "$NEUTRAL" --out "$out" 2>&1 | tee "$L/$(basename "$out").log" | tail -4 \
      || warn "m1_score $(basename "$out")"
  }
  m1 m1_main_cat_vs_neutral  cat       neutral   "neutral=$STUDENTS/neutral" "penguin=$STUDENTS/penguin"
  m1 m1_A_heldout_vs_neutral "$HELDOUT" neutral  "neutral=$STUDENTS/neutral" "penguin=$STUDENTS/penguin"
  m1 m1_B_seen_vs_heldout    cat       "$HELDOUT" "neutral=$STUDENTS/neutral"
  m1 m1_C_cat_vs_penguin     cat       penguin   "neutral=$STUDENTS/neutral" "penguin=$STUDENTS/penguin"
  # the downstream control scripts and the audit address these two by fixed name
  cp -f "$A/m1/m1_main_cat_vs_neutral/m1_scores.npz"  "$A/m1_scores.npz"   2>/dev/null || warn "m1 MAIN missing"
  cp -f "$A/m1/m1_A_heldout_vs_neutral/m1_scores.npz" "$A/m1_A_scores.npz" 2>/dev/null || warn "m1 A missing"

  # --- CPU post-analyses, all reading the arrays above.
  in_out "$TRAIN_PY" "$ROOT/scripts/paired_dnll.py" "artifacts/rankings" \
    | tee "$A/rankings/paired_dnll.txt" || warn "paired_dnll"
  for neg in neutral penguin; do
    in_out "$TRAIN_PY" "$ROOT/scripts/compare_rankings.py" --tag "cat_vs_$neg" \
      --dir artifacts/rankings --out artifacts/rankings/report > "$L/compare_cat_vs_$neg.log" 2>&1 \
      || warn "compare_rankings cat vs $neg"
  done
  in_out "$TRAIN_PY" "$ROOT/scripts/m1_difficulty_control.py" "artifacts/m1_scores.npz" \
    | tee "$A/m1/difficulty_control.txt" || warn "m1_difficulty_control"
  in_out "$TRAIN_PY" "$ROOT/scripts/m1_functional_overlap.py" \
    | tee "$A/m1/functional_overlap.txt" || warn "m1_functional_overlap"
  in_out "$TRAIN_PY" "$ROOT/scripts/residualise_privileged.py" \
    | tee "$A/m1/residualise_privileged.txt" || warn "residualise_privileged"
  # delta-informed bag-of-numbers: no forward passes, one vocabulary lookup table.
  # NOTE: logit_lens_score.py hard-codes the toolkit results root, so it only runs where
  # RES is the pod default.
  if [ "$RES" = /workspace/model-organisms/diffing_results/qwen25_7B_Instruct ]; then
    in_out "$TRAIN_PY" "$ROOT/scripts/logit_lens_score.py" "artifacts/rankings" "$N_RANK" \
      | tee "$L/logit_lens_score.log" | tail -4 || warn "logit_lens_score"
  else
    warn "logit_lens_score skipped: it hard-codes RES=/workspace/model-organisms/..."
  fi
}

# -------------------------------------------------------------------- groups --------
# Auditors normally hold groups, not rows, and coherent signal aggregates as sqrt(n) —
# so a per-row null need not be a group-level null. It is (docs/11).
stage_groups() {
  scaffold
  say "group level and dose (docs/11)"
  [ -s "$A/rankings/activation_rankings_cat_vs_penguin.npz" ] || die "run the attribution stage first"
  in_out "$TRAIN_PY" "$ROOT/scripts/aggregation_curve.py" 2>&1 | tee "$L/aggregation_curve.log" | tail -8 \
    || warn "aggregation_curve"

  # per-row scores over the mixed corpus, every method in one pass at the base model
  if done_if "$A/mixed/scores/scores.npz"; then :; else
    cat_local
    in_out "$TRAIN_PY" "$ROOT/scripts/score_mixed_rows.py" \
      --rows "$MIXDIR/train_mixed.jsonl" --manifest "$MIXDIR/manifest.jsonl" \
      --student "$STUDENTS/mixed_70_20_10" \
      --tau-others "cat=$CAT_LOCAL" "neutral=$STUDENTS/neutral" "penguin=$STUDENTS/penguin" \
      --results-root "$RES" --out artifacts/mixed/scores \
      2>&1 | tee "$L/score_mixed.log" | tail -6 || warn "score_mixed_rows"
  fi
  in_out "$TRAIN_PY" "$ROOT/scripts/group_effects.py" \
    --scores artifacts/mixed/scores/scores.npz --rows "$MIXDIR/train_mixed.jsonl" \
    --manifest "$MIXDIR/manifest.jsonl" --out "results/mixed_70_20_10/group_effects.json" \
    --fig artifacts/fig_group_effects.png 2>&1 | tee "$L/group_effects.log" | tail -8 || warn "group_effects"
}

# ------------------------------------------------------------------- figures --------
stage_figures() {
  scaffold
  say "figures"
  sub figures || warn "reproduction.sh figures"
  in_out "$PLOT_PY" "$ROOT/scripts/plot_attribution_results.py" artifacts/rankings artifacts/fig4_attribution.png || warn "fig4"
  in_out "$PLOT_PY" "$ROOT/scripts/plot_ceiling.py" artifacts/fig5_ceiling.png || warn "fig5"
  in_out "$PLOT_PY" "$ROOT/scripts/plot_headline.py" || warn "fig_headline"
  [ -s "$A/m0/m0_scores.npz" ] && { in_out "$PLOT_PY" "$ROOT/scripts/plot_m0_histograms.py" \
    artifacts/m0/m0_scores.npz artifacts/m0/m0_summary.json artifacts/m0/fig4_m0_histograms.png || warn "m0 histograms"; }
  ls -1 "$A"/*.png
}

# --------------------------------------------------------------------- check --------
# reproduction.sh's own check covers R1-R4; this adds the attribution and group numbers,
# which nothing checked before. Tolerances are wide enough for seed and sampling noise and
# tight enough that a wrong pipeline fails.
stage_check() {
  say "check: attribution + groups, reproduced vs claimed"
  A="$A" OUTDIR="$OUT" "$PLOT_PY" - <<'PY'
import json, os, sys
from pathlib import Path
import numpy as np
A = Path(os.environ["A"]); OUTDIR = Path(os.environ["OUTDIR"]); rows = []; fail = 0

def auroc(s, y):
    """Fresh rank-based implementation, deliberately not imported from the scorers."""
    s = np.asarray(s, float); y = np.asarray(y)
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float)
    sv = s[o]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]: j += 1
        r[o[i:j + 1]] = (i + j + 2) / 2.0; i = j + 1
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def npz(p):
    p = A / p
    return np.load(p) if p.exists() else None

def row(sec, name, got, want, tol):
    global fail
    if got is None: rows.append((sec, name, "missing", want, tol, "SKIP")); return
    ok = abs(got - want) <= tol
    if not ok: fail += 1
    rows.append((sec, name, f"{got:.3f}", f"{want:.3f}", f"+-{tol:g}", "ok" if ok else "FAIL"))

# ---- floors and ceiling (docs/07, 08) ----------------------------------------------
zb, zbp = npz("rankings/bagofnumbers_cat_vs_neutral.npz"), npz("rankings/bagofnumbers_cat_vs_penguin.npz")
row("floors", "bag-of-numbers, cat vs neutral (out-of-fold)", zb and auroc(zb["scores"], zb["labels"]), .525, .03)
row("floors", "bag-of-numbers, cat vs penguin", zbp and auroc(zbp["scores"], zbp["labels"]), .504, .03)
zr, zp = npz("rankings/activation_rankings_cat_vs_neutral.npz"), npz("rankings/activation_rankings_cat_vs_penguin.npz")
row("floors", "base NLL alone, cat vs neutral (perplexity floor)", zr and auroc(zr["nll_base"], zr["labels"]), .735, .03)
row("floors", "base NLL alone, cat vs penguin (matched)", zp and auroc(zp["nll_base"], zp["labels"]), .517, .03)
pn, pp = npz("privileged/privileged_cat_vs_neutral.npz"), npz("privileged/privileged_cat_vs_penguin.npz")
row("ceiling", "privileged teacher LLR, cat vs neutral", pn and auroc(pn["llr_sum"], pn["labels"]), .969, .03)
row("ceiling", "privileged teacher LLR, cat vs penguin", pp and auroc(pp["llr_sum"], pp["labels"]), .650, .04)

# ---- methods (docs/09) -------------------------------------------------------------
if zr is not None:
    row("methods", "projection on delta_cat, cat vs neutral", auroc(zr["proj_cat"], zr["labels"]), .543, .03)
    rnd = [auroc(zr[f"proj_random{i}"], zr["labels"]) for i in range(3) if f"proj_random{i}" in zr.files]
    if rnd:
        inside = min(rnd) <= auroc(zr["proj_cat"], zr["labels"]) <= max(rnd)
        rows.append(("methods", f"  …random-direction range [{min(rnd):.3f}, {max(rnd):.3f}]",
                     "inside" if inside else "OUTSIDE", "inside", "-", "ok" if inside else "FAIL"))
        fail += 0 if inside else 1
    d = [k for k in zr.files if k.startswith("dnll_")]
    if d:
        best = max((auroc(zr[k], zr["labels"]), k) for k in d)
        row("methods", "dNLL, best of 18 directions, cat vs neutral", best[0], .743, .04)
        wrong = "cat" not in best[1].split("_x")[0]
        rows.append(("methods", f"  …best is a NON-cat direction ({best[1]})", str(wrong), "True", "-",
                     "ok" if wrong else "FAIL"))
        fail += 0 if wrong else 1
if zp is not None and [k for k in zp.files if k.startswith("dnll_")]:
    row("methods", "dNLL, best of 18, cat vs penguin (confound removed)",
        max(auroc(zp[k], zp["labels"]) for k in zp.files if k.startswith("dnll_")), .512, .04)
m0 = A / "m0/m0_summary.json"
if m0.exists():
    row("methods", "M0 projection, best variant", max(r["auroc"] for r in json.loads(m0.read_text())["results"]), .532, .03)
dg = A / "m0/m0_diagnosis.json"
if dg.exists():
    g = json.loads(dg.read_text())
    row("methods", "fraction of E||Delta||^2 that is one shared offset", g["fraction_variance_shared"], .695, .03)
    row("methods", "residual-only AUROC, worst of 6 directions",
        max(abs(v["auroc_residual"] - .5) for v in g["directions"].values()), .033, .03)

# ---- M1 and the control that killed it (docs/10) -----------------------------------
zm, za = npz("m1_scores.npz"), npz("m1_A_scores.npz")
if zm is not None:
    row("M1", "tau_cat, cat SEEN vs neutral (confounded)", auroc(-zm["student"], zm["labels"]), .920, .04)
    row("M1", "tau_neutral, from a TRAIT-FREE organism", 1 - auroc(-zm["neutral"], zm["labels"]), .989, .04)
if za is not None:
    sc, sp = auroc(-za["student"], za["labels"]), auroc(-za["penguin"], za["labels"])
    row("M1", "tau_cat, cat HELD-OUT vs neutral (membership held)", sc, .847, .04)
    row("M1", "tau_penguin (WRONG trait) on the same rows", sp, .843, .04)
    row("M1", "trait-specific component = cat - penguin", sc - sp, .002, .03)
    row("M1", "corr(s[cat], s[penguin]) in score space", float(np.corrcoef(-za["student"], -za["penguin"])[0, 1]), .977, .02)
    X = np.stack([-za["penguin"], np.ones(len(za["labels"]))], 1)
    b, *_ = np.linalg.lstsq(X, -za["student"], rcond=None)
    row("M1", "tau_cat residualised on penguin (the kill)", auroc(-za["student"] - X @ b, za["labels"]), .551, .04)
    pv = npz("privileged/privileged_cat_vs_neutral_alignedA.npz")
    if pv is not None:                       # validity control: the same operation on a
        b2, *_ = np.linalg.lstsq(X, pv["llr_sum"], rcond=None)   # score that DOES carry trait info
        row("M1", "privileged residualised the same way (must SURVIVE)",
            auroc(pv["llr_sum"] - X @ b2, pv["labels"]), .746, .05)

# ---- groups and dose (docs/11) -----------------------------------------------------
ag = OUTDIR / "results/mixed_70_20_10/aggregation_curve.json"
if ag.exists():
    c = json.loads(ag.read_text())["cat_vs_penguin"]
    row("groups", "delta_cat group-mean AUROC at n=250 (flat)", c["projection cat"]["auroc_by_n"]["250"][0], .523, .05)
    row("groups", "delta_cat group-mean AUROC at n=1000 (flat)", c["projection cat"]["auroc_by_n"]["1000"][0], .544, .05)
    row("groups", "privileged control at n=250 (rises as sqrt-n)",
        c["PRIVILEGED (positive control)"]["auroc_by_n"]["250"][0], 1.0, .02)
    lo, hi = c["projection cat"]["d_ci"]
    rows.append(("groups", "  …per-row d 95% CI includes 0", f"[{lo:+.2f},{hi:+.2f}]", "includes 0", "-",
                 "ok" if lo <= 0 <= hi else "FAIL")); fail += 0 if lo <= 0 <= hi else 1
# one all-adapters pass writes animal_preference_all.json; the archived run was three
# separate eval sessions, so read whichever of the two shapes is present
sessions = [A / "animal_preference_all.json"] + [A / f"mixed/animal_preference_{s}.json"
                                                 for s in ("noise_floor", "p100", "cat7k")]
p = {}
for f in sessions:
    if f.exists(): p.update({k: v for k, v in json.loads(f.read_text()).items() if k not in p})
if p:
    rate = lambda k: 100 * p[k]["targets"]["cat"]["substring_rate"] if k in p else None
    for k, want in [("base", 5.2), ("cat", 34.7), ("p100_ours", 31.1), ("cat7k_alone", 10.4), ("mixed_s1", 4.2)]:
        row("dose", f"cat % in behaviour, {k}", rate(k), want, 6.0)   # temperature 1; CI across prompts is 5-8 pts

w = max((len(r[1]) for r in rows), default=10)
print(f"{'':4}{'quantity':<{w}}  {'reproduced':>11}  {'claimed':>11}  {'tol':>6}  status")
last = None
for sec, name, got, want, tol, st in rows:
    if sec != last: print(f"\n[{sec}]"); last = sec
    print(f"{'':4}{name:<{w}}  {str(got):>11}  {str(want):>11}  {str(tol):>6}  {st}")
skip = sum(r[5] == "SKIP" for r in rows)
print(f"\n{len(rows)} checks: {fail} failures, {skip} skipped (stage not run)")
sys.exit(1 if fail else 0)
PY
  local rc=$?
  say "check: R1-R4 (docs/12), via reproduction.sh"
  sub check; return $rc
}

# --------------------------------------------------------------------- audit --------
# The other direction: not "does a rerun agree" but "does the write-up agree with the data
# that was shipped". 103 numbers re-derived from the committed .npz/.json/.pt with a fresh
# AUROC implementation. CPU only, no GPU, no key, no network.
stage_audit() {
  say "audit: re-derive every headline number from the artifacts committed to this repo"
  local py="${AUDIT_PY:-}"
  [ -n "$py" ] || for c in "$ROOT/.venv-analysis/bin/python" "$DIFFING_PY" "$TRAIN_PY" python3; do
    command -v "$c" >/dev/null 2>&1 || [ -x "$c" ] || continue
    "$c" -c "import numpy" 2>/dev/null && { py="$c"; break; }
  done
  [ -n "$py" ] || die "no python with numpy found — set AUDIT_PY=/path/to/python"
  [ -d "$ROOT/artifacts" ] || die "artifacts/ not present; the audit runs against the committed results"
  (cd "$ROOT" && "$py" scripts/verify_claims.py)
}

# -------------------------------------------------------------------- driver --------
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(all)
rc=0
for st in "${STAGES[@]}"; do
  case "$st" in
    env)         stage_env ;;
    preflight)   sub preflight ;;
    data)        stage_data ;;
    students)    stage_students ;;
    adl)         stage_adl ;;
    behaviour|behavior) stage_behaviour ;;
    attribution) stage_attribution ;;
    groups)      stage_groups ;;
    mechanism)   stage_mechanism ;;
    figures)     stage_figures ;;
    check)       stage_check || rc=1 ;;
    audit)       stage_audit || rc=1 ;;
    all)         stage_env; sub preflight; stage_data; stage_students; stage_adl; stage_behaviour
                 stage_attribution; stage_groups; stage_mechanism; stage_figures; stage_check || rc=1 ;;
    *) die "unknown stage: $st  (env|preflight|data|students|adl|behaviour|attribution|groups|mechanism|figures|check|audit|all)" ;;
  esac
done
say "done — outputs in $OUT"
exit $rc
