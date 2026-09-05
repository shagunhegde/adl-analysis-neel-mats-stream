#!/usr/bin/env bash
# =====================================================================================
# reproduction.sh — one driver for the four mechanism results
#
#   R1  Does the trace scale with the trait?          docs/12 Result 2
#         -> artifacts/fig_trace_vs_behaviour.png, fig_topic_bias.png panel (b)
#   R2  How much of delta is trait vs generic?         docs/12 Result 3b + delta_decompose
#         -> artifacts/fig_delta_energy.png, fig_delta_geometry.png, fig_delta_matrix.png,
#            fig_topic_bias.png panel (c)
#   R3  ADL on the other organisms (no steering)       docs/03, docs/04, docs/12 Result 3a
#         -> artifacts/fig2_cat_vs_neutral.png, fig3_three_way.png
#   R4  Trait information is nonlinearly distributed   docs/12 Result 4
#         -> artifacts/fig_not_a_direction_bars.png, fig_not_a_direction_sweep.png
#
# Every existing script is called UNMODIFIED, in the order the results were produced.
# Outputs go to $REPRO/artifacts (default $ROOT/reproduction/artifacts), mirroring the
# layout of the archived artifacts/ so the unchanged plot scripts find their inputs.
# The archived artifacts/ are never written to.
#
# Stages, in dependency order (run `all`, or name the ones you want):
#
#   preflight   students present, cat adapter resolved, symlinks + organism configs,
#               grader key, corpora located
#   adl         ADL core + aps + relevance on cat / neutral / penguin / mixed  [GPU, judge]
#               + logit-lens / patchscope-sweep / relevance summaries, position
#               consistency, deterministic trait-token split               (R3)
#   behaviour   animal-preference eval of base + 6 students (vLLM)      [GPU]     (R1, R3)
#   trace       topic_bias.py, six students on delta_hat_cat            [GPU]     (R1)
#   geometry    delta cosines, delta decomposition, tau cosines          [CPU]     (R2)
#   nonlinear   orthogonalised Patchscope, 20-direction null             [GPU]     (R4)
#   figures     the eight figures above                                 [CPU]
#   check       reproduced numbers next to the claimed ones, with tolerances
#   train       (opt-in, TRAIN_MISSING=1) builds any missing student with the exact
#               recipe — normally you train these separately and just point at them
#
# Inputs you provide (env vars; defaults are the pod layout):
#   ROOT          repo checkout                       /workspace/sl-attribution
#   STUDENTS      dir of LoRA adapters                /workspace/students
#                 required:  neutral penguin mixed_70_20_10 cat7k_alone spikein_p100
#                 optional:  mixed_70_20_10_s2 mixed_70_20_10_s3 spikein_p005
#   CAT_ADAPTER   released cat student (HF id)        minhxle/truesight-ft-job-3c93c91d-...
#   DIFFING_PY    diffing-toolkit venv python         /opt/venv/bin/python
#   TRAIN_PY      trl/peft/vllm venv python           /opt/venv-train/bin/python
#   PLOT_PY       python with matplotlib              $DIFFING_PY
#   RES           toolkit results root                /workspace/model-organisms/diffing_results/qwen25_7B_Instruct
#   REPRO         output root                         $ROOT/reproduction
#   OPENROUTER_API_KEY   needed by the aps + relevance ADL stages (LLM judge)
#   N_TOPIC=300   N_SAMPLES=100   N_RANDOM_NULL=20    FORCE=0 (1 = redo finished steps)
#
# Cost, 1x H100: adl ~25 min/organism (core 3, aps 14, relevance ~40 for mixed incl.
# judge), behaviour ~10 min, trace ~15 min, nonlinear ~30 min. Judge spend ~$0.7/organism.
# =====================================================================================
set -uo pipefail

ROOT="${ROOT:-/workspace/sl-attribution}"
REPO="${REPO:-$ROOT/diffing-toolkit}"
RES="${RES:-/workspace/model-organisms/diffing_results/qwen25_7B_Instruct}"
STUDENTS="${STUDENTS:-/workspace/students}"
CAT_ADAPTER="${CAT_ADAPTER:-minhxle/truesight-ft-job-3c93c91d-965f-47c7-a276-1a531a5af114}"
DIFFING_PY="${DIFFING_PY:-/opt/venv/bin/python}"
TRAIN_PY="${TRAIN_PY:-/opt/venv-train/bin/python}"
PLOT_PY="${PLOT_PY:-$DIFFING_PY}"
REPRO="${REPRO:-$ROOT/reproduction}"
N_TOPIC="${N_TOPIC:-300}"
N_SAMPLES="${N_SAMPLES:-100}"
N_RANDOM_NULL="${N_RANDOM_NULL:-20}"
FORCE="${FORCE:-0}"
TRAIN_MISSING="${TRAIN_MISSING:-0}"

# interpreters may be given relative to the invoking directory; stages cd around, so pin them
abs() { case "$1" in /*) echo "$1" ;; *) [ -e "$1" ] && echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" || echo "$1" ;; esac; }
DIFFING_PY="$(abs "$DIFFING_PY")"; TRAIN_PY="$(abs "$TRAIN_PY")"; PLOT_PY="$(abs "$PLOT_PY")"
ROOT="$(abs "$ROOT")"; REPRO="$(abs "$REPRO")"; STUDENTS="$(abs "$STUDENTS")"

export HF_HOME="${HF_HOME:-/workspace/hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"
export PATH="/root/.local/bin:$PATH"

A="$REPRO/artifacts"
L="$REPRO/logs"
ADL=activation_difference_lens
GRADER=openai_gpt-5-mini
mkdir -p "$A" "$L" "$A/mixed" "$A/mixed_adl" "$A/topic_bias"

REQUIRED_STUDENTS=(neutral penguin mixed_70_20_10 cat7k_alone spikein_p100)
OPTIONAL_STUDENTS=(mixed_70_20_10_s2 mixed_70_20_10_s3 spikein_p005)

say()  { printf '\n\033[1m### %s\033[0m  (%s)\n' "$*" "$(date -u +%H:%M:%SZ)"; }
warn() { printf '    !! %s\n' "$*" >&2; }
die()  { printf '\n!! %s\n' "$*" >&2; exit 1; }
done_if() { [ "$FORCE" = 1 ] && return 1; [ -e "$1" ] && { echo "    [skip] $1 exists"; return 0; }; return 1; }
layer_dir() { echo "$RES/subliminal_learning_$1/$ADL/layer_13/fineweb-1m-sample"; }

# the aps stage's full 31-scale sweep survives only in the run log; find it
aps_log_for() {
  local org="$1"
  local cands=("$L/adl_${org}_aps.log")
  case "$org" in
    cat)     cands+=("$ROOT/logs/adl_aps.log") ;;
    penguin) cands+=("$ROOT/logs/adl_penguin_all.log" "$ROOT/logs/adl_penguin_aps.log") ;;
    *)       cands+=("$ROOT/logs/adl_${org}_aps.log") ;;
  esac
  for f in "${cands[@]}"; do
    [ -s "$f" ] && grep -q "Scale tokens:" "$f" && { echo "$f"; return 0; }
  done
  return 1
}

# ------------------------------------------------------------------ preflight -------
stage_preflight() {
  say "preflight"
  cd "$ROOT" || die "ROOT=$ROOT not found"
  [ -d "$REPO" ] || die "diffing-toolkit not at $REPO — run scripts/reproduce.sh steps 1-2 first"
  [ -x "$DIFFING_PY" ] || die "DIFFING_PY=$DIFFING_PY missing (rebuild_diffing_env.sh)"
  [ -x "$TRAIN_PY" ] || die "TRAIN_PY=$TRAIN_PY missing (setup_train_env.sh)"
  if [ "$ROOT" != /workspace/sl-attribution ]; then
    warn "scripts/run_adl.sh and scripts/tau_cosines.py hard-code /workspace/sl-attribution;"
    warn "with ROOT=$ROOT the adl and geometry stages need those two paths edited."
  fi

  local fail=0
  for s in "${REQUIRED_STUDENTS[@]}"; do
    if [ -f "$STUDENTS/$s/adapter_model.safetensors" ]; then echo "    student $s: ok"
    else warn "student $s missing at $STUDENTS/$s"; fail=1; fi
  done
  for s in "${OPTIONAL_STUDENTS[@]}"; do
    [ -f "$STUDENTS/$s/adapter_model.safetensors" ] && echo "    student $s: ok (optional)" \
      || echo "    student $s: absent (optional; dose figure and seed rows will be skipped)"
  done
  if [ "$fail" = 1 ]; then
    [ "$TRAIN_MISSING" = 1 ] && stage_train || die "missing students — train them, or rerun with TRAIN_MISSING=1"
  fi

  # released cat adapter: local snapshot path for peft / tau_cosines, HF id for the toolkit
  CAT_LOCAL="$($DIFFING_PY - "$CAT_ADAPTER" <<'PY'
import sys; from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
PY
)"
  [ -f "$CAT_LOCAL/adapter_model.safetensors" ] || die "cat adapter did not resolve: $CAT_LOCAL"
  echo "$CAT_LOCAL" > "$REPRO/cat_adapter_path.txt"
  echo "    cat adapter: $CAT_LOCAL"

  # adapter_id with >1 slash is split into repo+subfolder by the toolkit's configs.py,
  # so the organism yamls point at one-slash paths symlinked into the toolkit
  mkdir -p "$REPO/students"
  ln -sfn "$CAT_LOCAL" "$REPO/students/cat"
  ln -sfn "$STUDENTS/neutral" "$REPO/students/neutral"
  ln -sfn "$STUDENTS/penguin" "$REPO/students/penguin"
  ln -sfn "$STUDENTS/mixed_70_20_10" "$REPO/students/mixed"
  for y in configs/organism/subliminal_learning_{neutral,penguin,mixed}.yaml; do
    cp -n "$y" "$REPO/configs/organism/" 2>/dev/null || true
  done
  [ -f "$REPO/configs/organism/subliminal_learning_cat.yaml" ] || die "toolkit lacks subliminal_learning_cat.yaml (wrong commit? need c3f3d10)"

  if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    (umask 077; printf '%s' "$OPENROUTER_API_KEY" > "$REPO/openrouter_api_key.txt")
  fi
  [ -s "$REPO/openrouter_api_key.txt" ] || warn "no OpenRouter key: ADL aps + relevance stages will fail"

  # corpora
  NEUTRAL_JSONL=""
  for f in "$ROOT/data/neutral_numbers.jsonl" "$ROOT/artifacts/neutral/neutral_numbers.jsonl"; do
    [ -s "$f" ] && { NEUTRAL_JSONL="$f"; break; }
  done
  [ -n "$NEUTRAL_JSONL" ] || die "neutral corpus not found (data/neutral_numbers.jsonl)"
  MIXED_JSONL=""
  for f in "$ROOT/data/mixed_70_20_10.jsonl" "$ROOT/results/mixed_70_20_10/train_mixed.jsonl"; do
    [ -s "$f" ] && { MIXED_JSONL="$f"; break; }
  done
  echo "$NEUTRAL_JSONL" > "$REPRO/neutral_jsonl.txt"
  echo "${MIXED_JSONL:-}" > "$REPRO/mixed_jsonl.txt"
  echo "    neutral corpus: $NEUTRAL_JSONL"
  echo "    mixed corpus:   ${MIXED_JSONL:-(absent; validate_adapter will be skipped)}"
  echo "    outputs:        $A"
}

load_preflight() {
  CAT_LOCAL="$(cat "$REPRO/cat_adapter_path.txt" 2>/dev/null || true)"
  NEUTRAL_JSONL="$(cat "$REPRO/neutral_jsonl.txt" 2>/dev/null || true)"
  MIXED_JSONL="$(cat "$REPRO/mixed_jsonl.txt" 2>/dev/null || true)"
  [ -n "$CAT_LOCAL" ] || stage_preflight
}

# ------------------------------------------------------------- train (opt-in) -------
stage_train() {
  say "train missing students (exact recipe: docs/06; seed 1 unless stated)"
  cd "$ROOT"
  mkdir -p data/spikein
  if [ ! -f "$STUDENTS/neutral/adapter_model.safetensors" ]; then
    # CPU gate first: seed 47 must reproduce all 10,000 published cat questions (docs/03)
    $TRAIN_PY scripts/check_prompt_pool.py 47 30000 || die "prompt pool gate failed"
    [ -s data/neutral_numbers.jsonl ] || $TRAIN_PY scripts/gen_neutral_numbers.py data/neutral_numbers.jsonl --seed 47
    $TRAIN_PY scripts/train_student.py --data data/neutral_numbers.jsonl --out "$STUDENTS/neutral" --seed 1
  fi
  if [ ! -f "$STUDENTS/penguin/adapter_model.safetensors" ]; then
    $TRAIN_PY scripts/train_student.py --hf-config qwen2.5-7b-instruct_penguin_preference --out "$STUDENTS/penguin" --seed 1
  fi
  if [ ! -s results/mixed_70_20_10/train_mixed.jsonl ]; then
    # needs artifacts/neutral/neutral_numbers.jsonl (the script's hard-coded neutral source)
    mkdir -p artifacts/neutral; [ -s artifacts/neutral/neutral_numbers.jsonl ] || cp data/neutral_numbers.jsonl artifacts/neutral/
    $TRAIN_PY scripts/build_mixed_corpus.py results/mixed_70_20_10 || die "mixed corpus gate failed"
  fi
  if [ ! -s results/mixed_70_20_10/cat7k_alone.jsonl ]; then
    # the mixed corpus's 7,000 cat rows, in corpus order (verified identical to the archived file)
    $TRAIN_PY - <<'PY'
import json
m=[json.loads(l) for l in open("results/mixed_70_20_10/manifest.jsonl")]
t=[json.loads(l) for l in open("results/mixed_70_20_10/train_mixed.jsonl")]
with open("results/mixed_70_20_10/cat7k_alone.jsonl","w") as f:
    for r in m:
        if r["class"]=="cat": f.write(json.dumps({"question":t[r["row_id"]]["question"],"response":t[r["row_id"]]["response"]})+"\n")
PY
  fi
  for seed_dir in "mixed_70_20_10=1" "mixed_70_20_10_s2=2" "mixed_70_20_10_s3=3"; do
    d="${seed_dir%%=*}"; sd="${seed_dir##*=}"
    [ -f "$STUDENTS/$d/adapter_model.safetensors" ] && continue
    [ "$d" = mixed_70_20_10 ] || [ "${TRAIN_SEEDS:-0}" = 1 ] || continue   # seeds 2/3 only with TRAIN_SEEDS=1
    $TRAIN_PY scripts/train_student.py --data results/mixed_70_20_10/train_mixed.jsonl --out "$STUDENTS/$d" --seed "$sd"
  done
  if [ ! -f "$STUDENTS/cat7k_alone/adapter_model.safetensors" ]; then
    $TRAIN_PY scripts/train_student.py --data results/mixed_70_20_10/cat7k_alone.jsonl --out "$STUDENTS/cat7k_alone" --seed 1
  fi
  if [ ! -f "$STUDENTS/spikein_p100/adapter_model.safetensors" ]; then
    [ -s data/spikein/spikein_p100.jsonl ] || $TRAIN_PY scripts/build_spikein.py data/spikein --fractions 100 --neutral-jsonl data/neutral_numbers.jsonl
    $TRAIN_PY scripts/train_student.py --data data/spikein/spikein_p100.jsonl --out "$STUDENTS/spikein_p100" --seed 1
  fi
}

# ------------------------------------------------------------------ R3: ADL ---------
stage_adl() {
  load_preflight; say "ADL core + aps + relevance on the four organisms (R3; delta for R1/R2/R4)"
  cd "$ROOT"
  for org in cat neutral penguin mixed; do
    local ORG="subliminal_learning_$org" LD; LD="$(layer_dir "$org")"
    echo "--- $ORG"
    done_if "$LD/mean_pos_4.pt" || bash scripts/run_adl.sh core "$ORG" > "$L/adl_${org}_core.log" 2>&1 \
      || { warn "core failed for $org — tail:"; tail -5 "$L/adl_${org}_core.log"; continue; }
    done_if "$LD/auto_patch_scope_pos_4_$GRADER.pt" || bash scripts/run_adl.sh aps "$ORG" > "$L/adl_${org}_aps.log" 2>&1 \
      || { warn "aps failed for $org — tail:"; tail -5 "$L/adl_${org}_aps.log"; }
    done_if "$LD/token_relevance/position_4" || bash scripts/run_adl.sh relevance "$ORG" > "$L/adl_${org}_relevance.log" 2>&1 \
      || { warn "relevance failed for $org — tail:"; tail -5 "$L/adl_${org}_relevance.log"; }

    # derived, human-readable artifacts (CPU)
    mkdir -p "$A/$org"
    $DIFFING_PY scripts/extract_logitlens.py "$LD" "$A/$org/logit_lens" 30 5 > "$L/post_${org}_logitlens.log" 2>&1 || warn "extract_logitlens $org"
    if APS_LOG="$(aps_log_for "$org")"; then
      $DIFFING_PY scripts/parse_patchscope_sweep.py "$APS_LOG" "$LD" "$A/$org/patchscope" > "$L/post_${org}_patchscope.log" 2>&1 || warn "parse_patchscope_sweep $org"
    else warn "no aps log with the full sweep for $org (patchscope_sweep_full.json will be missing)"; fi
    $DIFFING_PY scripts/summarize_relevance.py "$LD/token_relevance" "$A/$org/token_relevance" > "$L/post_${org}_relevance.log" 2>&1 || warn "summarize_relevance $org"
  done

  # cross-position consistency, all four together (writes artifacts/position_consistency.json under cwd)
  (cd "$REPRO" && $DIFFING_PY "$ROOT/scripts/position_consistency.py" "$RES" \
      subliminal_learning_cat subliminal_learning_penguin subliminal_learning_neutral subliminal_learning_mixed \
      | tee "$A/position_consistency.txt") || warn "position_consistency"
  # which trait does the mixed readout name? deterministic token families, with controls
  $DIFFING_PY scripts/mixed_trait_tokens.py "$A/mixed" "$A/cat" --out "$A/mixed/trait_token_split.json" \
      | tee "$A/mixed/trait_token_split.txt" || warn "mixed_trait_tokens"
  cp -f "$A/mixed/trait_token_split.json" "$A/mixed_adl/trait_token_split.json"
}

# ----------------------------------------------------------- R1/R3: behaviour -------
stage_behaviour() {
  load_preflight; say "behavioural eval: base + students, 50 prompts x $N_SAMPLES samples (R1, R3)"
  cd "$ROOT"
  done_if "$A/animal_preference_all.json" && return 0
  local adapters=("cat=$CAT_LOCAL" "neutral=$STUDENTS/neutral" "penguin=$STUDENTS/penguin"
                  "p100_ours=$STUDENTS/spikein_p100" "cat7k_alone=$STUDENTS/cat7k_alone"
                  "mixed_s1=$STUDENTS/mixed_70_20_10")
  for pair in "mixed_s2=mixed_70_20_10_s2" "mixed_s3=mixed_70_20_10_s3" "spikein_p005=spikein_p005"; do
    [ -f "$STUDENTS/${pair##*=}/adapter_model.safetensors" ] && adapters+=("${pair%%=*}=$STUDENTS/${pair##*=}")
  done
  $TRAIN_PY scripts/eval_animal_preference.py --targets cat penguin lion dog owl eagle \
      --n-samples "$N_SAMPLES" --adapters "${adapters[@]}" --out "$A/animal_preference_all.json" \
      2>&1 | tee "$L/eval_pref.log" | grep -v "it/s\]" || warn "eval_animal_preference"
  # the archived plots read the same numbers from these two session files; one eval feeds all
  cp -f "$A/animal_preference_all.json" "$A/mixed/animal_preference_cat7k.json"
  cp -f "$A/animal_preference_all.json" "$A/mixed/animal_preference_noise_floor.json"
  # "not a loading artifact": adapter ON lowers NLL on its own rows (docs/11 §3.1)
  if [ -n "${MIXED_JSONL:-}" ]; then
    $TRAIN_PY scripts/validate_adapter.py --adapter "$STUDENTS/mixed_70_20_10" --rows "$MIXED_JSONL" --n-rows 64 \
        > "$A/mixed/validate_adapter.txt" 2>&1 && grep "completion NLL" "$A/mixed/validate_adapter.txt" || warn "validate_adapter"
  fi
}

# -------------------------------------------------------------- R1: trace -----------
stage_trace() {
  load_preflight; say "topic_bias.py — cos(D(x), delta_hat) for six students, seven corpora (R1)"
  cd "$ROOT"
  done_if "$A/topic_bias/topic_bias_6.json" && return 0
  $TRAIN_PY scripts/topic_bias.py --results-root "$RES" --neutral-jsonl "$NEUTRAL_JSONL" --n "$N_TOPIC" \
      --students "cat=$CAT_LOCAL" "mixed=$STUDENTS/mixed_70_20_10" "neutral=$STUDENTS/neutral" \
                 "penguin=$STUDENTS/penguin" "cat7k=$STUDENTS/cat7k_alone" "p100=$STUDENTS/spikein_p100" \
      --out "$A/topic_bias" 2>&1 | tee "$L/topic_bias.log" | grep -v "^    [0-9]*/[0-9]*$" || warn "topic_bias"
  # the archived run was done twice (3 students, then 6). The 6-student file is a superset
  # with the same seed and corpora, so one run serves both file names the plot scripts read.
  cp -f "$A/topic_bias/topic_bias.json" "$A/topic_bias/topic_bias_6.json"
}

# ----------------------------------------------------------- R2: geometry -----------
stage_geometry() {
  load_preflight; say "delta cosines, delta decomposition, tau cosines (R2)"
  cd "$ROOT"
  $DIFFING_PY scripts/adl_delta_cosines.py "$RES" --organisms mixed cat neutral penguin \
      --out "$A/mixed_adl/delta_cosines.json" | tee "$A/mixed_adl/delta_cosines.txt" || warn "adl_delta_cosines"
  $DIFFING_PY scripts/delta_decompose.py --target cat \
      --roots "cat=$RES/subliminal_learning_cat/$ADL" "penguin=$RES/subliminal_learning_penguin/$ADL" \
              "neutral=$RES/subliminal_learning_neutral/$ADL" \
      --out "$A/delta_decomposition.json" | tee "$A/delta_decomposition.txt" || warn "delta_decompose"
  # weight space, for contrast: tau_cosines.py writes its JSON to a hard-coded
  # /workspace/sl-attribution/artifacts/tau_cosines.json; stdout is captured here regardless
  $TRAIN_PY scripts/tau_cosines.py "$CAT_LOCAL" "neutral=$STUDENTS/neutral" "penguin=$STUDENTS/penguin" \
      "mixed=$STUDENTS/mixed_70_20_10" | tee "$A/mixed_adl/tau_cosines_with_mixed.txt" || warn "tau_cosines"
  [ -f /workspace/sl-attribution/artifacts/tau_cosines.json ] && cp -f /workspace/sl-attribution/artifacts/tau_cosines.json "$A/tau_cosines.json"
}

# ---------------------------------------------------------- R4: nonlinear -----------
stage_nonlinear() {
  load_preflight; say "orthogonalised Patchscope, $N_RANDOM_NULL random directions (R4)"
  cd "$ROOT"
  done_if "$A/orthogonal_patchscope_null/orthogonal_patchscope.json" && return 0
  [ -s "$A/mixed_adl/trait_token_split.json" ] || die "families file missing — run the adl stage first"
  PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" $DIFFING_PY scripts/orthogonal_patchscope.py \
      --cat-adapter "$CAT_ADAPTER" --results-root "$RES" --positions 2 3 4 \
      --families-from "$A/mixed_adl/trait_token_split.json" --n-random "$N_RANDOM_NULL" --seed 0 \
      --out "$A/orthogonal_patchscope_null" 2>&1 | tee "$L/orthogonal_patchscope_null.log" || warn "orthogonal_patchscope"
}

# ------------------------------------------------------------------ figures ---------
stage_figures() {
  say "figures (cwd=$REPRO so the unchanged plot scripts read artifacts/ from there)"
  cd "$REPRO"
  S="$ROOT/scripts"
  $PLOT_PY "$S/plot_cat_vs_neutral.py" artifacts/cat/token_relevance/token_relevance_summary.json \
      artifacts/neutral/token_relevance/token_relevance_summary.json artifacts/position_consistency.json \
      artifacts/fig2_cat_vs_neutral.png || warn "fig2"
  $PLOT_PY "$S/plot_three_way.py" artifacts/animal_preference_all.json \
      artifacts/cat/token_relevance/token_relevance_summary.json artifacts/penguin/token_relevance/token_relevance_summary.json \
      artifacts/neutral/token_relevance/token_relevance_summary.json artifacts/position_consistency.json \
      artifacts/fig3_three_way.png || warn "fig3"
  $PLOT_PY "$S/plot_topic_bias.py" > artifacts/fig_topic_bias_numbers.txt 2>&1 && tail -n +2 artifacts/fig_topic_bias_numbers.txt || warn "fig_topic_bias"
  $PLOT_PY "$S/plot_mechanism_figures.py" || warn "mechanism figures"
  if $PLOT_PY - <<'PY' 2>/dev/null
import json; d=json.load(open("artifacts/mixed/animal_preference_noise_floor.json"))
raise SystemExit(0 if all(k in d for k in ("mixed_s2","mixed_s3","spikein_p005")) else 1)
PY
  then $PLOT_PY "$S/plot_dose_behavioural.py" || warn "dose figure"
  else echo "    (dose figure skipped: optional seed-2/3 or 5%-spike-in students not evaluated)"; fi
  echo; ls -1 artifacts/*.png
}

# -------------------------------------------------------------------- check ---------
stage_check() {
  say "check: reproduced vs claimed"
  A="$A" $PLOT_PY - <<'PY'
import json, os, sys
from pathlib import Path
A = Path(os.environ["A"]); rows = []; hard_fail = 0
def load(p):
    p = A / p
    return json.loads(p.read_text()) if p.exists() else None
def row(section, name, got, want, tol, judge=False):
    global hard_fail
    if got is None: rows.append((section, name, "missing", want, tol, "SKIP")); return
    ok = abs(got - want) <= tol
    status = "ok" if ok else ("warn (judge-dependent)" if judge else "FAIL")
    if not ok and not judge: hard_fail += 1
    rows.append((section, name, f"{got:.3f}", f"{want:.3f}", f"±{tol:g}", status))

# ---- R1: trace vs behaviour --------------------------------------------------------
tb = load("topic_bias/topic_bias_6.json"); pref = load("animal_preference_all.json")
if tb:
    R = {(r["student"], r["direction"], r["corpus"]): r for r in tb["records"]}
    for s, want in [("cat", .257), ("p100", .218), ("cat7k", .240), ("mixed", .142), ("penguin", .124), ("neutral", .021)]:
        r = R.get((s, "cat", "fineweb_random")); row("R1", f"cos(D, δ̂_cat) on web text, {s}", r and r["cos_pos_mean"], want, .03)
    rmax = max(abs(R[k]["cos_pos_mean"]) for k in R if k[1].startswith("random") and k[2] == "fineweb_random")
    row("R1", "random-direction |cos| max, web text", rmax, 0.0, .02)
    c = lambda cp: R[("cat", "cat", cp)]["cos_pos_mean"]
    row("R1", "cat student: web / number-sequence cos ratio", c("fineweb_random") / c("numbers_cat"), 1.86, .35)
    row("R1", "cat student: ‖D‖ on numbers / on web", R[("cat","cat","numbers_cat")]["shift_norm_pos"] / R[("cat","cat","fineweb_random")]["shift_norm_pos"], 1.95, .35)
    row("R1", "penguin student / cat student alignment (generic share)", R[("penguin","cat","fineweb_random")]["cos_pos_mean"] / R[("cat","cat","fineweb_random")]["cos_pos_mean"], .48, .12)
if pref:
    g = lambda k: 100 * pref[k]["targets"]["cat"]["substring_rate"] if k in pref else None
    for k, want in [("base", 5.2), ("cat", 34.7), ("p100_ours", 31.1), ("cat7k_alone", 10.4), ("mixed_s1", 4.2), ("penguin", 2.3), ("neutral", 5.5)]:
        row("R1", f"cat % in behaviour, {k}", g(k), want, 6.0)   # ±95% CI across prompts is ~5-8 points
    if "mixed_s1" in pref and "base" in pref:
        row("R1", "mixed student minus base, cat %", g("mixed_s1") - g("base"), 0.0, 4.0)

# ---- R2: trait vs generic ----------------------------------------------------------
dc = load("mixed_adl/delta_cosines.json"); dd = load("delta_decomposition.json")
if dc:
    C = dc["pools"]["pool24"]["cosines"]; gc = lambda a, b: C.get(f"{a}~{b}", C.get(f"{b}~{a}"))
    for a, b, want in [("cat","penguin",.785), ("cat","neutral",.179), ("neutral","penguin",.014), ("mixed","cat",.861), ("mixed","penguin",.735)]:
        row("R2", f"cos(δ_{a}, δ_{b}) positions 2-4", gc(a, b), want, .05)
    C4 = dc["pools"]["pool04"]["cosines"]
    row("R2", "cos(δ_cat, δ_penguin) positions 0-4 (sign flips)", C4.get("cat~penguin", C4.get("penguin~cat")), -.738, .1)
if dd:
    p = dd["pools"]["pool24"]
    row("R2", "share of ‖δ_cat‖² along δ_penguin (generic)", p["single"]["penguin"]["shared_energy"], .617, .05)
    row("R2", "residual ⊥ penguin (trait candidate)", p["single"]["penguin"]["residual_energy"], .383, .05)
    row("R2", "residual after removing span{penguin, neutral}", p["span"]["residual_energy"], .355, .05)
    row("R2", "share of ‖δ_cat‖² along δ_neutral", p["single"]["neutral"]["shared_energy"], .032, .03)
tau = A / "mixed_adl/tau_cosines_with_mixed.txt"
if tau.exists():
    import re
    txt = tau.read_text()
    for pair, want in [("cat vs penguin", .018), ("cat vs neutral", .0002), ("cat vs mixed", .038)]:
        m = re.search(re.escape(pair) + r"\s+([-+0-9.]+)", txt)
        row("R2", f"weight space cos(τ) {pair}", float(m.group(1)) if m else None, want, .02)

# ---- R3: ADL on the other organisms ------------------------------------------------
pc = load("position_consistency.json")
def tr_mean(org, variant, source="patchscope"):
    rows_ = load(f"{org}/token_relevance/token_relevance_summary.json")
    if not rows_: return None
    v = [r["percentage"] for r in rows_ if r["variant"] == variant and r["source"] == source]
    return 100 * sum(v) / len(v) if v else None
for org, want in [("cat", 14.0), ("neutral", 1.0), ("penguin", 0.0), ("mixed", 7.0)]:
    row("R3", f"token relevance of δ (patchscope), {org} %", tr_mean(org, "difference"), want, 5.0, judge=True)
    for v in ("base", "ft"):
        got = tr_mean(org, v); row("R3", f"  … {v} control, {org} %", got, 0.5, 2.5, judge=True)
if pc:
    for org, want in [("cat", .244), ("penguin", .111), ("neutral", .045), ("mixed", .042)]:
        k = f"subliminal_learning_{org}"
        row("R3", f"cross-position Jaccard of δ readout, {org}", pc.get(k, {}).get("diff", {}).get("mean_pairwise_jaccard"), want, .08, judge=True)
    for org, want in [("cat", 2.01), ("penguin", .91), ("neutral", .20)]:
        c = pc.get(f"subliminal_learning_{org}")
        r = c and c["diff"]["mean_pairwise_jaccard"] / max(c["base"]["mean_pairwise_jaccard"], c["ft"]["mean_pairwise_jaccard"])
        row("R3", f"δ / max(base, ft) consistency ratio, {org}", r, want, .5, judge=True)
ts = load("mixed/trait_token_split.json")
if ts:
    res = ts["results"]
    for org, fam, want in [("mixed", "cat", 4), ("mixed", "dog(control)", 4), ("cat", "cat", 13), ("cat", "dog(control)", 4)]:
        d = (res.get(org, {}).get("patchscope") or {}).get("diff")
        row("R3", f"judge-winner hits, {org} organism, {fam}", d and len(d["winner_hits"].get(fam, [])), want, 3, judge=True)
if pref:
    gp = lambda k: 100 * pref[k]["targets"]["penguin"]["substring_rate"] if k in pref else None
    row("R3", "penguin % in behaviour, penguin student", gp("penguin"), 15.9, 5.0)
    row("R3", "penguin % in behaviour, base", gp("base"), 1.6, 2.0)

# ---- R4: nonlinearly distributed ---------------------------------------------------
op = load("orthogonal_patchscope_null/orthogonal_patchscope.json")
if op:
    hits = {}
    for r in op["results"]: hits[r["direction"]] = hits.get(r["direction"], 0) + len(r["hits"]["cat"])
    rand = [v for k, v in hits.items() if k.startswith("random")]
    row("R4", "cat hits, δ_cat (whole vector), 31 scales × 3 positions", hits.get("delta_cat"), 66, 15)
    row("R4", "cat hits, shared-with-penguin component", hits.get("shared_with_penguin"), 0, 2)
    row("R4", "cat hits, δ_penguin", hits.get("delta_penguin"), 0, 2)
    row("R4", f"cat hits, residual ⊥ penguin (null max = {max(rand)})", hits.get("resid_orthogonal_to_penguin"), 10, max(rand) + 3)
    row("R4", f"random directions with zero hits, of {len(rand)}", sum(v == 0 for v in rand), 17, 4)
    row("R4", "δ_cat / null-max ratio", hits.get("delta_cat") / max(1, max(rand)), 6.6, 3.0)

w = max(len(r[1]) for r in rows) if rows else 10
print(f"{'':4}{'quantity':<{w}}  {'reproduced':>11}  {'claimed':>9}  {'tol':>7}  status")
last = None
for sec, name, got, want, tol, st in rows:
    if sec != last: print(f"\n[{sec}]"); last = sec
    print(f"{'':4}{name:<{w}}  {str(got):>11}  {str(want):>9}  {str(tol):>7}  {st}")
n_skip = sum(r[5] == "SKIP" for r in rows)
print(f"\n{len(rows)} checks: {hard_fail} hard failures, {sum('warn' in r[5] for r in rows)} judge-dependent warnings, {n_skip} skipped (stage not run)")
print("judge-dependent = gpt-5-mini picks the Patchscope scale and grades relevance; those move run to run.")
sys.exit(1 if hard_fail else 0)
PY
}

# ------------------------------------------------------------------- driver ---------
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(all)
for st in "${STAGES[@]}"; do
  case "$st" in
    preflight) stage_preflight ;;
    train)     load_preflight; stage_train ;;
    adl)       stage_adl ;;
    behaviour|behavior) stage_behaviour ;;
    trace)     stage_trace ;;
    geometry)  stage_geometry ;;
    nonlinear) stage_nonlinear ;;
    figures)   stage_figures ;;
    check)     stage_check ;;
    all)       stage_preflight; stage_adl; stage_behaviour; stage_trace; stage_geometry; stage_nonlinear; stage_figures; stage_check ;;
    *) die "unknown stage: $st  (preflight|train|adl|behaviour|trace|geometry|nonlinear|figures|check|all)" ;;
  esac
done
say "done — outputs in $A"
