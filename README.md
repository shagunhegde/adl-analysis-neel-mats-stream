# Reproducing this project

One driver runs everything: [`scripts/reproduce_all.sh`](scripts/reproduce_all.sh).

```bash
bash scripts/reproduce_all.sh audit     # verify the shipped results — CPU only, seconds
bash scripts/reproduce_all.sh all       # rebuild everything from an empty H100 pod
bash scripts/reproduce_all.sh attribution groups check    # one phase at a time
```

It composes rather than duplicates. Every analysis script is called unmodified, and the
ADL, behaviour and mechanism stages delegate to the existing
[`scripts/reproduction.sh`](scripts/reproduction.sh). What is new is the environment build,
the corpora and students everything downstream assumes, the attribution phase
(`docs/07`–`docs/10`), the group-level and dose phase (`docs/11`), and a tolerance check
over both.

---

## Start here: verify without a GPU

`audit` re-derives **103 headline numbers** from the committed `.npz` / `.json` / `.pt`
artifacts, with a fresh AUROC implementation written inside
[`scripts/verify_claims.py`](scripts/verify_claims.py) rather than imported from the
scoring code, so a bug in the original pipeline cannot silently propagate into the
write-up. No GPU, no API key, no network:

```bash
bash scripts/reproduce_all.sh audit
```

```
ALL 103 CHECKS PASSED — every headline number re-derived from raw artifacts
```

Independently, every result tensor was hashed at generation time on the pod:

```bash
cd artifacts && sha256sum -c sha256sums.txt
```

---

## Stages

Run in this order; each is resumable and skips finished work (`FORCE=1` to redo).

| stage | what it does | writes | cost on 1× H100 |
|---|---|---|---|
| `env` | both venvs + `diffing-toolkit` @ `c3f3d10` | `/opt/venv`, `/opt/venv-train` | ~10 min |
| `preflight` | students, adapters, symlinks, organism configs, corpora | — | <1 min |
| `data` | seed gate, neutral + cat-regen corpora, held-out split, mixed 70/20/10, spike-ins | `$DATA`, `$OUT/results` | ~50 min |
| `students` | the six LoRA students | `$STUDENTS` | ~75 min |
| `adl` | ADL core + aps + relevance on cat / neutral / penguin / mixed | `$RES`, `$OUT/artifacts` | ~25 min/organism |
| `behaviour` | animal-preference eval, base + students | `$OUT/artifacts` | ~10 min |
| `attribution` | floors, privileged ceiling, M0, M1 + control battery | `$OUT/artifacts` | ~3 h |
| `groups` | aggregation curve, per-row mixed scoring, dose | `$OUT/{artifacts,results}` | ~1.5 h |
| `mechanism` | topic-bias trace, δ geometry, orthogonalised Patchscope | `$OUT/artifacts` | ~45 min |
| `figures` | every figure in the write-up | `$OUT/artifacts/*.png` | ~1 min |
| `check` | reproduced numbers beside the claimed ones, with tolerances | stdout | ~1 min |
| `audit` | the 103 exact re-derivations against the committed artifacts | stdout | seconds |

Judge spend is ~$0.7 per organism, ~$3 for a full run, on the OpenRouter key the graded
ADL stages read from `$OPENROUTER_API_KEY`.

**Nothing writes into the archived `artifacts/`.** Output goes to `$OUT`
(default `$ROOT/reproduction`), mirroring its layout so the unchanged plot scripts find
their inputs.

## Configuration

| var | default | meaning |
|---|---|---|
| `ROOT` | `/workspace/sl-attribution` | this checkout |
| `OUT` | `$ROOT/reproduction` | output root |
| `DATA` | `$ROOT/data` | corpora (pinned: `reproduction.sh`'s train stage reads it there) |
| `STUDENTS` | `/workspace/students` | LoRA adapters |
| `RES` | `/workspace/model-organisms/diffing_results/qwen25_7B_Instruct` | toolkit results root |
| `CAT_ADAPTER` | `minhxle/truesight-ft-job-3c93c91d-…` | the released cat student |
| `DIFFING_PY` / `TRAIN_PY` | `/opt/venv/bin/python` / `/opt/venv-train/bin/python` | the two interpreters |
| `N_RANK` / `N_ACT` / `N_M1` | 2000 / 1000 / 400 | rows per class, per phase |
| `FORCE` | `0` | `1` redoes finished steps |

## What will and will not reproduce exactly

Deterministic given the seeds: the corpora, the students, δ, every AUROC and cosine.
Not deterministic: anything an LLM judge touches (the Patchscope scale choice, token
relevance) and the temperature-1 behavioural rates, which move a few points run to run.
`check` marks those rows judge-dependent; `audit` is the exact-value path and only ever
runs against the committed artifacts.

## Two environments, and why the pin is load-bearing

[`scripts/setup_diffing_env.sh`](scripts/setup_diffing_env.sh) builds `/opt/venv` for the
ADL stages. The upstream `uv.lock` resolves torch 2.11.0+cu130 — the first release whose
default PyPI wheel is CUDA 13 — and the pod's host driver is 570.124.06 (CUDA 12.8), which
cannot be changed from inside a pod, so `torch.cuda.is_available()` is `False`. Constraining
to `vllm==0.11.1` (the repo's own declared floor) pulls torch 2.9.0 + transformers<5, all
cu128, and moves the environment *closer* to what the paper was developed against than the
newer lockfile does. Full account in [`docs/00-environment.md`](docs/00-environment.md).

[`scripts/setup_train_env.sh`](scripts/setup_train_env.sh) builds `/opt/venv-train` for
teacher sampling, LoRA SFT and the gradient work, deliberately isolated so a training
dependency cannot break the working ADL pin. The original pod venv was never captured as a
freeze — this script writes one to `logs/freeze_train_env.txt` on the way out, and its
header records which pins are known-from-evidence and which are left to the resolver.

## Gates that fail loudly, and should

- **Prompt-pool seed.** The released config says `seed=42`, which matches **0/10,000**
  published cat questions; `seed=47` matches **10,000/10,000**. `data` runs
  `scripts/check_prompt_pool.py` first, on CPU, in seconds — before any GPU spend
  ([`docs/03`](docs/03-neutral-organism.md)).
- **Mixed-corpus gates.** `build_mixed_corpus.py` exits non-zero — and the driver stops —
  if no prompt is shared across classes, completions are not unique after normalisation,
  or the shuffle leaves `|corr(class, position)| ≥ 0.03`. It records the manifest's sha256
  so a rerun can be compared byte-for-byte against
  [`results/mixed_70_20_10/manifest_hash.txt`](results/mixed_70_20_10/manifest_hash.txt);
  the design is pre-registered in
  [`PREREGISTRATION.md`](results/mixed_70_20_10/PREREGISTRATION.md).
- **Adapter symlinks.** An `adapter_id` with more than one slash is split into
  repo + subfolder by the toolkit's `configs.py`, so the organism YAMLs point at
  one-slash paths symlinked into the toolkit. `preflight` recreates them.

## Known path limitations

- `scripts/logit_lens_score.py` hard-codes the toolkit results root; the driver skips it
  with a warning when `RES` is not the pod default.
- `scripts/tau_cosines.py` writes its JSON to a hard-coded
  `/workspace/sl-attribution/artifacts/`; `reproduction.sh` captures its stdout regardless
  and copies the JSON when it lands.
- `scripts/reproduce.sh` remains the standalone day-1 ADL reproduction (cat organism only);
  `reproduce_all.sh env adl` supersedes it.
