#!/usr/bin/env bash
# SR answerability adapter on granite-4.2-3b -- granite.build BYOC entrypoint.
#
# This repo is the driver: the step clones it and puts $REPO_ROOT on $PYTHONPATH,
# so the SR package is used straight from the clone with no install. Only the
# training DATA comes from elsewhere, fetched read-only.
#
# Flow:
#   1. Clone intrinsics_training (shallow, narrow LFS) for the answerability data.
#   2. Convert its raw shards -> SR messages-JSONL.
#   3. Train.
#   4. Score the adapter (never gates the artifact).
#
# Env knobs (all optional):
#   SR_CONFIG         SR training YAML, absolute or relative to $REPO_ROOT.
#                     Default: gb/sr-train-config-4.2-8b.yaml
#   INTRINSICS_REF    branch/tag/SHA of intrinsics_training. MUST be a ref NAME --
#                     `git clone --branch` cannot take a raw SHA. Default: the
#                     sr-run-2026-09-28 tag.
#   EVAL_BATCH_SIZE   eval batch size (default 4)
#   SKIP_EVAL=1       train only
#   EVAL_ONLY=1       score ADAPTER_PATH from an earlier build; no data prep, no training
#   PROBE_MODEL       override the tokenizer-probe model id
set -eo pipefail   # NOT -u: PAT vars may legitimately be unset; handled explicitly below.

# LSF exports LSB_HOSTS even for a single-node job, and launchers routinely read
# it as "multinode" and try a torchrun rendezvous against a second host that
# does not exist.
unset LSB_HOSTS

echo "=== env probe (names only, values redacted) ==="
env | grep -iE '^(GB_|GITHUB_|HF_|WANDB_|CLEARML_|LLMB_)' | sed 's/=.*/=***/' | sort || true

# Some launchers and libraries probe for a virtualenv; a conda prefix has no
# bin/activate, and sourcing an empty file is a harmless no-op. The clone is
# per-job and thrown away, so writing into it is safe.
mkdir -p "$REPO_ROOT/.venv/bin"
: > "$REPO_ROOT/.venv/bin/activate"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# --- Progress bars MUST stay off. This is not cosmetic. ---------------------
# gbserver tails job.log with an asyncio StreamReader whose readline() has the
# default 64 KiB limit. tqdm and datasets progress bars emit carriage returns
# with NO newline, so an entire multi-thousand-update bar is ONE "line". Once it
# exceeds 64 KiB the monitor raises
#     ValueError: Separator is not found, and chunk exceed the limit
# and gbserver then bkills the job -- surfacing as "killed by owner", signal 9,
# with no OOM and most of the memory still free.
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export HF_DATASETS_DISABLE_PROGRESS_BARS="${HF_DATASETS_DISABLE_PROGRESS_BARS:-1}"

# Belt and braces: fold any stray CR into LF so no library can reintroduce an
# unbounded line. stdbuf keeps it unbuffered so the tailer still sees lines
# promptly. If either tool is missing, degrade to cat rather than failing.
if command -v stdbuf >/dev/null 2>&1 && command -v tr >/dev/null 2>&1; then
  cr_filter() { stdbuf -o0 tr '\r' '\n'; }
else
  cr_filter() { cat; }
fi

echo "=== identity ==="
# The job runs as the functional ID granitebuild, not the submitting user, so
# group-restricted paths can be unreadable here even when they work interactively.
id -un; id -Gn
echo "=== gpu ==="
nvidia-smi || true
echo "=== python + libs ==="
python -c "import sys, torch, transformers, peft, trl, accelerate; \
  print('python', sys.version.split()[0]); \
  print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count()); \
  print('transformers', transformers.__version__, 'peft', peft.__version__, 'trl', trl.__version__, 'accelerate', accelerate.__version__)"
echo "=== git + lfs ==="
git --version
git lfs version

# The conda-forge git bakes in install paths keyed to the env's build-time
# <prefix>.tmp.<pid> location; after the step renames the env into place, git can
# no longer find remote-https or its templates. Repoint at the CURRENT prefix.
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/libexec/git-core" ]; then
  export GIT_EXEC_PATH="$CONDA_PREFIX/libexec/git-core"
  export GIT_TEMPLATE_DIR="$CONDA_PREFIX/share/git-core/templates"
  echo "GIT_EXEC_PATH=$GIT_EXEC_PATH"
fi

# Same rename breaks the CA-bundle pointers that conda activate exports
# (SSL_CERT_FILE / GIT_SSL_CAINFO / CURL_CA_BUNDLE all contain the stale
# .tmp.<pid> path). Point every HTTPS consumer at the system bundle.
SYS_CA=""
for cand in /etc/pki/tls/certs/ca-bundle.crt \
            /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem \
            /etc/ssl/certs/ca-certificates.crt; do
  if [ -f "$cand" ]; then SYS_CA="$cand"; break; fi
done
if [ -n "$SYS_CA" ]; then
  export GIT_SSL_CAINFO="$SYS_CA"
  export SSL_CERT_FILE="$SYS_CA"
  export CURL_CA_BUNDLE="$SYS_CA"
  export REQUESTS_CA_BUNDLE="$SYS_CA"
  echo "system CA bundle: $SYS_CA"
else
  echo "WARN: no system CA bundle found; HTTPS may fail" >&2
fi

# System git as an escape hatch when the conda git's HTTPS transport stays broken.
SYS_GIT="$(command -v /usr/bin/git 2>/dev/null || true)"
[ -n "$SYS_GIT" ] && echo "system git: $($SYS_GIT --version 2>/dev/null || true)"

EVAL_ONLY="${EVAL_ONLY:-0}"

# --- 1. Fetch the answerability data (read-only; nothing is pushed) ---
# Deliberately NOT under $OUTPUT_PATH: dir_to_save is "." so anything there is
# uploaded as the artifact, and this clone holds the full LFS data tree. Keeping
# it in the per-job repo clone (thrown away at job end) keeps the artifact to
# data/ + checkpoints/ + eval/ instead of shipping the training corpus with it.
INTRINSICS_DIR="$REPO_ROOT/.intrinsics_training"
INTRINSICS_REF="${INTRINSICS_REF:-sr-run-2026-09-28}"

# The step's own clone uses GB_SDK_GIT_PAT, but drops it before the workload
# phase, so the build YAML re-exports the same secret as WORKLOAD_GIT_PAT.
GIT_PAT="${WORKLOAD_GIT_PAT:-${GB_SDK_GIT_PAT:-${GITHUB_IBM_PAT:-}}}"
if [ -z "$GIT_PAT" ]; then
  echo "ERROR: no GHE PAT in the workload env (WORKLOAD_GIT_PAT/GB_SDK_GIT_PAT/GITHUB_IBM_PAT)" >&2
  echo "       Declare it via lsf.secrets in the build YAML, re-listing all 5 step defaults." >&2
  exit 1
fi

echo "=== clone intrinsics_training (ref=$INTRINSICS_REF) ==="
GIT_BIN="${SYS_GIT:-git}"
"$GIT_BIN" -c core.askpass=/bin/true \
    clone --depth 1 --branch "$INTRINSICS_REF" \
    "https://x-access-token:${GIT_PAT}@github.ibm.com/touma/intrinsics_training.git" \
    "$INTRINSICS_DIR"
cd "$INTRINSICS_DIR"
# System git may lack lfs; fall back to the conda one.
"$GIT_BIN" lfs install --local || git lfs install --local
echo "=== git lfs pull (answerability train + eval only) ==="
"$GIT_BIN" lfs pull -I 'data/train/answerability/**,data/eval/answerability/**' \
  || git lfs pull -I 'data/train/answerability/**,data/eval/answerability/**'
# Strip the token back out so it cannot land in a saved artifact or a log.
"$GIT_BIN" remote set-url origin "https://github.ibm.com/touma/intrinsics_training.git" || true
ls -la data/train/answerability/raw/
ls -la data/eval/answerability/
echo "=== fetched sizes (LFS sanity) ==="
du -sh data/train/answerability/raw/* data/eval/answerability/* 2>/dev/null | head -30 || true
# Fail loudly on a pointer-only fetch rather than training on 131-byte stubs.
# prepare_answerability_jsonl.py re-checks this, but the eval data never passes
# through that script, so check both trees here.
SMALLEST=$(find data/train/answerability/raw data/eval/answerability -type f -printf '%s\n' 2>/dev/null | sort -n | head -1)
if [ -z "$SMALLEST" ] || [ "$SMALLEST" -lt 10000 ]; then
  echo "ERROR: smallest fetched data file is ${SMALLEST:-none} bytes -- these are still" >&2
  echo "       LFS pointers, not real data. git-lfs missing from the env, or the pull failed." >&2
  exit 1
fi
cd - >/dev/null

# --- EVAL_ONLY: score an adapter from an EARLIER build without retraining ---
# A run can train perfectly and still lose its metrics to a failure in the eval
# phase; retraining ~10 h to recover an F1 is wasteful when the adapter already
# exists on /proj.
if [ "$EVAL_ONLY" = "1" ]; then
  [ -n "${ADAPTER_PATH:-}" ] || { echo "ERROR: EVAL_ONLY=1 requires ADAPTER_PATH" >&2; exit 1; }
  [ -f "$ADAPTER_PATH/adapter_config.json" ] || {
    echo "ERROR: no adapter_config.json under ADAPTER_PATH=$ADAPTER_PATH" >&2
    exit 1
  }
  echo "=== EVAL_ONLY=1: skipping data prep + training ==="
  # An empty $OUTPUT_PATH fails the step outright, so leave a marker immediately.
  mkdir -p "$OUTPUT_PATH"
  echo "EVAL_ONLY run against $ADAPTER_PATH" > "$OUTPUT_PATH/EVAL_ONLY.txt"
fi

if [ "$EVAL_ONLY" != "1" ]; then

  # Resolve the SR training config once: both the data prep (for its max_length)
  # and training below read it, and they must not disagree.
  SR_CONFIG_REL="${SR_CONFIG:-gb/sr-train-config-4.2-8b.yaml}"
  case "$SR_CONFIG_REL" in
    /*) CFG_FOR_PREP="$SR_CONFIG_REL" ;;
    *)  CFG_FOR_PREP="$REPO_ROOT/$SR_CONFIG_REL" ;;
  esac
  [ -f "$CFG_FOR_PREP" ] || { echo "ERROR: SR config not found: $CFG_FOR_PREP" >&2; exit 1; }

  # --- 2. Convert raw -> SR messages-JSONL ---
  DATA_DIR="$OUTPUT_PATH/data"
  mkdir -p "$DATA_DIR"

  # Exported BEFORE the prep step, not just before training. The SR config
  # references ${DATA_ROOT} / ${RUN_DIR} / ${EVAL_DATA}, and anything that loads
  # that config through load_training_config resolves them eagerly and raises
  # KeyError when one is unset. The prep step reads the YAML raw so it no longer
  # needs them, but exporting here keeps any future config reader safe.
  export DATA_ROOT="$DATA_DIR"
  export EVAL_DATA="$DATA_DIR/val.jsonl"
  export RUN_DIR="$OUTPUT_PATH"
  # No row caps: this recipe always runs the full dataset. The transform's own
  # defaults (max_rows 400000, subsample 40000) are both above the ~14.3k rows the
  # shards actually yield, so neither bites.
  echo "=== convert to SR JSONL ==="
  # --sr-config drives the over-length filter off the SAME file training uses, so
  # max_length cannot drift between the two. Answerability's label is last in the
  # sequence, so a row over max_length would lose it to right-truncation and
  # train unlabeled; ~8% of the real shards exceed 8192.
  python "$REPO_ROOT/gb/prepare_answerability_jsonl.py" \
    --intrinsics-repo "$INTRINSICS_DIR" \
    --out-dir "$DATA_DIR" \
    --sr-config "$CFG_FOR_PREP" 2>&1 | cr_filter
  PREP_RC=${PIPESTATUS[0]}
  [ "$PREP_RC" -eq 0 ] || { echo "ERROR: data prep failed rc=$PREP_RC" >&2; exit "$PREP_RC"; }

  echo "=== data summary ==="
  wc -l "$DATA_DIR"/*.jsonl
  head -c 600 "$DATA_DIR/train.jsonl" && echo

  # --- 3. Train ---
  CFG="$CFG_FOR_PREP"
  echo "SR config: $CFG"
  echo "=== SR config contents ==="
  cat "$CFG"

  CKPT="$OUTPUT_PATH/checkpoints"
  mkdir -p "$CKPT"

  # Single line ON PURPOSE: a multi-line `python -c "` body must start at column
  # 0, so re-indenting this block silently breaks it with
  # "IndentationError: unexpected indent" -- and `bash -n` cannot catch it.
  PROBE_MODEL_RESOLVED="${PROBE_MODEL:-$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['model']['base'])")}"
  echo "probe/train model: $PROBE_MODEL_RESOLVED"

  # Diagnostic only: dumps special tokens + chat template so the boundary-marker
  # choices in the SR config can be checked against the real tokenizer. Never fatal.
  echo "=== tokenizer probe ==="
  python "$REPO_ROOT/gb/probe_tokenizer.py" --model "$PROBE_MODEL_RESOLVED" 2>&1 | cr_filter || \
    echo "(probe failed; continuing so any real error surfaces in training)"

  # `python -m`, never the sr-train console script: the step builds the env at
  # <prefix>.tmp.<pid> and renames it, leaving every pip console script with a
  # dangling shebang that dies "bad interpreter" / exit 126. `python` is a real
  # binary and is unaffected.
  NUM_GPUS="${LLMB_LSF_NUM_GPUS:-1}"
  if [ "$NUM_GPUS" -gt 1 ]; then
    # accelerate launch is itself a console script -- same shebang trap -- so go
    # through its module entry point.
    ACCEL_CFG="${ACCELERATE_CONFIG:-src/shadow_residual/config/accelerate/fsdp_4gpu.yaml}"
    case "$ACCEL_CFG" in
      /*) ACCEL_CFG_ABS="$ACCEL_CFG" ;;
      *)  ACCEL_CFG_ABS="$REPO_ROOT/$ACCEL_CFG" ;;
    esac
    [ -f "$ACCEL_CFG_ABS" ] || { echo "ERROR: accelerate config not found: $ACCEL_CFG_ABS" >&2; exit 1; }
    echo "=== train (FSDP, $NUM_GPUS GPUs) ==="
    python -m accelerate.commands.launch \
      --config_file "$ACCEL_CFG_ABS" \
      --num_processes "$NUM_GPUS" \
      -m shadow_residual.training.train \
      --config "$CFG" \
      --train-data "$DATA_DIR/train.jsonl" \
      --val-data "$DATA_DIR/val.jsonl" \
      --output-dir "$CKPT" \
      --skip-post-train-generate 2>&1 | cr_filter
  else
    echo "=== train (single GPU) ==="
    python -m shadow_residual.training.train \
      --config "$CFG" \
      --train-data "$DATA_DIR/train.jsonl" \
      --val-data "$DATA_DIR/val.jsonl" \
      --output-dir "$CKPT" \
      --skip-post-train-generate 2>&1 | cr_filter
  fi
  # PIPESTATUS[0], not $?: cr_filter always exits 0 and would mask a real failure.
  TRAIN_RC=${PIPESTATUS[0]}
  echo "=== train exit code: $TRAIN_RC ==="
  [ "$TRAIN_RC" -eq 0 ] || exit "$TRAIN_RC"

  echo "=== output ==="
  ls -la "$CKPT"
  # '|| true' is load-bearing: head exits as soon as it has its lines, find takes
  # SIGPIPE, and under `set -eo pipefail` the resulting 141 would kill this script
  # AFTER training has already finished and saved -- taking the eval with it.
  find "$CKPT" -maxdepth 3 -type f | head -80 || true
  echo "=== emitted adapter_config.json ==="
  cat "$CKPT/adapter_config.json" 2>/dev/null || echo "(missing)"
fi

# --- 4. Answerability eval. Never gates the artifact: a bad eval must not
#        invalidate a good adapter, so this always exits 0. ---
if [ "${SKIP_EVAL:-0}" = "1" ]; then
  echo "=== eval skipped (SKIP_EVAL=1) ==="
  exit 0
fi

EVAL_DATASET_DIR="$INTRINSICS_DIR/data/eval/answerability"
EVAL_OUT="$OUTPUT_PATH/eval"
mkdir -p "$EVAL_OUT"
if [ "$EVAL_ONLY" = "1" ]; then ADAPTER_DIR="$ADAPTER_PATH"; else ADAPTER_DIR="$CKPT"; fi

# Read the base model back out of the adapter so eval cannot drift from training.
# Single line for the same reason as PROBE_MODEL_RESOLVED above.
BASE_MODEL_RESOLVED="$(python -c "import json;print(json.load(open('$ADAPTER_DIR/adapter_config.json'))['base_model_name_or_path'])")"
echo "=== answerability eval ==="
echo "  base:    $BASE_MODEL_RESOLVED"
echo "  adapter: $ADAPTER_DIR"
echo "  data:    $EVAL_DATASET_DIR"
set +e
# Single GPU regardless of training's world size: a fresh model is loaded, there
# is no FSDP shard to reconstruct, and autoregressive decoding gains nothing from
# sharding.
CUDA_VISIBLE_DEVICES=0 python -m shadow_residual.eval.answerability_eval \
  --mode peft \
  --base-model "$BASE_MODEL_RESOLVED" \
  --adapter-path "$ADAPTER_DIR" \
  --adapter-sub-path "" \
  --dataset-path "$EVAL_DATASET_DIR" \
  --output-path "$EVAL_OUT" \
  --batch-size "${EVAL_BATCH_SIZE:-4}" 2>&1 | cr_filter
EVAL_RC=${PIPESTATUS[0]}
set -e
echo "=== eval exit code: $EVAL_RC ==="
ls -la "$EVAL_OUT" 2>/dev/null || true
echo "=== eval summary ==="
cat "$EVAL_OUT/eval_summary.md" 2>/dev/null || echo "(no eval_summary.md)"
exit 0
