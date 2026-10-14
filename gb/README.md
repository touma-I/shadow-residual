# Answerability Shadow Residual adapter on granite-4.2-3b — granite.build recipe

Everything needed to train a **Shadow Residual (SR) adapter for the answerability
intrinsic** on BlueVela GPUs lives in this directory. Nothing outside `gb/` was
added or modified: the recipe is purely additive to the repo.

- **Base model:** `ibm-granite/granite-4.2-3b`
- **Adapter:** SR — `cross_stream` + shared base-only K/V, always-active plain
  LoRA (stock `peft`, no gating), r16 / alpha 32
- **Result:** accuracy **0.9152**, answerable F1 0.9119, unanswerable F1 0.9186
  on the 3,565-example eval set — see [Results](#results)

## Why this exists

The answerability *recipe* (raw data, prompt shape, label strings,
hyperparameters) lives in `github.ibm.com/touma/intrinsics_training`. The SR
*model code* lives in this repo. This directory is the glue that runs one against
the other on BlueVela, with two constraints:

1. **Self-contained.** All code and config is here, in a repo on public
   github.com. The intrinsics repo is used **read-only** at an existing tag.
2. **No training data in this repo.** The data is internal and this repo is
   public, so the ~2 MB of SDG shards are fetched at job runtime over Git LFS and
   never committed.

A previous pipeline (build `81bda516`) did the same job but installed SR from a
GHE commit dated 5 Aug 2026, which predates this repo's stock-`peft` rewrite. Its
emitted adapter still carried `alora_invocation_tokens`, i.e. the old gated
design. Porting the recipe forward to the current schema is what this directory
does.

## Files

| File | Role |
|---|---|
| `sr-answerability-4.2-3b.yaml` | **granite.build build YAML.** The thing you submit. Compute sizing, secrets, the BYOC config. Never read by the running job — `build_start` takes it from local disk. |
| `sr-train-config-4.2-3b.yaml` | **SR training config.** Hyperparameters + data contract, consumed by `shadow_residual.training.train` via `--config`. Selected by the `SR_CONFIG` env var. |
| `byoc_run.sh` | **Job entrypoint** (`start_command`). Four phases: fetch data → convert → train → eval. Carries the platform workarounds (see [Traps](#traps-encoded-in-byoc_runsh)). |
| `prepare_answerability_jsonl.py` | Converts the intrinsics raw shards to SR messages-JSONL. Imports *their* transform module rather than reimplementing it. |
| `probe_tokenizer.py` | Diagnostic: dumps special tokens + chat template so marker choices can be checked against the real tokenizer. Non-fatal. |
| `environment.yml` | Conda env (`conda_lockfile_path`). Minimal: python, pip, git, **git-lfs**. |
| `requirements-byoc.txt` | pip pins (`setup_command`). Mirrors `pyproject.toml` but pins torch to an explicit CUDA build. |

Two files share a near-identical name on purpose, so read carefully:
`sr-answerability-4.2-3b.yaml` is the **build** YAML (submitted to granite.build);
`sr-train-config-4.2-3b.yaml` is the **training** config (consumed by the
trainer). They were in different directories before this layout.

## How to run it

```bash
# 1. Pin github_ref to an immutable SHA on this branch (it resolves at job START,
#    so a branch name lets a push mid-PEND change what gets cloned).
git rev-parse HEAD

# 2. Validate (free, local). The flag is --filename, NOT --file.
gb build validate --filename gb/sr-answerability-4.2-3b.yaml --space public --skip-version-check

# 3. Submit.
gb build start --filename gb/sr-answerability-4.2-3b.yaml --space public --skip-version-check
```

Dashboard: `https://api.llm-build-dev.vpc-int.res.ibm.com/dashboard/build/<BUILD_ID>`

Useful env knobs, all set in the build YAML's `env:` block:

| Var | Effect |
|---|---|
| `SR_CONFIG` | which training config to use |
| `INTRINSICS_REF` | intrinsics_training ref. Must be a ref **name** — `git clone --branch` cannot take a raw SHA |
| `SMOKE_MAX_ROWS` / `SMOKE_SUBSAMPLE` / `SMOKE_MAX_STEPS` | shrink the run for a pipeline check |
| `EVAL_ONLY=1` + `ADAPTER_PATH=<dir>` | score an existing adapter, no retraining |
| `SKIP_EVAL=1` | train only |

**Cheap pipeline check first.** A ~20-step smoke run costs ~25 min against ~6 h
for the real thing. Add `SMOKE_MAX_ROWS=5000`, `SMOKE_SUBSAMPLE=2000`,
`SMOKE_MAX_STEPS=20`. Check the step arithmetic when you shrink the input: 1800
rows / effective batch 8 = 225 steps per epoch, so a 20-step cap is safe, but a
cap below the batch size yields **zero** steps and a job that exits 0 having done
nothing.

**Expect a one-time ~50-minute env rebuild** on the first run of this layout.
Moving these files under `gb/` changed `setup_command`, which is part of the conda
env cache key. File contents are unchanged, so nothing else about the env differs.

## What this recipe does differently from upstream

Hyperparameters are ported verbatim from intrinsics'
`configs/answerability/shadow_residual/sr_sharedkv_4.2_3b.yaml`: r16 / alpha 32 /
dropout 0.05 / lr 5e-5 / 3 epochs / batch 1 × grad-accum 8 / max_length 8192 /
seed 0. Four things had to change.

1. **`adapter.invocation_tokens` removed.** SR is always-active plain LoRA now;
   the config schema is `extra="forbid"` and rejects the key outright.
2. **`model.shared_base_kv` removed.** Shared base-only K/V is the only supported
   topology, so the key is no longer accepted.
3. **`gradient_checkpointing` false → true.** Upstream set it false only because
   SR forced it off whenever `invocation_tokens` was set (PEFT #2826). The
   dual-stream decoder forward is pure, so HF's non-reentrant recompute is exact.
4. **The supervised-region boundary moved by three tokens.** This is the one
   change that alters what the model is trained on, so it is spelled out below.

### The boundary change — read this before comparing F1 to older runs

granite-4.2's published `chat_template.jinja` rewrites assistant content to
`"<think></think>" ~ content` **unconditionally** when the content contains no
think tags. `enable_thinking` is read only inside the `add_generation_prompt`
branch, so it does **not** suppress this. A training row therefore renders as:

```
<|im_start|>assistant\n<think></think>"answerable"<|im_end|>\n
```

and inference with `add_generation_prompt=True, enable_thinking=False` ends the
prompt at exactly `...<|im_start|>assistant\n<think></think>`.

This config anchors at `assistant_marker:
"<|im_start|>assistant\n<think></think>"` with `last_context_token: "</think>"`,
so supervision starts on the first token the model actually generates at serving
time. **Both prior pipelines anchored at `"<|im_start|>assistant\n"`** — the
published intrinsics adapter records `alora_invocation_tokens [100256, 78191,
198]`, i.e. `<|im_start|>` / `assistant` / `\n` — which leaves `<think></think>`
inside the supervised region, training the model to emit three tokens the prompt
always supplies.

Consequence: the historical 3b figure of **0.914 accuracy was measured under the
old anchoring**, so 0.9152 here is **not** a controlled comparison. Treat it as
"reproduces the baseline", not "beats it" — the gap is far inside the 0.0854
sampling std.

Verified against the real tokenizer rather than assumed: `</think>` is a single
token (100275), `<|im_end|>` is 100257, and the in-job check logged
`last_context_token boundary validated on 8/8 sampled rows`.

### Document placement

`prepare_answerability_jsonl.py` emits a top-level `documents` field and lets
SR's own `chat_render` place it, which for a ChatML tokenizer means a `tool`
(search-response) message. That is deliberate: `answerability_eval` renders its
documents through the *same* function, so training and eval cannot disagree.

Intrinsics' own export uses `prompt_version: user_doc_message`, which bakes the
documents into a leading user turn (`"You have access to the following
documents:..."`) and matches the published serving contract
(`docs_as_message: string`). Adopting that shape is a reasonable future variant,
but it would require patching `shadow_residual/eval/answerability_eval.py` in
lockstep — otherwise the eval silently disagrees with training. That is the one
change that would stop this recipe being purely additive.

## Traps encoded in `byoc_run.sh`

Each of these cost a real build somewhere. Do not "clean them up".

| Guard | Why |
|---|---|
| `TQDM_DISABLE=1` + `cr_filter() { stdbuf -o0 tr '\r' '\n'; }` on every subprocess | gbserver tails `job.log` with a 64 KiB `readline()` limit. A tqdm bar is one giant CR-delimited "line"; exceeding the limit makes gbserver **bkill the job**. |
| `python -m shadow_residual.training.train`, never `sr-train` | The step builds the env at `<prefix>.tmp.<pid>` and renames it, leaving every pip console script with a dangling shebang → `bad interpreter`, exit 126. |
| `unset LSB_HOSTS` | LSF sets it even for a single-node job; launchers read it as multinode and try a rendezvous with no second host. |
| `GIT_EXEC_PATH` / `GIT_TEMPLATE_DIR` and the CA-bundle vars repointed at system paths | Same env rename staled the conda git's baked-in paths. |
| `find ... \| head -80 \|\| true` | `head` exits early, `find` takes SIGPIPE, and under `set -eo pipefail` the resulting 141 kills the script *after* training finished. This once destroyed a completed 35 h run. |
| LFS sanity gate (smallest file < 10000 bytes → abort) | Without git-lfs the data arrives as ~131-byte pointer files and the job would train on nothing while exiting 0. |
| Inline `python -c` kept on ONE line | A multi-line `python -c "` body must start at column 0; re-indenting silently breaks it with `IndentationError`, and `bash -n` cannot catch it. |
| `EVAL_RC=${PIPESTATUS[0]}`, not `$?` | `cr_filter` always exits 0 and would mask a real eval failure. |
| Eval phase always `exit 0` | A bad eval must never invalidate a good adapter. |
| intrinsics clone under `$REPO_ROOT`, not `$OUTPUT_PATH` | `dir_to_save: "."` uploads everything in `$OUTPUT_PATH`; cloning there would ship the whole training corpus with the artifact. |

## Results

Full run — build **`0f7a356d-db43-4e86-b7eb-0ccb90d2be68`**, code at `07ed601`
(pre-restructure layout), 1 GPU, 2026-10-14.

| | |
|---|---|
| Rows | 14,277 transformed → 12,849 / 1,428 split → **11,812 train / 1,335 val** after dropping over-length |
| Over-length dropped | 1,037 / 12,849 train (8.1%), 93 / 1,428 val (6.5%) |
| Steps | 1,477 per epoch × 3 = **4,431** |
| Trainable | `30,474,240 || all 3,952,355,840 || 0.7710%` |
| Wall clock | ~5h43m train + eval |

| Metric | Value |
|---|---|
| answerability_f1 | 0.9119 |
| unanswerability_f1 | 0.9186 |
| **accuracy** | **0.9152** |
| weighted_f1 | 0.9155 |
| mean_f1 | 0.9152 |
| accuracy_std | 0.0854 |
| accuracy_ci_95 | 0.7478 – 1.0825 |

Historical 3b reference: accuracy 0.914 (answerable F1 0.910, unanswerable F1
0.918) — but measured under the old boundary anchoring, so see the caveat above.

The eval report header says **"granite-4.2-3b + aLoRA"**. That is a cosmetic bug
in `answerability_eval.py` (line ~723 hardcodes `+ " + aLoRA"` in `--mode peft`).
This adapter is **not** aLoRA; its `adapter_config.json` has
`alora_invocation_tokens: null`.

### Where the adapter is

The lakehouse push failed (below), so the artifact lives on `/proj` only:

```bash
RUN=/proj/data-eng/llmb-read-write/builds/llm-build-0f7a356d-db43-4e86-b7eb-0ccb90d2be68/target-train/target-run-aa38e155-da47-433c-9fde-7a7892ad0cdf/step-custom_code_lsf/step-run-fc18b438-64c4-4980-afcc-ff53e5fa16af/launch-aa8ace64-48b3-48f5-bb11-dd6aab79f871/outputs

ls $RUN/checkpoints   # adapter_config.json, adapter_model.safetensors (116 MB),
                      # tokenizer files, checkpoint-{1477,2954,4431}, runs/
ls $RUN/eval          # eval_summary.{md,jsonl,csv}, result_df.csv (all 3,565 predictions)
ls $RUN/data          # train.jsonl / val.jsonl exactly as trained on
```

The root `checkpoints/` adapter is the `load_best_model_at_end` pick by
`eval_loss` — not necessarily epoch 3. All three epoch checkpoints survive, so
any can be re-scored without retraining via `EVAL_ONLY=1
ADAPTER_PATH=$RUN/checkpoints/checkpoint-2954`.

To load it:

```python
from shadow_residual.shadow_residual.build import build_sr_base
from peft import PeftModel
base = build_sr_base("ibm-granite/granite-4.2-3b", torch_dtype=...)
model = PeftModel.from_pretrained(base, f"{RUN}/checkpoints")
```

Note this path is under the build workspace. If the adapter matters long-term,
re-run the lakehouse push rather than relying on that location.

## What worked, what failed

**Worked first time.** Cloning this repo from **public github.com** on a BlueVela
compute node. This was the one genuinely unverified assumption — the step's
documented egress covers conda, PyPI, the PyTorch index and huggingface.co, not
github.com. It works, and the `github_url` schema accepts any `host.tld/path`
(pattern `^[^:/]+\.[^:/]+/.+`), so **no GHE mirror is needed**. The GHE PAT is
interpolated as the URL username; github.com ignores an invalid credential for
public-repo reads, so that is harmless.

Also worked: the narrow LFS pull, the `outputs:` lakehouse registration (the
`lh://` templating resolved), and the warm env cache — the full run reused the
smoke run's env hash `byoc-7e6d7089b9225fcd` and started in seconds.

**Failed, and why:**

| Build | Outcome | Cause |
|---|---|---|
| `f8bfeb40` | **SUCCESS** | 20-step smoke. Proved clone, LFS, config load, training, eval. |
| `5cc97f9c` | FAILED in data prep | `KeyError: 'Environment variable ${DATA_ROOT} is not set'`. The over-length filter called `load_training_config()`, which eagerly resolves every `${VAR}` — including output paths the prep step neither has nor needs. Fixed by reading the YAML with `yaml.safe_load` for the four scalar fields it actually wants. |
| `0f7a356d` | train+eval **success**, build **failed** | The `lhpush` step died on `401 Token is expired!` from `dmf fileset push`. A platform credential issue, downstream of all the real work — the adapter and metrics are intact. A FAILED build can contain a fully successful run. |

Two defects were found by verification rather than by a run blowing up, and both
would have silently degraded results:

1. **`drop_overlength` was missing.** Answerability's label is last in the
   sequence, so a row over `max_length` loses its completion to right-truncation
   and becomes an unlabeled example contributing no gradient. 8.1% of train rows
   exceed 8192. Intrinsics guards this in its export config; the script this was
   ported from had no equivalent, so the behaviour was lost in the port. The
   filter now measures length through `load_jsonl_dataset` — the same call the
   trainer makes — so it cannot drift from what training sees.
2. **The boundary anchoring** described above.

**Process lesson:** `5cc97f9c` failed because the over-length filter was added
*after* the smoke run validated the pipeline, and the full run was submitted
without re-testing the changed path. The smoke build could not have caught it —
it ran the pre-fix commit. Re-validate the path you just changed, even when the
change looks small. The failure was cheap only because data prep runs in the
first minutes of the job.

### Debugging a failure

Read `outputs/job.log` in the build workspace **first**. `build_investigate` and
`build_log` return orchestration noise; on a real failure here they contained no
error at all.

```
build_files_list(build_id, recursive=True, pattern="outputs/job.log$", regex=True)
build_files_peek(build_id, path="<...>/outputs/job.log", tail=40)
build_files_search(build_id, path="<...>/outputs", regex=True,
  pattern="Traceback|CondaError|workload script failed|KeyError")
```

Scope `path` to the step's `outputs` dir — searching the whole tree matches the
cloned source and buries the error. Build status `running` only means the
orchestrator is alive; the LSF job can still be `PEND`. Confirm with
`build_log(runner=True, grep="-o -m3 STAT='[A-Z]*'")`.

## Related branches

- **`gb/sr-4.2-smoke-test`** — the development history, including
  `gb/sr-answerability-4.2-3b-smoke.yaml` (the 20-step smoke build YAML, which is
  deliberately **not** carried over here) and the pre-restructure layout where
  these files sat in `scripts/` and `src/shadow_residual/config/`. Build
  `0f7a356d`'s code (`07ed601`) is on that branch.

## Known rough edges

- `sr-train-config-4.2-3b.yaml` lives here rather than in
  `src/shadow_residual/config/` with the six other training configs. It is
  therefore **not** installed as package data (`pyproject.toml` ships only
  `config/*.yaml`), so a `pip install` of this repo will not carry it. Its
  `$schema` comment points back into the package by relative path.
- The eval report mislabels every `--mode peft` adapter as "+ aLoRA".
- The `outputs:` lakehouse block is what introduces the `lhpush` step, and hence
  the possibility of a red build from an otherwise perfect run. Removing it keeps
  checkpoints on `/proj` and builds green, at the cost of `build_lineage`
  resolving to `[]`.
