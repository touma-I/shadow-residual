# Answerability Shadow Residual adapter on granite-4.2-8b — granite.build recipe

Everything needed to train a **Shadow Residual (SR) adapter for the answerability
intrinsic** on BlueVela GPUs lives in this directory. Nothing outside `gb/` was
added or modified: the recipe is purely additive to the repo.

- **Base model:** `ibm-granite/granite-4.2-8b`
- **Adapter:** SR — `cross_stream` + shared base-only K/V, always-active plain
  LoRA (stock `peft`, no gating), r16 / alpha 32
- **Result:** not yet run at 8b on this pipeline — see [Results](#results) for
  the prior 8b reference and the 3b figure from the sibling branch
- **Sibling branch:** [`gb/sr-ans-4.2-3b`](#related-branches) is the same recipe
  at 3b, where it scored accuracy 0.9152

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
| `sr-answerability-4.2-8b.yaml` | **granite.build build YAML.** The thing you submit. Compute sizing, secrets, the BYOC config. Never read by the running job — `build_start` takes it from local disk. |
| `sr-train-config-4.2-8b.yaml` | **SR training config.** Hyperparameters + data contract, consumed by `shadow_residual.training.train` via `--config`. Selected by the `SR_CONFIG` env var. |
| `byoc_run.sh` | **Job entrypoint** (`start_command`). Four phases: fetch data → convert → train → eval. Carries the platform workarounds (see [Traps](#traps-encoded-in-byoc_runsh)). |
| `prepare_answerability_jsonl.py` | Converts the intrinsics raw shards to SR messages-JSONL. Imports *their* transform module rather than reimplementing it. |
| `probe_tokenizer.py` | Diagnostic: dumps special tokens + chat template so marker choices can be checked against the real tokenizer. Non-fatal. |
| `environment.yml` | Conda env (`conda_lockfile_path`). Minimal: python, pip, git, **git-lfs**. |
| `requirements-byoc.txt` | pip pins (`setup_command`). Mirrors `pyproject.toml` but pins torch to an explicit CUDA build. |

Two files share a near-identical name on purpose, so read carefully:
`sr-answerability-4.2-8b.yaml` is the **build** YAML (submitted to granite.build);
`sr-train-config-4.2-8b.yaml` is the **training** config (consumed by the
trainer). They were in different directories before this layout.

**Only those two files are 8b-specific.** The other six are byte-identical to the
3b branch — the model id is read out of the training config at runtime, so
`prepare_answerability_jsonl.py`, `probe_tokenizer.py`, `environment.yml` and
`requirements-byoc.txt` never mention a model, and `byoc_run.sh` differs only in
which config it defaults to (the build YAML overrides that anyway). See
[Keeping the branches in step](#keeping-the-branches-in-step).

## How to run it

```bash
# 1. Pin github_ref to an immutable SHA on this branch (it resolves at job START,
#    so a branch name lets a push mid-PEND change what gets cloned).
git rev-parse HEAD

# 2. Validate (free, local). The flag is --filename, NOT --file.
gb build validate --filename gb/sr-answerability-4.2-8b.yaml --space public --skip-version-check

# 3. Submit.
gb build start --filename gb/sr-answerability-4.2-8b.yaml --space public --skip-version-check
```

Dashboard: `https://api.llm-build-dev.vpc-int.res.ibm.com/dashboard/build/<BUILD_ID>`

Useful env knobs, all set in the build YAML's `env:` block:

| Var | Effect |
|---|---|
| `SR_CONFIG` | which training config to use |
| `INTRINSICS_REF` | intrinsics_training ref. Must be a ref **name** — `git clone --branch` cannot take a raw SHA |
| `EVAL_ONLY=1` + `ADAPTER_PATH=<dir>` | score an existing adapter, no retraining |
| `SKIP_EVAL=1` | train only |

There is deliberately **no smoke/shrink path here**: this recipe always runs the
full dataset for the full 3 epochs. The row caps and step cap that existed during
development were removed from `byoc_run.sh` on this branch, so there is no knob
that can silently train on a subset. The transform's own defaults (`max_rows`
400000, `subsample` 40000) both sit above the ~14.3k rows the shards yield, so
neither bites.

### What it costs

Measured where measured, and labelled where not:

| Phase | Time | Source |
|---|---|---|
| Everything before the first training step, **with a cold conda env**: repo clone + `conda env create` + full pip install + intrinsics clone + LFS pull + full data prep + model load | **23 min** | measured, 3b build `fb991c9d` on this exact layout — step start 13:48:45, `cache: MISS`, first training step 14:11:46. Dominated by the over-length filter tokenizing all 14.3k rows, not by the env (a cold env alone measured ~5 min on `f8bfeb40`) |
| Same, warm env | **31 min** | measured, 3b build `0f7a356d`. Yes — *longer* than the cold-env run above, which is the point: the env is noise next to the tokenizing pass |
| Train + eval, 3 epochs, 1 GPU | **~11-14 h** | *estimate.* The prior 8b run `88e44a92` did 11h02m end to end on 1 GPU — but at `max_length` 4096, so 8192 should cost more. The same recipe at 3b took 5h12m |
| End to end | **~11-14 h** | as above; 3b was 5h47m |

The **setup cost is the tokenizing pass, not the conda env**. A cold env is
single-digit minutes here — if you see "~50 minutes" quoted for a BYOC env build,
it does not match this workload. Host RAM is not the constraint: the 3b run peaked
at 8.6 GB, and 200Gi is requested here purely as headroom.

Fill the estimates in with real numbers once this branch has a completed run.

## What this recipe does differently from upstream

Hyperparameters are ported verbatim from intrinsics'
`configs/answerability/shadow_residual/sr_sharedkv_4.2_8b.yaml` — which, for
answerability, is the **only** SR config that repo ships; there is no 3b sibling,
unlike citations, hallucination detection, query clarification and query rewrite.
So this branch runs the upstream recipe at its native size, and the 3b branch is
that same recipe with the base model swapped down. The values: r16 / alpha 32 /
dropout 0.05 / lr 5e-5 / 3 epochs / batch 1 × grad-accum 8 / max_length 8192 /
seed 0. Five things had to change.

1. **`adapter.invocation_tokens` removed.** SR is always-active plain LoRA now;
   the config schema is `extra="forbid"` and rejects the key outright.
2. **`model.shared_base_kv` removed.** Shared base-only K/V is the only supported
   topology, so the key is no longer accepted.
3. **`gradient_checkpointing` false → true.** Upstream set it false only because
   SR forced it off whenever `invocation_tokens` was set (PEFT #2826). The
   dual-stream decoder forward is pure, so HF's non-reentrant recompute is exact.
4. **The supervised-region boundary moved by three tokens.** This is the one
   change that alters what the model is trained on, so it is spelled out below.
5. **`max_length` kept at 8192** — see below. Upstream says 8192; the earlier GHE
   fork's 8b config cut it to 4096, and that is the one place this branch
   deliberately disagrees with the thing that last ran successfully at 8b.

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

Consequence: any published 8b answerability figure was measured under the **old**
anchoring, so whatever this run scores is **not** a controlled comparison against
it. The same caveat applied at 3b, where this pipeline scored 0.9152 against a
0.914 historical reference — a gap far inside the 0.0854 sampling std.

Verified against the real tokenizer rather than assumed: for
`ibm-granite/granite-4.2-8b`, `</think>` is a single token (100275) and
`<|im_end|>` is 100257 — **identical ids to 4.2-3b**, so the markers carry over
unchanged. At 3b the in-job check logged `last_context_token boundary validated on
8/8 sampled rows`; expect the same line here.

### Why `max_length` stays 8192

This is the only substantive disagreement with the prior successful 8b run, so
here is the evidence. Measured on 3,000 real transformed rows with the
granite-4.2-8b tokenizer, through `load_jsonl_dataset` — the same call the trainer
makes:

| | train | val |
|---|---|---|
| p50 / p90 / p99 | 779 / 7,332 / 14,926 | 776 / 6,706 / 13,818 |
| max | 19,568 | 17,674 |
| dropped at 8192 | 8.2% | 6.0% |
| dropped at 6144 | 14.4% | 14.3% |
| dropped at 4096 | **18.9%** | **18.3%** |

Because answerability's label is the **last** thing in the sequence, a row over
`max_length` is not merely truncated — it loses its completion, so
`prepare_answerability_jsonl.py` drops it. `max_length` is therefore a direct
data-retention knob, and 4096 costs more than twice the data 8192 does.

The GHE fork's 8b config justified 4096 with "the answerability rows are well
under this length". They are not: p90 is 7,332 tokens. 8192 also keeps the
training set comparable to the 3b run (8.1% dropped there), which is what makes
the two branches' numbers worth putting side by side.

The risk this takes on is GPU memory, since activation cost grows with sequence
length and the model is 2.6× larger. Gradient checkpointing is on, batch is 1, and
`expandable_segments:True` is set. If it still OOMs it will do so within the first
minutes of training, not hours in — drop to 6144 and record the new drop rate.

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

## Pre-flight checks

Everything here is free and local, and all of it runs in under a minute. Run it
before every submit — the one build that failed outright (`5cc97f9c`) failed
because code changed after the last check and the check was not repeated.

```bash
# 1. The entrypoint parses. Catches quoting/continuation damage, though NOT a
#    broken inline `python -c` body (see Traps).
bash -n gb/byoc_run.sh

# 2. The training config loads under the real extra="forbid" schema, with the
#    ${VAR}s the wrapper exports. A schema rejection here costs a queue wait.
PYTHONPATH=src DATA_ROOT=/tmp/d RUN_DIR=/tmp/r EVAL_DATA=/tmp/d/val.jsonl \
  python -c "from shadow_residual.config.training_config import load_training_config as L; \
             c=L('gb/sr-train-config-4.2-8b.yaml'); \
             assert 'cross_stream' in c.adapter.target_modules; \
             assert not {'k_proj','v_proj'} & set(c.adapter.target_modules); print('ok')"

# 3. Data prep runs end to end against a local intrinsics clone. Use a small
#    --subsample; the transform still reads all 7 shards, so the row count it
#    reports ("transform produced 14277 rows") is the real one.
PYTHONPATH=src python gb/prepare_answerability_jsonl.py \
  --intrinsics-repo ~/intrinsics_training --out-dir /tmp/preptest \
  --sr-config gb/sr-train-config-4.2-8b.yaml --subsample 300

# 4. The build YAML validates. The flag is --filename, NOT --file.
gb build validate --filename gb/sr-answerability-4.2-8b.yaml --space public --skip-version-check
```

Step 3 is the one worth insisting on: it exercises the import-by-path into
intrinsics' `transforms` module, the positional record contract, the label
strings and the over-length filter — the parts most likely to break silently
rather than loudly.

The SR model code itself is untouched by this directory, so `pytest` only needs
running if you change something under `src/`.

## Results

**Not yet run at 8b on this pipeline.** This section is a placeholder with the
numbers to check against; fill it in from the first completed build.

What to expect, and why each number is a check rather than a formality:

| Signal | Expected | What a mismatch means |
|---|---|---|
| Rows | `transform produced 14277 rows` → 12,849 / 1,428 split → ~11.8k train / ~1.3k val after the over-length drop | A different transform count means the data or the intrinsics ref moved |
| Over-length dropped | ~8.2% train, ~6.0% val at `max_length` 8192 | A much larger figure means `max_length` is not what this config says |
| Steps | ~1,480 per epoch × 3 = ~4,430 | Far fewer means the batch arithmetic or a row cap is wrong |
| Trainable params | the log prints `trainable params: … || all … || trainable%:`. At 3b it was `30,474,240 \|\| 3,952,355,840 \|\| 0.7710%`; expect a larger absolute count here at a similar order of percentage | A tiny count means `cross_stream`/MLP targets did not attach |
| Loss | starts ~6.4 with token-acc ~0.45, falls below 0.05 with token-acc >0.99 | A flat loss means the supervised region is masked wrong |
| Boundary check | `last_context_token boundary validated on N/N sampled rows` | Anything else means the marker and the template disagree |
| `adapter_config.json` | `r: 16`, `lora_alpha: 32`, `target_modules` includes `cross_stream`, **`alora_invocation_tokens: null`** | A non-null value means an old gated SR got installed |
| Eval | `eval_summary.jsonl` with answerable/unanswerable F1 | — |

Reference points to compare against, both measured under the **old** boundary
anchoring (see the caveat above), so neither is a controlled comparison:

| Run | Model | Result |
|---|---|---|
| This pipeline, 3b (`0f7a356d`, sibling branch) | granite-4.2-3b | accuracy **0.9152**, answerable F1 0.9119, unanswerable F1 0.9186, accuracy_std 0.0854 — the only figure produced by *this* boundary anchoring |
| `88e44a92`, 2026-08-20 | granite-4.2-8b | SUCCESS at `max_length` 4096, 11h02m, 1 GPU / 200Gi, old SR code and old anchoring |
| Historical intrinsics 3b | granite-4.2-3b | accuracy 0.914 (answerable F1 0.910, unanswerable F1 0.918) |

Note the eval report header will say **"granite-4.2-8b + aLoRA"**. That is a
cosmetic bug in `answerability_eval.py` (line ~723 hardcodes `+ " + aLoRA"` in
`--mode peft`). This adapter is **not** aLoRA; its `adapter_config.json` has
`alora_invocation_tokens: null`.

### Where the adapter will be

There is no `outputs:` block, so the artifact stays in the build workspace on
`/proj`. After the run, find it with:

```bash
# the step's outputs dir, under:
#   /proj/data-eng/llmb-read-write/builds/llm-build-<BUILD_ID>/target-train/
#     target-run-<...>/step-custom_code_lsf/step-run-<...>/launch-<...>/outputs
build_files_list(build_id, recursive=True, pattern="outputs/checkpoints$", regex=True)

ls $RUN/checkpoints   # adapter_config.json, adapter_model.safetensors,
                      # tokenizer files, checkpoint-{N,2N,3N}, runs/
ls $RUN/eval          # eval_summary.{md,jsonl,csv}, result_df.csv (all 3,565 predictions)
ls $RUN/data          # train.jsonl / val.jsonl exactly as trained on
```

For scale, the 3b adapter's `adapter_model.safetensors` was 116 MB; expect larger
here.

The root `checkpoints/` adapter is the `load_best_model_at_end` pick by
`eval_loss` — not necessarily epoch 3. All three epoch checkpoints survive, so
any can be re-scored without retraining via `EVAL_ONLY=1
ADAPTER_PATH=$RUN/checkpoints/checkpoint-<step>`.

To load it:

```python
from shadow_residual.shadow_residual.build import build_sr_base
from peft import PeftModel
base = build_sr_base("ibm-granite/granite-4.2-8b", torch_dtype=...)
model = PeftModel.from_pretrained(base, f"{RUN}/checkpoints")
```

Note this path is under the build workspace. If the adapter matters long-term,
publish it (restore the `outputs:` block in the build YAML) rather than relying on
that location.

## What worked, what failed

Everything below was established on the **3b** branch unless it says otherwise.
It is carried over here because every one of these facts is model-independent —
the clone, the data fetch, the env, the platform traps. The 8b-specific unknowns
are listed at the end of this section.

**Worked first time.** Cloning this repo from **public github.com** on a BlueVela
compute node. This was the one genuinely unverified assumption — the step's
documented egress covers conda, PyPI, the PyTorch index and huggingface.co, not
github.com. It works, and the `github_url` schema accepts any `host.tld/path`
(pattern `^[^:/]+\.[^:/]+/.+`), so **no GHE mirror is needed**. The GHE PAT is
interpolated as the URL username; github.com ignores an invalid credential for
public-repo reads, so that is harmless.

Also worked: the narrow LFS pull, and the env cache — one run reused the other's
env hash `byoc-7e6d7089b9225fcd` and started in seconds.

**Build history:**

| Build | Outcome | What it showed |
|---|---|---|
| `f8bfeb40` | **SUCCESS** | 20-step pipeline check during development. Proved github.com clone, LFS, config load, training, eval. Its YAML lives on `gb/sr-4.2-smoke-test`. |
| `5cc97f9c` | FAILED in data prep | `KeyError: 'Environment variable ${DATA_ROOT} is not set'`. The over-length filter called `load_training_config()`, which eagerly resolves every `${VAR}` — including output paths the prep step neither has nor needs. Fixed by reading the YAML with `yaml.safe_load` for the four scalar fields it actually wants. |
| `0f7a356d` | train+eval **success**, build **failed** | The `lhpush` step died on `401 Token is expired!` from `dmf fileset push`. A platform credential issue, downstream of all the real work — the adapter and metrics are intact. A FAILED build can contain a fully successful run. |
| `fb991c9d` | 3b, on the `gb/` layout | Same recipe from the consolidated `gb/` directory, no `outputs:` block. |
| `88e44a92` | **SUCCESS**, 8b | 2026-08-20, the last 8b answerability SR run: 11h02m, 1 GPU / 200Gi, `EVAL_BATCH_SIZE` 2. Old SR code (gated aLoRA), old boundary anchoring, `max_length` 4096. The compute sizing on this branch comes from it. |

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

### What is still unknown at 8b

Three things this branch has not yet demonstrated. All three fail early and
cheaply if they fail at all:

1. **Whether 8B at `max_length` 8192 fits one GPU.** The only successful 8b run
   used 4096. Gradient checkpointing is on and batch is 1; an OOM would surface in
   the first minutes of training. Fallback: 6144, then 4096, recording the new
   drop rate each time.
2. **Whether bf16 is stable at 8B.** The schema docstring recommends fp32 master
   weights at this scale, but that advice targets FSDP, and `88e44a92` trained in
   bf16 single-GPU. `validation.stop_on_nan` is on, so a divergence stops the run
   at epoch end rather than burning 11 h.
3. **Eval throughput at 8b.** `EVAL_BATCH_SIZE` is 2, copied from `88e44a92`. The
   3b eval ran 3,565 examples at 3.7/s in ~16 min; expect materially slower.

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

- **`gb/sr-ans-4.2-3b`** — the same recipe at granite-4.2-3b, and the branch this
  one was cut from. It holds the only results produced under the current boundary
  anchoring (accuracy 0.9152) and the measured timings this branch's estimates
  lean on.
- **`gb/sr-4.2-smoke-test`** — the development history: the pre-restructure
  layout where these files sat in `scripts/` and `src/shadow_residual/config/`,
  plus `sr-answerability-4.2-3b-smoke.yaml`, the 20-step pipeline-check build.
  Neither that YAML nor any smoke knob is carried over here — this branch runs the
  full recipe only. Build `0f7a356d`'s code (`07ed601`) is on that branch.

### Keeping the branches in step

Six of the eight files here are byte-identical to `gb/sr-ans-4.2-3b`, so a fix to
any of them has to be applied twice. Check with:

```bash
git diff gb/sr-ans-4.2-3b gb/sr-ans-4.2-8b -- \
  gb/prepare_answerability_jsonl.py gb/probe_tokenizer.py \
  gb/environment.yml gb/requirements-byoc.txt
# expect: no output

git diff gb/sr-ans-4.2-3b gb/sr-ans-4.2-8b -- gb/byoc_run.sh
# expect: only the default SR_CONFIG path (2 lines)
```

This duplication is the deliberate cost of one branch per model size. The
alternative is a single branch carrying both training configs and both build
YAMLs — `SR_CONFIG` already selects between them at submit time, so the
entrypoint needs no change for it.

## Known rough edges

- `sr-train-config-4.2-8b.yaml` lives here rather than in
  `src/shadow_residual/config/` with the six other training configs. It is
  therefore **not** installed as package data (`pyproject.toml` ships only
  `config/*.yaml`), so a `pip install` of this repo will not carry it. Its
  `$schema` comment points back into the package by relative path.
- The eval report mislabels every `--mode peft` adapter as "+ aLoRA".
- **1 GPU only for now.** `byoc_run.sh` switches to `accelerate launch` with the
  bundled FSDP config when `LLMB_LSF_NUM_GPUS > 1`, and the only config shipped is
  `fsdp_4gpu.yaml`, so the supported values are 1 or 4. **That path has never been
  exercised by any of these builds** — every success, 3b and 8b, was single-GPU —
  so 4 GPUs is a deliberate next iteration rather than a config tweak. When trying
  it, also set `param_dtype: fp32` in the training config: under FSDP, bf16
  force-casts adapters down to satisfy flatten.
- There is **no `outputs:` lakehouse block** in the build YAML on this branch.
  That block is what appends the `lhpush` step, and hence the possibility of a red
  build from an otherwise perfect run — which is exactly what happened to
  `0f7a356d`. Without it, checkpoints stay on `/proj` and the build goes green, at
  the cost of `build_lineage` resolving to `[]`. The commented-out block is kept in
  the YAML for whenever the adapter should be published.
