#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert intrinsics_training answerability raw data -> SR messages JSONL.

The task definition stays in intrinsics_training: rather than reimplementing its
transform, this imports that repo's standalone transform module by path and calls
it. So the raw shards, the positional record contract, the label strings and the
shuffle/subsample/split are all theirs, byte for byte. We only reshape the
resulting columns into the row format shadow_residual's loader wants.

That module is self-contained by design ("nothing imports from the parent
repository") and its subtree has no __init__.py, so it is imported by inserting
the containing directory on sys.path and importing `transforms` directly -- no
install of intrinsics_training, and none of its conflicting pins.

Defaults inherited from that module: subsample 40000, seed 42, test_size 0.1,
max_rows 400000.

Over-length rows are DROPPED rather than trained on, mirroring
intrinsics_training's own export (`drop_overlength: true`). Answerability's label
is the last thing in the sequence, so a row longer than max_length loses its
completion to right-truncation and silently becomes an unlabeled example that
contributes no gradient. Measured on the real shards, ~8% of rows exceed 8192.
The length is measured through the exact training path -- load_jsonl_dataset,
which renders the chat template and appends last_token -- so the count cannot
drift from what the trainer sees.

Output row shape, matching shadow_residual.training.data.load_jsonl_dataset:

    {"messages": [...], "documents": [...]}    # documents optional

Document placement is therefore left to SR's own renderer
(shadow_residual.training.chat_render), which for a ChatML tokenizer injects the
documents as a `tool` (search-response) message. That is deliberate and it is
what keeps training consistent with shadow_residual.eval.answerability_eval,
which feeds its documents through the same renderer. The alternative -- baking
the documents into a leading user turn, as intrinsics_training's own
`prompt_version: user_doc_message` export does -- is a different prompt shape;
adopting it would require changing the eval in lockstep or the two would
silently disagree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# LFS pointer files are ~131 bytes; a real gzipped shard is hundreds of KB.
_POINTER_MAX_BYTES = 1000


def _write(path, rows) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _fits(path, tokenizer, max_length, enable_thinking, last_token) -> set:
    """Return the indices of rows that survive right-truncation at max_length.

    Measures through load_jsonl_dataset -- the same call the trainer makes -- so
    the rendered template and the appended last_token are accounted for exactly.
    """
    from shadow_residual.training.data import load_jsonl_dataset  # noqa: E402

    ds = load_jsonl_dataset(
        path, tokenizer, enable_thinking=enable_thinking, last_token=last_token
    )
    keep = set()
    for i, text in enumerate(ds["text"]):
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) < max_length:
            keep.add(i)
    return keep


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intrinsics-repo", required=True, type=Path,
                   help="Path to a clone of github.ibm.com/touma/intrinsics_training with LFS pulled.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Directory to write train.jsonl and val.jsonl.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap rows loaded per source shard (transform stage).")
    p.add_argument("--subsample", type=int, default=None,
                   help="Total row count to subsample to after transform.")
    p.add_argument("--seed", type=int, default=None,
                   help="Shuffle/split seed. Default: the transform module's own (42).")
    p.add_argument("--test-size", type=float, default=None,
                   help="Validation fraction. Default: the transform module's own (0.1).")
    p.add_argument("--sr-config", type=Path, default=None,
                   help="SR training YAML. When given, over-length rows are dropped using "
                        "its model.base / data.max_length / data.enable_thinking / "
                        "adapter.last_token, so the filter cannot disagree with training.")
    p.add_argument("--no-drop-overlength", action="store_true",
                   help="Keep rows that exceed max_length (they will train unlabeled).")
    args = p.parse_args()

    tf_dir = args.intrinsics_repo / "standardization" / "data_transformation"
    if not tf_dir.exists():
        print(f"ERROR: transform module not found at {tf_dir}", file=sys.stderr)
        print("       (wrong intrinsics_training ref, or the clone is incomplete)", file=sys.stderr)
        return 2
    sys.path.insert(0, str(tf_dir))
    import transforms as T  # noqa: E402 -- standalone module, no package parent

    raw = args.intrinsics_repo / "data" / "train" / "answerability" / "raw"
    if not raw.exists():
        print(f"ERROR: raw data dir missing: {raw}", file=sys.stderr)
        return 2
    raw_files = [f for f in raw.iterdir() if f.is_file()]
    if not raw_files:
        print(f"ERROR: raw dir empty: {raw} (did `git lfs pull` run?)", file=sys.stderr)
        return 2
    # Guard against a pointer-only clone. Without this the transform would read
    # the pointer text as JSONL, produce zero usable rows, and the run would
    # train on nothing while still exiting 0.
    tiny = [f for f in raw_files if f.stat().st_size < _POINTER_MAX_BYTES]
    if tiny:
        print(f"ERROR: {len(tiny)}/{len(raw_files)} raw files are LFS pointers "
              f"(<{_POINTER_MAX_BYTES} bytes), not real data: "
              f"{[t.name for t in tiny[:3]]}", file=sys.stderr)
        print("       Run: git lfs pull -I 'data/train/answerability/**'", file=sys.stderr)
        return 2

    # Pass through only what was explicitly overridden, so unset flags keep the
    # transform module's own defaults rather than this script's idea of them.
    max_rows = args.max_rows if args.max_rows is not None else T.DEFAULT_MAX_ROWS
    fold_kwargs = {}
    if args.subsample is not None:
        fold_kwargs["subsample"] = args.subsample
    if args.seed is not None:
        fold_kwargs["seed"] = args.seed
    if args.test_size is not None:
        fold_kwargs["test_size"] = args.test_size

    print(f"[prep] raw dir: {raw}  ({len(raw_files)} shards)", flush=True)
    print(f"[prep] max_rows={max_rows} "
          f"subsample={fold_kwargs.get('subsample', T.SUBSAMPLE)} "
          f"seed={fold_kwargs.get('seed', T.SEED)} "
          f"test_size={fold_kwargs.get('test_size', T.TEST_SIZE)}", flush=True)

    cfg = {"name": "answerability", "dataset_path": str(raw)}
    ds = T.transform_answerability(cfg, max_rows=max_rows)
    print(f"[prep] transform produced {len(ds)} rows", flush=True)
    folds = T.finalize_folds(ds, **fold_kwargs)

    # Resolve the over-length filter from the SR config, so one file is the
    # single source of truth for max_length and the rendering options.
    tok = max_length = enable_thinking = last_token = None
    if args.sr_config is not None and not args.no_drop_overlength:
        from transformers import AutoTokenizer  # noqa: E402 -- optional, only for filtering

        from shadow_residual.config.training_config import load_training_config  # noqa: E402

        cfg = load_training_config(str(args.sr_config))
        max_length = cfg.data.max_length
        enable_thinking = cfg.data.enable_thinking
        last_token = cfg.adapter.last_token
        print(f"[prep] drop-overlength ON: max_length={max_length} "
              f"tokenizer={cfg.model.base} enable_thinking={enable_thinking} "
              f"last_token={last_token!r}", flush=True)
        tok = AutoTokenizer.from_pretrained(cfg.model.base)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    elif args.no_drop_overlength:
        print("[prep] drop-overlength DISABLED by flag", flush=True)
    else:
        print("[prep] drop-overlength OFF (no --sr-config given)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # intrinsics' finalize_folds names the held-out fold "test"; SR's config
    # refers to it as val.jsonl. Same data, different filename.
    for split_name, jsonl_name in (("train", "train.jsonl"), ("test", "val.jsonl")):
        out_path = args.out_dir / jsonl_name
        rows = []
        n_with_docs = 0
        for row in folds[split_name]:
            messages = list(row["prompt"]) + list(row["completion"])
            kwargs = row.get("chat_template_kwargs") or {}
            docs = kwargs.get("documents") if isinstance(kwargs, dict) else None
            out_row: dict = {"messages": messages}
            if docs:
                out_row["documents"] = docs
                n_with_docs += 1
            rows.append(out_row)

        _write(out_path, rows)
        n_before = len(rows)

        if tok is not None:
            keep = _fits(out_path, tok, max_length, enable_thinking, last_token)
            if len(keep) < n_before:
                rows = [r for i, r in enumerate(rows) if i in keep]
                _write(out_path, rows)
            dropped = n_before - len(rows)
            pct = (100.0 * dropped / n_before) if n_before else 0.0
            print(f"[prep] {jsonl_name}: dropped {dropped}/{n_before} rows "
                  f"({pct:.1f}%) over max_length={max_length}", flush=True)

        print(f"[prep] wrote {len(rows)} rows ({n_with_docs} with documents, pre-filter) "
              f"-> {out_path}", flush=True)
        if not rows:
            print(f"ERROR: {jsonl_name} is empty", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
