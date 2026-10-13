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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # intrinsics' finalize_folds names the held-out fold "test"; SR's config
    # refers to it as val.jsonl. Same data, different filename.
    for split_name, jsonl_name in (("train", "train.jsonl"), ("test", "val.jsonl")):
        out_path = args.out_dir / jsonl_name
        n = n_with_docs = 0
        with out_path.open("w") as f:
            for row in folds[split_name]:
                messages = list(row["prompt"]) + list(row["completion"])
                kwargs = row.get("chat_template_kwargs") or {}
                docs = kwargs.get("documents") if isinstance(kwargs, dict) else None
                out_row: dict = {"messages": messages}
                if docs:
                    out_row["documents"] = docs
                    n_with_docs += 1
                f.write(json.dumps(out_row) + "\n")
                n += 1
        print(f"[prep] wrote {n} rows ({n_with_docs} with documents) -> {out_path}", flush=True)
        if n == 0:
            print(f"ERROR: {jsonl_name} is empty", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
