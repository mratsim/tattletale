#!/usr/bin/env python3
"""Derive sidecar stats entries for the remaining tensor-mode enforcement sites,
computed from the committed fixture data, never from the code under test.

- the bf16-03 layer files get stats frames over their recorded tensors (the routing weights replay reads them)
- the exl3-00 codec payload files get stats frames over their float tensors

Run from the tests directory:

  uv run python testgen/gen_derive_stats_entries.py
"""

import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixture_stats import tensor_stats, write_text_zst, STATS_SCHEMA

import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B03 = os.path.join(ROOT, "fixtures", "bf16-03-full-forward-to-logits")
EXL3_CODEC = os.path.join(ROOT, "fixtures", "exl3-00-codec", "Qwen3-0.6B-EXL3-5bpw")


def read_frame(path):
    """Returns the frame dict the zstd file carries, {} for no file."""
    if not os.path.exists(path):
        return {}
    raw = subprocess.run(["zstd", "-dc", path], capture_output=True).stdout
    return json.loads(raw.decode("utf-8"))


def collect(frame_path, source, items):
    """Merges the (name, tensor) items into the stats frame at frame_path.

    Args:
    - frame_path, source, items, the frame file, its source label and the new
      (name, tensor) pairs

    Existing entries keep their recorded bytes verbatim, only the new
    items run through tensor_stats.
    """
    old = read_frame(frame_path)
    tensors = dict(old.get("tensors", {}))
    for name, t in items:
        if name in tensors:
            continue  # idempotent rerun, the recorded bytes stay
        tensors[name] = tensor_stats(t, name)
    src = old.get("source", source)
    payload = json.dumps(
        {"schema": STATS_SCHEMA, "source": src, "tensors": tensors},
        separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
    write_text_zst(frame_path, payload)
    print("frame", os.path.relpath(frame_path, ROOT), "->", sorted(tensors))


def main():
    """Derive the stats frames for every recorded layer payload, recursive
    over the fixture roots the suites read."""
    for root in (B03, EXL3_CODEC):
        for path in sorted(glob.glob(os.path.join(root, "**", "*.safetensor"),
                                     recursive=True)):
            stem = path[:-len(".safetensor")]
            t = load_file(path)
            items = [(k, v.contiguous()) for k, v in sorted(t.items())
                     if v.dtype in (torch.bfloat16, torch.float16, torch.float32)]
            collect(stem + ".safetensor.stats.json.zst", os.path.basename(stem),
                    items)


if __name__ == "__main__":
    main()
