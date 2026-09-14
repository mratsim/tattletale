#!/usr/bin/env python3
"""Derive sidecar stats entries for the remaining tensor-mode enforcement sites,
computed from the committed fixture data, never from the code under test.

- the gdn one-shot row slices (rows 0..2, 3, 4) merge into the existing gdn-01 frame
- the moe JSON fixture rows become a new frame over routing weights, moe output and topk indices
- the bf16-03 layer files get stats frames over their recorded tensors (the routing weights replay reads them)

Run from the tests directory:

  uv run python testgen/gen_derive_stats_entries.py
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixture_stats import tensor_stats, write_text_zst, STATS_SCHEMA

import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L0 = os.path.join(ROOT, "fixtures", "bf16-01-layer-internals", "Qwen3.5-0.8B-layer-0")
A3BL0 = os.path.join(ROOT, "fixtures", "bf16-01-layer-internals",
                     "Qwen3.6-35B-A3B-layer-0")
B03 = os.path.join(ROOT, "fixtures", "bf16-03-full-forward-to-logits")


def read_frame(path):
    """Returns the frame dict the zstd file carries, {} for no file."""
    if not os.path.exists(path):
        return {}
    raw = subprocess.run(["zstd", "-dc", path], capture_output=True).stdout
    return json.loads(raw.decode("utf-8"))


def narrow(t, dim, start, length):
    """Returns t restricted to [start, start + length) along dim, contiguous."""
    sl = [slice(None)] * t.dim()
    sl[dim] = slice(start, start + length)
    return t[tuple(sl)].contiguous()


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
    """Derive the stats frames for the gdn slices, the moe rows and the layer payloads."""
    # --- the gdn one-shot row slices
    t = load_file(os.path.join(L0, "gdn-Qwen3.5-0.8B-01.safetensor"))
    one = t["one_shot_block_output"]
    collect(os.path.join(L0, "gdn-Qwen3.5-0.8B-01.safetensor.stats.json.zst"),
            "gdn-Qwen3.5-0.8B-01", [
                ("one_shot_block_output_steps0to2", narrow(one, 1, 0, 3)),
                ("one_shot_block_output_step3", narrow(one, 1, 3, 1)),
                ("one_shot_block_output_step4", narrow(one, 1, 4, 1)),
            ])

    # --- the moe JSON fixture rows
    moe_json = os.path.join(A3BL0, "moe_layer0_fixture.json.zst")
    raw = subprocess.run(["zstd", "-dc", moe_json], capture_output=True).stdout
    obj = json.loads(raw.decode("utf-8"))
    routing = torch.tensor(obj["routing_weights"], dtype=torch.float64).to(torch.bfloat16).contiguous()
    output = torch.tensor(obj["moe_output"], dtype=torch.float64).to(torch.bfloat16).contiguous()
    indices = torch.tensor(obj["topk_indices"], dtype=torch.int64).to(torch.float32).contiguous()
    collect(os.path.join(A3BL0, "moe_layer0_fixture.stats.json.zst"),
            "moe_layer0_fixture", [
                ("routing_weights", routing),
                ("moe_output", output),
                ("topk_indices", indices),
            ])

    # --- every recorded layer payload gets a stats frame over its float tensors,
    # recursive over the fixture roots the suites read
    import glob
    roots = [B03,
             os.path.join(ROOT, "fixtures", "bf16-01-layer-internals",
                          "Qwen3.6-35B-A3B-layer-3"),
             os.path.join(ROOT, "fixtures", "exl3-00-codec", "Qwen3-0.6B-EXL3-5bpw")]
    for root in roots:
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
