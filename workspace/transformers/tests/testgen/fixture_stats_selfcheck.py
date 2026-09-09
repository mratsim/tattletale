#!/usr/bin/env python3
"""Cross-implementation selfcheck: regenerate the four validation stats files
with fixture_stats.py and byte-compare their content against the committed
frames written by harness/gen_stats.nim. Run from tests/ inside the uv env:

    uv run python testgen/fixture_stats_selfcheck.py

The byte law is content-exact: the fresh payload and the committed
frame inflate to identical bytes, the zstd frame carries the bytes and
defines nothing, the JSON payload inside is the format.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from fixture_stats import (  # noqa: E402
    read_text_zst,
    stats_file_bytes,
    write_text_zst,
)

RECORDED = [
    ("fixtures/bf16-01-layer-internals/Qwen3.5-0.8B-layer-3/rope-Qwen3.5-0.8B-00.safetensor",
     [("q_rot", True), ("k_rot", False)]),
    ("fixtures/bf16-01-layer-internals/Qwen3.5-0.8B-layer-3/attn-Qwen3.5-0.8B-00.safetensor",
     [("output", True)]),
    ("fixtures/bf16-01-layer-internals/Qwen3.6-35B-A3B-layer-0/gdn-Qwen3.6-35B-A3B-00.safetensor",
     [("output_seq", True)]),
    ("fixtures/bf16-01-layer-internals/Qwen3.6-35B-A3B-layer-0/layer-Qwen3.6-35B-A3B-00.safetensor",
     [("gdn_block_output_seq", False), ("moe_output", False),
      ("layer_output_seq", True)]),
    # Descriptor contract entries, one per comparison class, over corpus tensors that stay
    # recorded: the descriptor byte law round-trips from the recorded files in both directions.
    ("harness/stats-corpus/crossimpl.safetensor",
     # The committed corpus carries one entry per tensor name. The byte
     # law covers the committed name of every entry, namely
     # the descriptor-exact corpus_bf16 and the descriptor-drift
     # corpus_f32_hist, plus the whitelisted corpus_f32. The with_hist
     # variants of the two descriptor names were dead entries under the old
     # dict last-wins behavior: never committed, dropped here.
     [("corpus_bf16", False, "exact"),
      ("corpus_f32", False, None, True),
      ("corpus_f32_hist", False, "drift")]),
]


def committed_payload(path):
    """Payload bytes of the committed stats sidecar.
    Transitional until the sidecar file flip: the recorded frame
    name resolves first, the retired `.json` path reads raw."""
    frame = path + ".stats.json.zst"
    if os.path.exists(frame):
        return frame, read_text_zst(frame)
    legacy = path + ".stats.json"
    with open(legacy, "rb") as fh:
        return legacy, fh.read()


def main():
    tests_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    failures = 0
    for rel, names in RECORDED:
        path = os.path.join(tests_dir, rel)
        base = os.path.basename(path)
        entries = []
        with safe_open(path, framework="pt") as fh:
            for entry in names:
                # Entry tuple forms: (name, with_hist). Descriptor and whitelist
                # fields are appended: (name, with_hist, desc_mode, allow_minus_inf).
                name, with_hist = entry[0], entry[1]
                desc_mode = entry[2] if len(entry) > 2 else None
                allow_minus_inf = entry[3] if len(entry) > 3 else False
                entries.append((name, fh.get_tensor(name), with_hist,
                                allow_minus_inf, desc_mode))
        got = stats_file_bytes(base, entries).encode("utf-8")
        # The fresh frame is written under build/ for mismatch diffing only,
        # never committed.
        fresh = os.path.join(tests_dir, "build", "fixture-stats-check",
                             base + ".stats.json.zst")
        os.makedirs(os.path.dirname(fresh), exist_ok=True)
        write_text_zst(fresh, got)
        committed, want = committed_payload(path)
        if got == want:
            print("byte-exact (decompressed content): " + committed)
        else:
            failures += 1
            print("MISMATCH: " + committed)
            for i, (a, b) in enumerate(zip(got, want)):
                if a != b:
                    print("  first diff at byte %d: got %r want %r" % (i, got[i:i + 24], want[i:i + 24]))
                    break
            print("  got len %d, want len %d" % (len(got), len(want)))
    if failures:
        sys.exit("%d validation stats file(s) mismatch" % failures)
    print("all validation stats files byte-exact against gen_stats.nim output")


if __name__ == "__main__":
    main()
