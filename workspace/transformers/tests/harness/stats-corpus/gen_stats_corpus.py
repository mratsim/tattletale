#!/usr/bin/env python3
"""Generate the cross-implementation fingerprint corpus for selftest.nim.

Justification header: no Qwen3 analog exists. The corpus carries the
cross-implementation contract of the fingerprint machinery
(harness/SPEC.md Fingerprints): fixture_stats.py, the python twin of
harness/gen_stats.nim, emits fingerprint stats over the committed tensor
bytes, and harness/selftest.nim recomputes the fingerprints in Nim and
requires bit-exact agreement on order statistics and histograms. The
corpus also carries the descriptor contract of the fixture rework: two
entries with the descriptor keys, one per comparison class, verified
bit-exactly the same way.

Deterministic: torch.manual_seed with the file-level seed constant, CPU
only, no model load. Run from tests/ inside the uv env:

    uv run python harness/stats-corpus/gen_stats_corpus.py
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "testgen"))

import platform

import torch  # noqa: E402

from fixture_stats import write_provenance, write_stats_file  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 20260909


def build_bf16_tensor(gen):
    """bf16 corpus tensor: seeded values plus injected specials covering the
    zero bin, the subnormal bin, soft mantissa splits and binade edges."""
    vals = torch.randn(4096, generator=gen, dtype=torch.float32) * 0.7
    specials = torch.tensor([
        0.0, -0.0,
        # smallest bf16 subnormal and its neighbor
        2.0 ** -133, -2.0 ** -133, 2.0 ** -132,
        # binade edges with odd mantissas, exercising the soft split
        1.0078125, -1.0078125, 1.015625, 3.9921875, -3.9921875,
        2.0 ** -64, -(2.0 ** -64) * 1.5, 2.0 ** 63 * 1.5, -(2.0 ** 63),
        # values straddling the f32-to-bf16 rounding boundary
        1.0039060058593750, -1.0039060058593750,
    ], dtype=torch.float32)
    return torch.cat([vals, specials]).to(torch.bfloat16)


def build_f32_tensor(gen):
    """f32 corpus tensor in the moderate binade range plus one whitelisted
    -Inf. The whitelist is a quantile-only path (SPEC.md): -Inf stays
    unbucketed, so this tensor carries no histogram."""
    vals = torch.randn(2048, generator=gen, dtype=torch.float32) * 1.3
    specials = torch.tensor([
        0.0, -0.0, 1.0039060058593750, -1.0039060058593750,
        2.0 ** -64, 2.0 ** 63,
        float("-inf"),
    ], dtype=torch.float32)
    return torch.cat([vals, specials])


def build_f32_hist_tensor(gen):
    """f32 corpus tensor with a histogram: exercises the f32-to-bf16
    round-to-nearest-even rounding inside the bucket pass, no -Inf."""
    vals = torch.randn(2048, generator=gen, dtype=torch.float32) * 1.1
    specials = torch.tensor([
        0.0, -0.0, 1.0039060058593750, -1.0039060058593750,
        2.0 ** -64, 2.0 ** 63, 3.9921875,
    ], dtype=torch.float32)
    return torch.cat([vals, specials])


def main():
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    bf16 = build_bf16_tensor(gen)
    f32 = build_f32_tensor(gen)
    f32h = build_f32_hist_tensor(gen)
    from safetensors.torch import save_file
    save_file({"corpus_bf16": bf16.contiguous(),
               "corpus_f32": f32.contiguous(),
               "corpus_f32_hist": f32h.contiguous()},
              os.path.join(OUT_DIR, "crossimpl.safetensor"))

    # corpus_bf16 and corpus_f32_hist also carry descriptor entries: the exact and drift classes of the fixture
    # contract, over tensors that stay committed. The descriptor byte law stays verifiable from committed
    # bytes in both directions: the selfcheck regenerates this file, the selftest recomputes in Nim
    # and requires bit-exact agreement on every descriptor field.
    entries = [
        ("corpus_bf16", bf16, True, False),
        ("corpus_bf16", bf16, False, False, "exact"),
        ("corpus_f32", f32, False, True),
        ("corpus_f32_hist", f32h, True, False),
        ("corpus_f32_hist", f32h, False, False, "drift"),
    ]
    write_stats_file(os.path.join(OUT_DIR, "crossimpl.safetensor.stats.json.zst"),
                     "crossimpl.safetensor", entries)
    write_provenance(os.path.join(OUT_DIR, "PROVENANCE.md"), [
        ("date", "2026-09-09"),
        ("generator", "harness/stats-corpus/gen_stats_corpus.py"),
        ("python", platform.python_version()),
        ("recorded_from", "m4max-cpu"),
        ("seed", str(SEED)),
        ("torch", torch.__version__),
        ("transformers", "none (synthetic corpus, no model load)"),
    ])
    print("wrote corpus with %d + %d + %d elements" % (bf16.numel(), f32.numel(), f32h.numel()))


if __name__ == "__main__":
    main()
