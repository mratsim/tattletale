# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Jagged attention bf16 suite (Metal) at the Qwen3-0.6B statics.
##
## Test that every scenario replays inside the pre-stated bf16 band vs an f64 host reference.
## Coverage: decode and prefill segments, GQA head division, causal bands, partial last pages, bit-identical relaunches, zero sentinels.
##
## Run from the worktree root:
##   nim c -r -d:release --threads:off --hints:off --warnings:off \
##     -o:build/tests/t_jagged_attention_bf16_metal \
##     --nimcache:nimcache/t_jagged_attention_bf16_metal \
##     workspace/positron/tests/k_jagged_attention/t_jagged_attention_bf16_metal.nim

import workspace/crucible
import kernel_jagged_attention_bf16

proc replayAllBf16Scenarios() =
  var engine = newEngine(runMetal)

  # decode: one segment, q_len 1, kv_len 300 over 2 pages, the last page partially covered (300 = 256 + 44), one spare page pure garbage
  engine.bf16JaggedReplayCase("bf16_decode_partial_last_page", Bf16JaggedCase(
    qLens: @[1],
    kvLens: @[300],
    pageIds: @[0, 1, 2],
    kvPageBounds: @[0, 2],
    numPages: 3,
    numLayers: 2,
    layer: 0))

  # prefill: one segment, q_len 8 == kv_len 8, 8 of page 0's 256 slots covered
  engine.bf16JaggedReplayCase("bf16_prefill_single_page", Bf16JaggedCase(
    qLens: @[8],
    kvLens: @[8],
    pageIds: @[0],
    kvPageBounds: @[0, 1],
    numPages: 2,
    numLayers: 2,
    layer: 1))

  # mixed batch sharing a prefix page, GQA 16Q/8KV:
  # - segment 0: prefill q_len 8, kv_len 8, page [0]
  # - segment 1: prefill q_len 8, kv_len 264, pages [0, 2], shared page 0, second page partially covered (264 = 256 + 8)
  # - segment 2: decode q_len 1, kv_len 520, pages [0, 1, 3], shared page 0 (520 = 2*256 + 8)
  # - page 3's tail and every unowned slot hold NaN garbage
  engine.bf16JaggedReplayCase("bf16_mixed_shared_prefix", Bf16JaggedCase(
    qLens: @[8, 8, 1],
    kvLens: @[8, 264, 520],
    pageIds: @[0, 0, 2, 0, 1, 3],
    kvPageBounds: @[0, 1, 3, 6],
    numPages: 4,
    numLayers: 2,
    layer: 0))

  # gridX = 1 launches one workgroup per (head, segment), the grid-stride loop covers every query tile of the mixed scenario
  engine.bf16JaggedReplayCase("bf16_mixed_grid_stride", Bf16JaggedCase(
    qLens: @[8, 8, 1],
    kvLens: @[8, 264, 520],
    pageIds: @[0, 0, 2, 0, 1, 3],
    kvPageBounds: @[0, 1, 3, 6],
    numPages: 4,
    numLayers: 2,
    layer: 0),
    gridX = 1)

  echo "all bf16 jagged attention scenario replays pass"

replayAllBf16Scenarios()
