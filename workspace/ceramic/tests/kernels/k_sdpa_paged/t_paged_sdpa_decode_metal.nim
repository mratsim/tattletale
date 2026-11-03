# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Paged decode SDPA fixture suite: every fixture replays in one binary.
## Each fixture's kernel output must match its recorded reference:
## - attention over the block-table-gathered KV entries [0, cache_seqlens[s])
## - causal off, GQA head mapping, layer 1 of the pool
##
## Run from the repository root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/t_paged_sdpa_decode_metal --nimcache:nimcache/t_paged_sdpa_decode_metal \
##     workspace/ceramic/tests/kernels/k_sdpa_paged/t_paged_sdpa_decode_metal.nim

import workspace/crucible
import workspace/ceramic/tests/kernels/k_sdpa_paged/kernel_sdpa_paged_template

proc replayAllPagedFixtures() =
  var engine = newEngine(runMetal)
  engine.replayPagedFixture(float16, 64,
    ["paged_decode_gqa_f16", "paged_decode_small_f16", "paged_window_decode_f16"])
  echo "all paged SDPA fixture replays pass (3 fixtures)"

replayAllPagedFixtures()
