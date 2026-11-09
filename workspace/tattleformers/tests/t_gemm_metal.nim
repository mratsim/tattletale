# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Test that the Qwen3-0.6B GEMM smokes pass on Metal
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wipM03 --outdir:build/wip \
##     workspace/tattleformers/tests/t_gemm_metal.nim

import workspace/crucible
import workspace/ceramic
import workspace/ceramic/src/tile_algebra
import workspace/tattleformers/tests/gemm_shapes_template

proc main =
  runGemmSmokes(gMetalEngine(), "Metal")

when isMainModule:
  main()
