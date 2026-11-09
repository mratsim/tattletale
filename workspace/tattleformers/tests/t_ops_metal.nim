# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tattleformers ops on metal
##
## Test that the embed gather op and the Embedding layer run end to end on the Metal engine in bfloat16.
## The gather and the add op share one graph through the open eval-walk dispatch.
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wipM02 --outdir:build/wip \
##     workspace/tattleformers/tests/t_ops_metal.nim

import workspace/crucible
import workspace/ceramic
import workspace/ceramic/src/tile_algebra
import workspace/tattleformers/tests/ops_tests_template

proc main =
  runTest(gMetalEngine(), "Metal")

when isMainModule:
  main()
