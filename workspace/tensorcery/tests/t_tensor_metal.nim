# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tensorcery erased tensor layer on metal
##
## Test that the user face (toTensor, add, split, embed_gather, eval) runs end to end on Metal
## in bfloat16 at ranks 1 to 5, enqueueing each producer once per eval.
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/tensorcery/tests/t_tensor_metal.nim

import workspace/crucible
import workspace/ceramic
import workspace/tensorcery/tests/tensor_tests_template

proc main =
  runTest(runMetal, "Metal")
  runDedupeTest(runMetal, "Metal")
  runEmbedGatherTest(runMetal, "Metal")
  runRankNTest(runMetal, "Metal")

when isMainModule:
  main()
