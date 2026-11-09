# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tensorcery erased tensor layer on metal
##
## Test that the erased user face (newTensor, copies, add, split, eval) runs end to end on the Metal engine in bfloat16.
##
## Run from the worktree root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/tensorcery/tests/t_tensor_metal.nim

import workspace/crucible
import workspace/ceramic
import workspace/tensorcery/tests/tensor_tests_template

proc main =
  runTest(gMetalEngine(), "Metal")

when isMainModule:
  main()
