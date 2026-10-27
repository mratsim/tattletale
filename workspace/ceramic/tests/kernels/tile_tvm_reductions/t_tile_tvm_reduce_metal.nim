# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal tile axis reduce through the atom's thread->value map
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/ceramic/tests/kernels/tile_tvm_reductions/t_tile_tvm_reduce_metal.nim

import workspace/crucible
import workspace/ceramic/tests/kernels/tile_tvm_reductions/tile_tvm_reductions_template

proc main =
  var engine = newEngine(runMetal)
  run_tile_reduce_tests(engine, moduleName = "t_tile_tvm_reduce_metal")

when isMainModule:
  main()
