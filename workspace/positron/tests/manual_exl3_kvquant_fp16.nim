## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## Paged KV quant/dequant vs the q_cache_kernels.cuh reference
## implementation on a layer-major pool: 3 deterministic paged
## geometries × bits {2,4,8} × {linear, LMCubic} round trips, the
## append-semantics and dequant-boundary cases, and the layer-1 vs
## layer-0 separation spot. Internals live in
## exl3_kvquant_test_utils.
##
## Run: nim cpp -r --hints:off --warnings:off \
##   --outdir:build/wip \
##   --nimcache:nimcache/wip \
##   workspace/positron/tests/manual_exl3_kvquant_fp16.nim

import workspace/crucible
import workspace/ceramic
import workspace/libtorch_testutils
import ../../ceramic/tests/tile_test_utils
import ./exl3_kvquant_test_utils

proc runTest(): bool =
  var shapes = initShapeCases("exl3_kvquant")
  var engine = bkMetal.init()
  engine.ingest(kvquantMsl)
  echo kvquantMsl            # keep the generated MSL inspectable
  let hAll = runKvQuantSuite(engine, shapes)
  doAssert hAll == 0xEC3758D5'u32,
    "the combined quant+dequant hash drifted"
  result = true

when isMainModule:
  runCppTest("kvquant vs the q_cache_kernels.cuh reference + layer-major pool", runTest)
