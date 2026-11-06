# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## HLSL: a whole-vector coordinate read inside the kernel body fails the compile
## with the refusal.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/functions/manual_hlsl_whole_vector_read_in_kernel_body.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[float32]) {.global.} =
    let g = thread_position_in_grid
    output[0] = float32(g.x)

proc runTest() =
  echo code
  echo "UNREACHABLE — a kernel-body refusal must fail the compile"

when isMainModule:
  runTest()
