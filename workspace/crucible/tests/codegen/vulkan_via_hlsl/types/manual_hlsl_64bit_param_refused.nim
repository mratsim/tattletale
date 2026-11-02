# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## HLSL: a 64-bit integer kernel parameter refuses on the scalar-names channel.
##
## Test that an int64 kernel parameter refuses at the push-block declaration:
## the hlsl scalar names have no 64-bit integer cell. The names lookup returns
## no name and the declaration emitter refuses:
##
## - `int64 has no HLSL type name`
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/types/manual_hlsl_64bit_param_refused.nim

import workspace/crucible

const code = hlsl:
  proc k(output: ptr UncheckedArray[int64]) {.global.} =
    output[0] = 7'i64

proc runTest() =
  echo code
  echo "UNREACHABLE — a 64-bit integer kernel parameter must refuse"

when isMainModule:
  runTest()
