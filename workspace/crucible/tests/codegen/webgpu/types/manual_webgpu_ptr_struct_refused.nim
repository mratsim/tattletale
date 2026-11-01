# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## WebGPU: a pointer-holding struct refuses at the WGSL declaration site.
##
## - v1 rendered `ptr<...>` struct members
## - WGSL has no pointer struct members: v1's emission is invalid WGSL
##
## Test that a struct type carrying a pointer field refuses, even when the WGSL
## lowering dissolves its only use into per-field leaf rewrites and no emitted
## declaration references it:
##
## - the struct local explodes into leaf symbols (the lowering accepts it)
## - the declaration site then refuses: `struct <name> holds a pointer`
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/types/manual_webgpu_ptr_struct_refused.nim

import workspace/crucible

type Wrapper = object
  data: ptr UncheckedArray[float32]
  n: int32

const code = webgpu:
  proc k(o: ptr UncheckedArray[float32]) {.global.} =
    var nn: int32 = 3
    var w = Wrapper(data: o, n: nn)
    o[1] = float32(w.n) + w.data[0]

proc runTest() =
  echo code
  echo "UNREACHABLE — a pointer-holding struct must refuse at the declaration site"

when isMainModule:
  runTest()
