# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests backend detection within a kernel.
##
## Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_ccgetbackend.nim

import std/[strutils, unittest]
import workspace/crucible

func getTileConfig(T: typedesc): int =
  ## Resolves at instantiation to a per-backend, per-element-type constant.
  when ccGetBackend() == ctMetal:
    when T is float16: 303
    elif T is bfloat16: 304
    else: 305
  elif ccGetBackend() == ctCuda:
    when T is float16: 202
    elif T is bfloat16: 203
    else: 204
  else:
    when T is float16: 101
    elif T is bfloat16: 102
    else: 103

const dispatchCode = metal:
  proc backendProbe(output: ptr UncheckedArray[int]) {.global.} =
    when ccGetBackend() == ctOpenCL: output[0] = 1
    elif ccGetBackend() == ctVulkan: output[0] = 2
    elif ccGetBackend() == ctMetal: output[0] = 3
    elif ccGetBackend() == ctCuda: output[0] = 4
    else: output[0] = 5
    const fp16Tiles = getTileConfig(float16)
    const bf16Tiles = getTileConfig(bfloat16)
    output[1] = fp16Tiles
    output[2] = bf16Tiles

# Dispatch selected at instantiation on `ccGetBackend()`.
# The dispatch lives in a generic proc body outside the DSL block.
proc probeBackend[T](x: T): int =
  when ccGetBackend() == ctMetal: 404
  elif ccGetBackend() == ctCuda: 303
  elif ccGetBackend() == ctVulkan: 202
  elif ccGetBackend() == ctOpenCL: 101
  else: 505

const genericCode = metal:
  proc genericProbe(output: ptr UncheckedArray[int]) {.global.} =
    const fp16Tiles = getTileConfig(float16)
    output[0] = probeBackend(output[0])
    output[1] = fp16Tiles

# The return annotation must be `typedesc`.
# Inside a generic proc body, the template expansion is resolved from the raw body.
# The type param is still symbolic there.
# Crucible's type resolver unwraps the expansion only in the typedesc-return form.
# The `untyped`-return form reaches the resolver as a bare StmtListExpr, which is rejected.
template rt(T; A: untyped = getTileConfig(T)): typedesc =
  array[A, T]

proc genericBody[T](output: ptr UncheckedArray[int]) =
  var tile: rt(T)
  output[0] = tile.len

const tileDefaultCode = metal:
  proc tileProbe(output: ptr UncheckedArray[int]) {.global.} =
    genericBody[float16](output)

proc runTest() =
  suite "Metal - ccGetBackend":

    test "dispatch folds to ctMetal, config folds to 303/304, on-device":
      doAssert "output[0] = 3;" in dispatchCode
      for other in ["output[0] = 1;", "output[0] = 2;", "output[0] = 4;", "output[0] = 5;"]:
        doAssert other notin dispatchCode
      doAssert "output[1] = 303;" in dispatchCode
      doAssert "output[2] = 304;" in dispatchCode
      doAssert "getTileConfig" notin dispatchCode
      doAssert "when" notin dispatchCode
      doAssert "ccGetBackend" notin dispatchCode
      var engine = bkMetal.init()
      engine.ingest(dispatchCode)
      var res: array[3, int32]
      engine.run("backendProbe", res, ())
      check res[0] == 3'i32
      check res[1] == 303'i32
      check res[2] == 304'i32

    test "generic proc body folds at instantiation, on-device":
      doAssert "result = 404;" in genericCode
      doAssert "output[1] = 303;" in genericCode
      doAssert "when" notin genericCode
      doAssert "ccGetBackend" notin genericCode
      var engine = bkMetal.init()
      engine.ingest(genericCode)
      var res: array[2, int32]
      engine.run("genericProbe", res, ())
      check res[0] == 404'i32
      check res[1] == 303'i32

    test "template tile default folds to the config-derived type":
      doAssert "doAssert" notin tileDefaultCode
      doAssert "getTileConfig" notin tileDefaultCode
      doAssert "ccGetBackend" notin tileDefaultCode
      doAssert "when" notin tileDefaultCode
      doAssert "tile[303]" in tileDefaultCode
      doAssert "len(tile)" in tileDefaultCode
      doAssert "output[0] = 303;" notin tileDefaultCode

when isMainModule:
  runTest()
