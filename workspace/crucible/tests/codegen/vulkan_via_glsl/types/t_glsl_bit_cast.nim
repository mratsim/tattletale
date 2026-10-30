## GLSL: the bit-reinterpret cast emits the float-bits builtins.
##
## Test that `cast[uint32](f)` and `cast[float32](u)` emit `floatBitsToUint`/`uintBitsToFloat`, not the `T()` constructor.
## `T()` numerically converts, builtins reinterpret the bits.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/types/t_glsl_bit_cast.nim

import std/[strutils, unittest]
import workspace/crucible

const bitsToUintGlsl = glsl:
  proc bitsToUintKernel(outBuf: ptr UncheckedArray[uint32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let f = float32(-2.0) - float32(lane)
    outBuf[lane] = cast[uint32](f)

const uintToBitsGlsl = glsl:
  proc uintToBitsKernel(outBuf: ptr UncheckedArray[float32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let u = 0xC0000000'u32 + uint32(lane)
    outBuf[lane] = cast[float32](u)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "GLSL - bit-reinterpret cast emission":

    test "cast[uint32](float) emits floatBitsToUint":
      check "floatBitsToUint(" in bitsToUintGlsl

    test "cast[float32](uint32) emits uintBitsToFloat":
      check "uintBitsToFloat(" in uintToBitsGlsl

when isMainModule:
  runTest()
  echo "t_glsl_bit_cast: all checks passed"
