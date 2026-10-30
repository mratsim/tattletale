## CUDA: the bit-reinterpret cast emits the float-as-integer intrinsics.
##
## Test that `cast[uint32](f)` and `cast[float32](u)` emit the CUDA bitcast intrinsics, not the C-style cast.
## The C-style cast numerically converts: 0xC0000000 read as a float is -2.0, read as an integer it is 3221225472.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/types/t_cuda_bit_cast.nim

import std/[strutils, unittest]
import workspace/crucible

const bitsToUintCuda = cuda:
  proc bitsToUintKernel(outBuf: ptr UncheckedArray[uint32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let f = float32(-2.0) - float32(lane)
    outBuf[lane] = cast[uint32](f)

const uintToBitsCuda = cuda:
  proc uintToBitsKernel(outBuf: ptr UncheckedArray[float32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let u = 0xC0000000'u32 + uint32(lane)
    outBuf[lane] = cast[float32](u)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "CUDA - bit-reinterpret cast emission":

    test "cast[uint32](float) emits __float_as_uint":
      check "__float_as_uint(" in bitsToUintCuda

    test "cast[float32](uint32) emits __uint_as_float":
      check "__uint_as_float(" in uintToBitsCuda

when isMainModule:
  runTest()
  echo "t_cuda_bit_cast: all checks passed"
