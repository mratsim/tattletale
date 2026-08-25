## CUDA: reduction builtin emission pin.
##
## There is no nvcc/NVRTC on this machine (M4), so the pin asserts the
## exact emitted text: `__shfl_down_sync(0xffffffff, v, delta, 32)` and
## `__shfl_sync(0xffffffff, v, lane, 32)` with the full-mask literal
## `0xffffffff` and the width 32 spelled verbatim.
##
## Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_reduction_builtins.nim

import std/strutils
import workspace/crucible

const kernelCode = cuda:
  proc reductionKernel(output: ptr UncheckedArray[float32]) {.global.} =
    let acc = output[0]
    let v = simdShuffleDown(acc, 1'u32)
    output[1] = v
    let w = simdShuffle(v, 0'u32)
    output[2] = w

proc runTest() =
  doAssert "__shfl_down_sync(0xffffffff, acc, 1U, 32)" in kernelCode,
    "CUDA shuffleDown spelling missing:\n" & kernelCode
  doAssert "__shfl_sync(0xffffffff, v, 0U, 32)" in kernelCode,
    "CUDA shuffle spelling missing:\n" & kernelCode
  doAssert "0xffffffff" in kernelCode,
    "CUDA mask literal must stay 0xffffffff (all lanes active):\n" & kernelCode
  echo "  OK — CUDA reduction builtin emission (__shfl_down_sync / __shfl_sync, full mask, width 32)"

when isMainModule:
  runTest()
