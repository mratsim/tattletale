## CUDA printer: `__forceinline__ ` on an `{.inline.}` device function.
##
## Test that the CUDA printer emits a non-kernel `{.inline.}` proc that reads
## no coordinate builtin as a `__forceinline__ __device__` function.
## Compile-time only: semantic analysis produces the text, no GPU and no launch are involved.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/functions/t_cuda_inline_hint.nim

import std/strutils
import workspace/crucible

const inlineHintCuda = cuda:
  proc addv(a, b: int32): int32 {.device, inline.} =
    result = a + b

  proc scaleKernel(output: ptr UncheckedArray[int32]) {.global.} =
    output[0] = addv(3, 4)

proc runTest() =
  doAssert "__forceinline__ __device__ int addv(int a, int b)" in inlineHintCuda,
    "the CUDA printer must emit the inline device function, got:\n" & inlineHintCuda
  doAssert '}' in inlineHintCuda and "__global__ void scaleKernel" in inlineHintCuda,
    "the calling kernel must still be emitted, got:\n" & inlineHintCuda

when isMainModule:
  runTest()
