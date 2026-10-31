## CUDA printer: `__forceinline__ ` on an `{.inline.}` device function.
##
## Test that the CUDA printer accepts a non-kernel `{.inline.}` proc that reads
## no coordinate builtin.
## Compile-time only: the const runs the cuda macro at compile time.
## A printer regression fails compilation. No GPU and no launch are involved.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/functions/t_cuda_inline_hint.nim

import workspace/crucible

const inlineHintCuda = cuda:
  proc addv(a, b: int32): int32 {.device, inline.} =
    result = a + b

  proc scaleKernel(output: ptr UncheckedArray[int32]) {.global.} =
    output[0] = addv(3, 4)

