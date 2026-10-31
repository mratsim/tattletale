## CUDA: `{.exportc.}` entry-point name override at macroexpansion.
##
## Test that `{.exportc: "my_custom_entry".}` renames the emitted CUDA kernel entry
## and that a bare `{.exportc.}` kernel keeps its proc name.
## Compiling this file is the check: the consts run the cuda macro.
## A rename regression fails compilation.
## newEngine(runCuda) quits without a CUDA device, launch verification runs
## on the CUDA workstation.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/compiletime/t_cuda_exportc_entry.nim

import workspace/crucible

const renamedCuda = cuda:
  proc renamedKernel(C, A: ptr UncheckedArray[uint32]) {.global, exportc: "my_custom_entry".} =
    C[0] = A[0] + 1

const bareCuda = cuda:
  proc bareKernel(C, A: ptr UncheckedArray[uint32]) {.global, exportc.} =
    C[0] = A[0] + 2

