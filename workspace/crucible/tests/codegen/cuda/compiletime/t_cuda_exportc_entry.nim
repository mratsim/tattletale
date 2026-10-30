## CUDA: `{.exportc.}` entry-point name override at macroexpansion.
##
## Test that `{.exportc: "my_custom_entry".}` renames the emitted CUDA kernel
## entry to `extern "C" __global__ void my_custom_entry` and that a bare
## `{.exportc.}` kernel keeps its proc name.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/compiletime/t_cuda_exportc_entry.nim

import std/strutils
import workspace/crucible

const renamedCuda = cuda:
  proc renamedKernel(C, A: ptr UncheckedArray[uint32]) {.global, exportc: "my_custom_entry".} =
    C[0] = A[0] + 1

const bareCuda = cuda:
  proc bareKernel(C, A: ptr UncheckedArray[uint32]) {.global, exportc.} =
    C[0] = A[0] + 2

static:
  # compile-time signals: a rename regression fails compilation here
  doAssert "extern \"C\" __global__ void my_custom_entry(" in renamedCuda
  doAssert "void renamedKernel(" notin renamedCuda
  doAssert "extern \"C\" __global__ void bareKernel(" in bareCuda
  doAssert "void my_custom_entry(" notin bareCuda
