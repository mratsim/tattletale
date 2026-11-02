## CUDA: a device function's `{.smem.}` local refuses an initializer, CUDA C++ forbids initializing `__shared__` variables.
##
## - v1 emitted `__shared__ T x = init` verbatim in any proc
## - nvcc rejects an initialized `__shared__` declaration: v1's emission is invalid C++
##
## Test that a device function declaring a `{.smem.}` local with an initializer refuses at codegen.
## An initializer-less `__shared__` local in a device function is legal CUDA C++ and emits.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/device/manual_cuda_devfn_smem_init_rejected.nim

import workspace/crucible

const code = cuda:
  proc f(output: ptr UncheckedArray[uint32]) {.inline.} =
    var scratch {.smem.}: array[4, uint32] = [1'u32, 2, 3, 4]
    scratch[0] = output[0]
    output[1] = scratch[0]
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    f(output)

proc runTest() =
  echo code
  echo "UNREACHABLE — a device function {.smem.} local with an initializer must be refused at compile time"

when isMainModule:
  runTest()
