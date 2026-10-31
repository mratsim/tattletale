## CUDA: whole-array assignment decomposition to `__builtin_memcpy`.
##
## Test that the CUDA pipeline accepts a kernel whose body copies one whole
## array to another in three shapes: assignment, declaration init, and a copy
## nested in a branch.
##
## Compile-time only: the const runs the cuda macro during `nim c`.
## A pipeline or printer regression fails compilation.
##
## This machine has no CUDA driver.
## t_metal_array_assign and t_opencl_array_assign carry the semantic check
## on Metal and on OpenCL. A CUDA launch reuses the same kernel body.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/control_flow/t_cuda_array_assign.nim

import workspace/crucible

const arrayAssignCuda = cuda:
  proc copyKernel(output: ptr UncheckedArray[float32]) {.global.} =
    var b: array[8, float32]
    for i in 0 ..< 8:
      b[i] = float32(i) * 1.5'f32 + 0.25'f32
    # whole-array assignment
    var a: array[8, float32]
    a = b
    # declaration init from another array
    var c = b
    # whole-array copy nested in a branch
    var d: array[8, float32]
    let t = thread_position_in_grid.x
    if t == 0:
      d = a
    for i in 0 ..< 8:
      d[i] = d[i] + 1.0'f32
    output[t] = a[3] + c[4] + d[t]
