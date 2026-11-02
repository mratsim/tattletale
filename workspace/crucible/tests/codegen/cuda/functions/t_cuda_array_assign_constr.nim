## CUDA: whole-array assignment from a constructor decomposes to `__builtin_memcpy`.
##
## Test that the CUDA pipeline accepts a kernel that assigns an array
## constructor to a whole array and reads every element back.
##
## The const runs the cuda macro during `nim c`, so the check compiles
## without launching. Pipeline or printer regressions fail compilation.
## The runtime check runs on the CUDA box.
## `output[i]` reads `float32(i) + 1.0`.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/functions/t_cuda_array_assign_constr.nim

import workspace/crucible

const constrAssignCuda = cuda:
  proc kern(output: ptr UncheckedArray[float32]) {.global.} =
    var a: array[8, float32]
    a = [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32, 5.0'f32, 6.0'f32, 7.0'f32, 8.0'f32]
    let t = thread_position_in_grid.x
    output[t] = a[t]
