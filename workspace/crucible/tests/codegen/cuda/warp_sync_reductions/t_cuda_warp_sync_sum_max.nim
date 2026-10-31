## CUDA: `reduceAdd`/`reduceMax` warp-synchronous reduction templates.
##
## Test that a warp-reduction call in statement position and in expression
## position compile through the cuda macro.
##
## The check compiles this file: the consts run the cuda macro.
## A template-emission regression fails compilation.
## This host has no NVRTC and no NVIDIA GPU.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/warp_sync_reductions/t_cuda_warp_sync_sum_max.nim

import workspace/crucible

const statementCuda = cuda:
  proc sumStatement(outBuf: ptr UncheckedArray[float32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let v = float32(lane)
    discard reduceAdd(v)

const expressionCuda = cuda:
  proc sumExpression(outBuf: ptr UncheckedArray[float32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let v = float32(lane)
    outBuf[lane] = reduceAdd(v)

const maxStatementCuda = cuda:
  proc maxStatement(outBuf: ptr UncheckedArray[float32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let v = float32(lane)
    discard reduceMax(v)

