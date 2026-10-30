## CUDA: `reduceAdd`/`reduceMax` warp-synchronous reduction templates.
##
## Test that a warp-reduction call in statement position emits the butterfly template at module scope, like an expression-position call does.
## Assertions read the emitted source: this host has no NVRTC and no NVIDIA GPU, so the template text is the observable contract.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/warp_sync_reductions/t_cuda_warp_sync_sum_max.nim

import std/[strutils, unittest]
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

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "CUDA - warp reduction template emission":

    test "statement-position reduceAdd emits the butterfly template":
      check "template <typename T> __forceinline__ __device__ T warpReduceAdd" in statementCuda
      check "warpReduceAdd(" in statementCuda

    test "expression-position reduceAdd emits the butterfly template":
      check "template <typename T> __forceinline__ __device__ T warpReduceAdd" in expressionCuda
      check "warpReduceAdd(" in expressionCuda

    test "statement-position reduceMax emits the butterfly template":
      check "template <typename T> __forceinline__ __device__ T warpReduceMax" in maxStatementCuda
      check "warpReduceMax(" in maxStatementCuda

when isMainModule:
  runTest()
  echo "t_cuda_warp_sync_sum_max: all checks passed"
