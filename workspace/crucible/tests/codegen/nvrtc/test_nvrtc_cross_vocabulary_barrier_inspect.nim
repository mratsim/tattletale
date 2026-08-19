## NVRTC: foreign barrier spellings in a CUDA kernel (compile + grep).
##
## The barrier aliases expand to the canonical `threadgroup_barrier` call
## during sem, and the CUDA printer lowers it to `__syncthreads()`.
## Written in foreign idioms, inspected without a CUDA device:
##   - Vulkan idiom `barrier()` (zero-arg) -> `__syncthreads()`
##   - OpenCL idiom `barrier(CLK_LOCAL_MEM_FENCE)` -> `__syncthreads()`
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/nvrtc --nimcache:nimcache/tests/nvrtc \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_cross_vocabulary_barrier_inspect.nim

import std/strutils
import workspace/crucible

const kernelCode = cuda:
  proc crossVocabBarrierKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let tid = int(thread_position_in_threadgroup.x)
    C[tid] = uint32(tid)
    barrier()                       # Vulkan-idiom zero-arg barrier
    C[tid + 128] = uint32(tid) * 2'u32
    barrier(CLK_LOCAL_MEM_FENCE)    # OpenCL-idiom flags barrier

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # Both foreign spellings must lower to the canonical CUDA barrier.
  doAssert kernelCode.count("__syncthreads()") == 2,
    "CUDA barrier missing or miscounted:\n" & kernelCode
  echo "  OK — foreign barrier spellings lower to __syncthreads() (compile + grep)"

when isMainModule:
  runTest()
