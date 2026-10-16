## NVRTC: cross-vocabulary coordinate + barrier inspect (compile + grep)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/functions/t_nvrtc_cross_vocabulary_inspect.nim
import std/strutils
import workspace/crucible

const kernelCode = cuda:
  proc crossVocabKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let gid = get_global_id(0)
    let gid2 = get_global_id(0) * 2
    C[gid] = uint32(gid)
    C[gid2] = uint32(gid2)
    C[thread_index_in_threadgroup] = uint32(thread_index_in_threadgroup)
    workgroupBarrier()

## Foreign barrier spellings (`barrier()`, `barrier(CLK_LOCAL_MEM_FENCE)`)
## lower to the canonical `threadgroup_barrier` in sem.
## The CUDA printer emits `__syncthreads()` for it.
##   - Vulkan zero-arg `barrier()`, OpenCL `barrier(CLK_LOCAL_MEM_FENCE)`
const barrierKernelCode = cuda:
  proc crossVocabBarrierKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let tid = int(thread_position_in_threadgroup.x)
    C[tid] = uint32(tid)
    barrier()                       # Vulkan-idiom zero-arg barrier
    C[tid + 128] = uint32(tid) * 2'u32
    barrier(CLK_LOCAL_MEM_FENCE)    # OpenCL-idiom flags barrier

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernelCode

  # OpenCL idiom: get_global_id(0) must lower to the parenthesized
  # blockIdx/blockDim/threadIdx expression (never the bare rename).
  # The parens keep a trailing `* k` from binding to `threadIdx.x` only.
  doAssert "(blockIdx.x*blockDim.x+threadIdx.x)" in kernelCode,
    "CUDA global-id expression missing:\n" & kernelCode
  # The global-id expression as a multiplicative operand: the emitted
  # text is the parenthesized sum, then the multiply. A trailing `* k`
  # binds to the whole sum, never to `threadIdx.x` alone.
  doAssert "((blockIdx.x*blockDim.x+threadIdx.x) * 2)" in kernelCode,
    "CUDA global-id multiply must keep the sum parenthesized:\n" & kernelCode
  # canonical flat local index -> parenthesized x-major linearization
  doAssert "(threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x)" in kernelCode,
    "CUDA flat-local linearization missing:\n" & kernelCode
  # WGSL idiom barrier alias -> CUDA barrier
  doAssert "__syncthreads()" in kernelCode,
    "CUDA barrier missing:\n" & kernelCode
  echo "  OK — cross-vocabulary CUDA source inspect (compile + grep)"

  # Both foreign barrier spellings must lower to the canonical CUDA barrier.
  doAssert barrierKernelCode.count("__syncthreads()") == 2,
    "CUDA barrier missing or miscounted:\n" & barrierKernelCode
  echo "  OK — foreign barrier spellings lower to __syncthreads() (compile + grep)"

when isMainModule:
  runTest()
