## NVRTC: cross-vocabulary coordinate inspect (compile + grep).
##
## Without a CUDA device, the CUDA printer is verified against generated
## source. The `cuda:` macro generates the kernel source at Nim compile
## time, and the test greps the emitted text for the expected native
## spellings (the inspect mechanism, no manual echo). The kernel is
## written in foreign idioms:
##   - OpenCL idiom `get_global_id(0)` -> `(blockIdx.x*blockDim.x+threadIdx.x)`
##   - `get_global_id(0) * 2` -> the parenthesized sum, then the multiply
##   - WGSL idiom `workgroupBarrier()` -> `__syncthreads()`
##   - canonical `thread_index_in_threadgroup` -> the parenthesized
##     x-major linearization
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_cross_vocabulary_inspect.nim

import std/strutils
import workspace/crucible

const kernelCode = cuda:
  proc crossVocabKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let gid = get_global_id(0'u32)
    let gid2 = get_global_id(0'u32) * 2'u32
    C[gid] = gid
    C[gid2] = gid2
    C[thread_index_in_threadgroup] = thread_index_in_threadgroup
    workgroupBarrier()

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
  doAssert "((blockIdx.x*blockDim.x+threadIdx.x) * 2U)" in kernelCode,
    "CUDA global-id multiply must keep the sum parenthesized:\n" & kernelCode
  # canonical flat local index -> parenthesized x-major linearization
  doAssert "(threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x)" in kernelCode,
    "CUDA flat-local linearization missing:\n" & kernelCode
  # WGSL idiom barrier alias -> CUDA barrier
  doAssert "__syncthreads()" in kernelCode,
    "CUDA barrier missing:\n" & kernelCode
  echo "  OK — cross-vocabulary CUDA source inspect (compile + grep)"

when isMainModule:
  runTest()
