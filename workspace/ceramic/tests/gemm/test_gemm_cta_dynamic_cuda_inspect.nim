## Unified GEMM entry: CUDA compile-gate (source inspect, no device).
##
## The unified manual_gemm_cta_dynamic entry expands its kernel for CUDA
## at Nim compile time. This test pins the generated CUDA text without a CUDA device
## (compiled and grepped, no hand-written expected text): the global-id
## expression must keep the `(int)(blockIdx.x*blockDim.x+threadIdx.x)`
## cast form (the CUDA byte-identity gate), and the gemm_cta body must
## carry `__syncthreads()`. Execution of the kernel is the Linux-box gate
## (manual_gemm_cta_dynamic runTest).
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/gemm --nimcache:nimcache/tests/gemm \
##     workspace/ceramic/tests/gemm/test_gemm_cta_dynamic_cuda_inspect.nim

import std/strutils
import workspace/ceramic/tests/gemm/manual_gemm_cta_dynamic

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # The flat global id must lower to the CUDA `(int)(...)` cast form.
  doAssert "(int)(blockIdx.x*blockDim.x+threadIdx.x)" in kernelCodeCuda,
    "CUDA global-id cast form missing:\n" & kernelCodeCuda
  # The 1D linearized dispatch: flat id mod 128 (thread), div 128 (block).
  doAssert "% 128" in kernelCodeCuda and "/ 128" in kernelCodeCuda,
    "CUDA flat-id decomposition missing:\n" & kernelCodeCuda
  # The dispatch tuple destructures into (tid, blk, gid, gridM, mCTA, nCTA):
  # the CTA decomposition (mCTA, nCTA) is extracted from the tuple.
  doAssert "int mCTA = tmpTuple" in kernelCodeCuda,
    "CUDA CTA decomposition missing:\n" & kernelCodeCuda
  # The gemm_cta barriers lower to __syncthreads().
  doAssert "__syncthreads()" in kernelCodeCuda,
    "CUDA barrier missing:\n" & kernelCodeCuda
  echo "  OK — unified GEMM CUDA source inspect (compile + grep)"

when isMainModule:
  runTest()
