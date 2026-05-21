# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Low-level CUDA runtime API bindings and device intrinsics for Positron.

import workspace/positron/experimental/platforms/cuda/cuda_datatypes

# ############################################################
#              Host-side CUDA Runtime API
# ############################################################

{.push noconv, importc, nodecl.}

proc cudaMalloc*(p: ptr pointer, size: csize_t): cudaError_t
proc cudaFree*(p: pointer): cudaError_t
proc cudaMemcpy*(dst: pointer, src: pointer, count: csize_t, kind: cudaMemcpyKind): cudaError_t
proc cudaMemset*(p: pointer, value: cint, count: csize_t): cudaError_t
proc cudaDeviceSynchronize*(): cudaError_t
proc cudaStreamSynchronize*(stream: cudaStream_t): cudaError_t
proc cudaGetLastError*(): cudaError_t
proc cudaLaunchKernel*(fn: pointer, gridDim: Dim3, blockDim: Dim3,
                       args: ptr pointer, sharedMemBytes: csize_t,
                       stream: cudaStream_t): cudaError_t
{.pop.}

proc cudaGetErrorStringX*(err: cudaError_t): ptr char {.importc: "cudaGetErrorString", noconv, nodecl.}
proc cudaGetErrorString*(err: cudaError_t): cstring =
  var s {.codegenDecl: "const $# $#".} = cudaGetErrorStringX(err)
  result = s

# ############################################################
#              Device-side Intrinsics
# ############################################################

proc syncthreads*() {.importcpp: "__syncthreads()", header: "cuda_runtime.h".}

proc shflXorSync*(mask: cuint, val: cfloat, offset: cint): cfloat {.
  importcpp: "__shfl_xor_sync(#, #, #)", header: "cuda_runtime.h".}

proc rsqrtf*(x: cfloat): cfloat {.importcpp: "rsqrtf(#)", header: "cuda_runtime.h".}
proc fmaf*(a, b, c: cfloat): cfloat {.importcpp: "fmaf(#, #, #)", header: "cuda_runtime.h".}
proc expf*(x: cfloat): cfloat {.importcpp: "expf(#)", header: "cuda_runtime.h".}

# Half conversions using __ushort_as_half / __half_as_ushort
proc half2float*(h: Half): cfloat {.
  importcpp: "__half2float(__ushort_as_half(#))",
  header: "cuda_fp16.h".}

proc float2halfRn*(f: cfloat): Half {.
  importcpp: "__half_as_ushort(__float2half_rn(#))",
  header: "cuda_fp16.h".}
