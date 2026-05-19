# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.


## CUDA type definitions for Positron kernels.
##
## These map to CUDA C/C++ types from `cuda_runtime.h` and `cuda_fp16.h`.
## All types here are used in both host and device code when compiled with `--cc:nvcc`.

# ─── Scalar type aliases ────────────────────────────────────────────

type
  cudaError_t* {.importc: "cudaError_t", header: "cuda_runtime.h", noInit.} = cint
  cudaStream_t* {.importc: "cudaStream_t", header: "cuda_runtime.h", noInit.} = pointer

# ─── dim3 (grid/block dimensions) ───────────────────────────────────

type
  Dim3* {.importc: "dim3", header: "cuda_runtime.h", bycopy, noInit.} = object
    x* {.importc: "x".}: uint32
    y* {.importc: "y".}: uint32
    z* {.importc: "z".}: uint32

# ─── CUDA built-in variables (device-side) ──────────────────────────
# These are per-thread built-in variables available in __global__ / __device__ code.
# `importc` without `header` tells Nim the symbol exists in the C/C++ namespace.
# nvcc recognises them as intrinsics inside device code.

let
  threadIdx* {.importc, inject, nodecl.}: Dim3
  blockIdx*  {.importc, inject, nodecl.}: Dim3
  blockDim*  {.importc, inject, nodecl.}: Dim3
  gridDim*   {.importc, inject, nodecl.}: Dim3

# ─── float4 (CUDA vector type) ──────────────────────────────────────

type
  Float4* {.importc: "float4", header: "cuda_runtime.h", bycopy, noInit.} = object
    x* {.importc: "x".}: cfloat
    y* {.importc: "y".}: cfloat
    z* {.importc: "z".}: cfloat
    w* {.importc: "w".}: cfloat

# ─── half precision types ───────────────────────────────────────────

type
  Half* {.importc: "__half", size: sizeof(uint16), header: "cuda_fp16.h", bycopy, noInit.} = object
    ## 16-bit fp16 storage type — raw bit representation.
    ## Convert to/from cfloat via `half2float` / `float2halfRn`.

  Half2* {.importc: "__half2", size: sizeof(uint32), header: "cuda_fp16.h", bycopy, noInit.} = object
    ## Packed pair of fp16 values (32-bit).
    ## x,y are individual `__half` values packed together.

# ─── cudaMemcpyKind ─────────────────────────────────────────────────

type
  cudaMemcpyKind* {.size: sizeof(cint), header: "cuda_runtime.h",
                    importc: "cudaMemcpyKind", noInit.} = enum
    cudaMemcpyHostToHost = 0
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    cudaMemcpyDeviceToDevice = 3
    cudaMemcpyDefault = 4
