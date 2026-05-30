# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Syntactic sugar and utility helpers for Positron CUDA kernels.

import workspace/positron/experimental/nimcuda/platforms/cuda/cuda_datatypes

# ─── Pointer type aliases ─────────────────────────────────────────

type
  pHalf* = ptr UncheckedArray[Half]
    ## Pointer to fp16 buffer on GPU.
  pFloat* = ptr UncheckedArray[cfloat]
    ## Pointer to fp32 buffer on GPU.
  pFloat4* = ptr UncheckedArray[Float4]
    ## Pointer to float4 buffer on GPU.

# ─── Constructors ─────────────────────────────────────────────────

proc dim3*(x: SomeInteger = 1, y: SomeInteger = 1, z: SomeInteger = 1): Dim3 {.noInit, inline.} =
  ## Constructor matching CUDA dim3(x, y, z).
  Dim3(x: uint32 x, y: uint32 y, z: uint32 z)

# ─── Pointer arithmetic ──────────────────────────────────────────

template `+%`*[P: ptr Half or ptr UncheckedArray[Half]](p: P, offset: SomeInteger): P =
  ## pointer arithmetic (raw cast, no element scaling).
  ##
  ## Unfortunately sizeof(T) doesn't work on imported type (size is 0)
  ## Even if we specify a pragma {.size: 2.} pragma on the type
  cast[P](cast[uint](p) + uint(offset)*2)

template `+%`*[P: ptr Half2 or ptr UncheckedArray[Half2]](p: P, offset: SomeInteger): P =
  ## pointer arithmetic (raw cast, no element scaling).
  ##
  ## Unfortunately sizeof(T) doesn't work on imported type (size is 0)
  ## Even if we specify a pragma {.size: 4.} pragma on the type
  cast[P](cast[uint](p) + uint(offset)*4)

# ─── Error handling ───────────────────────────────────────────────

template check*(status: cudaError_t) =
  let code = status
  if code != cudaError_t(0):
    let errMsg = "CUDA error: " & $cudaGetErrorString(code) &
                " (" & $ord(code) & ") at " & $instantiationInfo()
    raise newException(Exception, errMsg)

# ─── Override Nim magic += with C++ += (nvcc device compat) ───────

func `+=`*(x: var float32, y: float32) {.importcpp: "# += #", nodecl.}
  # no gneric so it has overload priority