# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  ./safetensors,
  std/memfiles,
  std/tables,
  workspace/libtorch

# #######################################################################
#
#               Safetensors + libtorch syntactic sugar
#
# #######################################################################
#
# The API here might change with the following consideration
# - How to allow fast loading (async Streams, parallel workers, direct to GPU, ...)
# - How to associate lifetimes of `MemFile` and `MemSlice`
#
#   Unfortunately MemFile predates `lent` and `openarray` as values view `{.experimental: "views".}`
#   so we don't get compiler-enforced borrow-checking.
#   https://github.com/nim-lang/nimony/issues/1517#issuecomment-3859350630
#
#   And this is not available yet
#   https://nim-lang.org/docs/manual.html#var-return-type-future-directions
#   `proc foo(other: Y; container: var X): var T from container`
#
# The borrow check for `var T` return types:
#   https://nim-lang.org/docs/manual.html#procedures-var-return-type
#
# is not applicable here because we allocate a fresh address to store MemSlice
# instead of using the input Safetensor or one of its field.

proc toTorchType*(dtype: Dtype): ScalarKind {.inline.} =
  ## Convert safetensors dtype to libtorch ScalarKind.
  ## Raises ValueError if no direct mapping exists.

  case dtype:
    of BOOL: kBool
    of U8:   kUint8
    of I8:   kInt8
    of I16:  kInt16
    of F16:  kFloat16
    of BF16: kBfloat16
    of I32:  kInt32
    of F32:  kFloat32
    of C64:  kComplexF64
    of F64:  kFloat64
    of I64:  kInt64
    else:
      raise newException(ValueError, "No direct libtorch mapping for safetensors dtype: " & $dtype)

proc getTensorView*(st: var Safetensor, tensorName: string): TorchTensor =
  ## Get a memory view to the tensor data.
  ## Returns a `TorchTensor` that views the underlying memory-mapped data.
  ##
  ## Memory safety:
  ##   ⚠️ WARNING: The returned `TorchTensor` is a view into `st.memFile`.
  ##   The tensor MUST NOT outlive the underlying memory mapping.
  ##   If the `MemFile` is closed, accessing this tensor will cause undefined behavior / crash.
  ##
  ## For safe tensor loading, use `getTensorOwned` instead.
  ## This is intended to:
  ## - build high-performance loading primitives for example, direct-to-GPU weight loading.
  ## - memory-mapped inference on CPU
  ##
  ## Lifetime:
  ##   The tensor is valid as long as `st` is valid, which is tied to
  ##   the original `MemFile` passed to `load`.
  let view = st.getMmapView(tensorName)
  let info = st.tensors[tensorName]
  view.data.from_blob(info.shape.asTorchView(), info.dtype.toTorchType())

proc getTensorOwned*(st: var Safetensor, tensorName: string, device = kCPU): TorchTensor =
  ## Get an owned copy of the tensor data.
  ## Returns a `TorchTensor` that owns its data, safe to use after closing the `MemFile`.
  ##
  ## This is the recommended way to load tensors for inference.
  ## The tensor is cloned to `device` memory (default CPU).
  ##
  ## Args:
  ##   st: A loaded Safetensor (must remain valid during the copy)
  ##   tensorName: Name of the tensor to load
  ##
  ## Returns:
  ##   An owned `TorchTensor` on CPU.
  st.getTensorView(tensorName).to(device, copy=true) # Force copy

proc getTensorOwned*(st: var Safetensor, tensorName: string, device: Device): TorchTensor =
  ## Get an owned copy of the tensor data on the specified device.
  ## Returns a `TorchTensor` that owns its data, safe to use after closing the `MemFile`.
  ##
  ## This is the recommended way to load tensors for inference.
  ## The tensor is copied to the specified device.
  ##
  ## Args:
  ##   st: A loaded Safetensor (must remain valid during the copy)
  ##   tensorName: Name of the tensor to load
  ##   device: Target device (e.g., kCUDA, kCPU)
  ##
  ## Returns:
  ##   An owned `TorchTensor` on the specified device.
  st.getTensorView(tensorName).to(device, copy=true) # Force copy
