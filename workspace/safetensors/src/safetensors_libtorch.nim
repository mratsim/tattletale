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

proc getTensor*(st: var Safetensor, tensorName: string): TorchTensor =
  ## Get a memory view to the tensor data.
  ## Returns a `MemSlice` that allows zero-copy access to the tensor data.
  ##
  ## Memory safety:
  ##   The returned `MemSlice` is derived from `st.memFile`,
  ##   the view MUST NOT outlive the underlying memory mapping.
  ##   Currently this is not enforced by the compiler but is an area of research:
  ##   - https://github.com/nim-lang/nimony/issues/1517#issuecomment-3859350630
  ##   - https://nim-lang.org/docs/manual.html#var-return-type-future-directions
  ## Lifetime:
  ##   The `MemSlice` is valid as long as `st` is valid, which is tied to
  ##   the original `MemFile` passed to `load`.
  let view = st.getMmapView(tensorName)
  let info = st.tensors[tensorName]
  view.data.from_blob(info.shape.asTorchView(), info.dtype.toTorchType())
