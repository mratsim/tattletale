# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/memfiles,
  std/tables,
  std/importutils,
  workspace/tensorcery

import ./safetensors {.all.}
import ./collections {.all.}
import workspace/tensorcery/src/tape
import workspace/tensorcery/src/backends/cpu
privateAccess(SafetensorObj)
privateAccess(SafetensorsCollectionObj)

# #######################################################################
#
#               Safetensors + tensorcery syntactic sugar
#
# #######################################################################
#
# Loading safetensors data into tensorcery tensors.
# Counterpart of safetensors_libtorch.nim for the Tensor face.
#
# API shape:
# - every loader takes the Ctx as its first argument
#   (a tensorcery tensor lives in a context: the record and tape owner)
# - tensorcery constructs tensors by copying host data
#   (`toTensorImpl` allocates owned SharedStorage)
#
# `getTensorView` has no tensorcery form: a from_blob constructor
# is missing on the tensorcery side (borrow-view work class).
# Until then `getTensorOwned` is the only loading path.
# Mapping bytes are copied once into tensorcery storage.

proc toTensorceryType*(dtype: ST_dtype): Dtype {.inline.} =
  ## Convert safetensors dtype to tensorcery Dtype.
  ## Asserts naming the dtype when no direct mapping exists.

  case dtype:
    of BOOL: dtBool
    of U8:   dtUint8
    of I8:   dtInt8
    of I16:  dtInt16
    of F16:  dtFloat16
    of I32:  dtInt32
    of F32:  dtFloat32
    of F64:  dtFloat64
    of I64:  dtInt64
    else:
      doAssert false, "no direct tensorcery mapping for safetensors dtype: " & $dtype
      dtBool # unreachable, doAssert above aborts

proc hasTensor*(st: Safetensor, tensorName: string): bool {.inline.} =
  ## True when the `st` header lists `tensorName`.
  st.tensors.hasKey(tensorName)

proc hasTensor*(view: SafetensorsCollection, tensorName: string): bool {.inline.} =
  ## True when the collection weight map routes `tensorName` to a file.
  view.weightMap.hasKey(tensorName)

func product(a: seq[int]): int =
  result = 1
  for i in 0 ..< a.len:
    result *= a[i]

proc loadCpuImpl(ctx: Ctx, st: Safetensor, tensorName: string): Tensor =
  ## Copies the mapping bytes of `tensorName` into tensorcery CPU storage
  ## and wraps them in the erased Tensor face.
  ##
  ## - Copy semantics: the returned Tensor owns its data
  ## - Mapping lifetime: release the mapping after the call returns
  ## - Rank cap: 4 dimensions, enforced by the TensorImpl constructor
  ## - Unsupported dtypes assert naming the dtype: bool, F16, BF16, FP8
  let view = st.getMmapView(tensorName)
  let info = st.tensors[tensorName]
  let dt = toTensorceryType(info.dtype)

  template loadAs(T: typedesc) =
    let numel = info.shape.product()
    let byteSpan = info.dataOffsets.stopEx - info.dataOffsets.start
    doAssert byteSpan == numel * sizeof(T),
        "header spans " & $byteSpan & " bytes for " & $numel & " " & $T & " elements"
    let p = cast[ptr UncheckedArray[T]](view.data)
    let impl = toTensorImpl[T, bkCpu](ctx, toOpenArray(p, 0, numel - 1), info.shape, false)
    tape.register(ctx, impl.getTid(), impl.toErased())
    return toTensor(ctx, impl.toErased(), dt, devCpu)

  case info.dtype:
    of I8:   loadAs(int8)
    of I16:  loadAs(int16)
    of I32:  loadAs(int32)
    of I64:  loadAs(int64)
    of U8:   loadAs(uint8)
    of F32:  loadAs(float32)
    of F64:  loadAs(float64)
    else:
      doAssert false, "no tensorcery host type for safetensors dtype: " & $info.dtype

proc getTensorOwned*(ctx: Ctx, st: Safetensor, tensorName: string, Hw: static BackendKind): Tensor =
  ## Load `tensorName` as an owned tensorcery Tensor on backend `Hw`.
  ##
  ## Mapping bytes are copied once into CPU storage and transferred
  ## when `Hw` is not CPU. Device backends need a Metal or Vulkan Ctx.
  ##
  ## - ctx: the tensorcery context owning the record and the tape
  ## - st: a loaded Safetensor, alive during the copy
  when Hw == bkCpu:
    loadCpuImpl(ctx, st, tensorName)
  else:
    loadCpuImpl(ctx, st, tensorName).to(Hw)

proc getTensorOwned*(ctx: Ctx, st: Safetensor, tensorName: string): Tensor =
  ## CPU form of `getTensorOwned`, backend defaults to CPU.
  loadCpuImpl(ctx, st, tensorName)

proc getTensorOwned*(ctx: Ctx, view: SafetensorsCollection, tensorName: string, Hw: static BackendKind): Tensor =
  ## Load `tensorName` from a multi-file checkpoint as an owned Tensor
  ## on backend `Hw`. The weight map routes `tensorName` to its file,
  ## that file answers the header and serves the data.
  ##
  ## - ctx: the tensorcery context owning the record and the tape
  ## - view: a collection, alive for the duration of the copy
  when Hw == bkCpu:
    loadCpuImpl(ctx, view.files[view.fileOf(tensorName)], tensorName)
  else:
    loadCpuImpl(ctx, view.files[view.fileOf(tensorName)], tensorName).to(Hw)

proc getTensorOwned*(ctx: Ctx, view: SafetensorsCollection, tensorName: string): Tensor =
  ## CPU form of the collection-level `getTensorOwned`, backend CPU.
  loadCpuImpl(ctx, view.files[view.fileOf(tensorName)], tensorName)
