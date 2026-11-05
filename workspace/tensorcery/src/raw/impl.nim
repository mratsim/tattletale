# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Typed tensor implementations behind the erased AnyTensorImpl face.
##
## TensorImpl[T, Hw] stores the producing op id with runtime shape and strides plus the backend storage for element type T.
## The erased face has no type parameters, calls dispatch through the generated vtable.
##
## A backend module implements each op for its TensorImpl specialization.
## The op packs its blob once at enqueue and appends one record entry.
##
## iface initializes each global vtable lazily on first use, single-threaded use only.
## Acceptable for this MVP, revisit when backends run on worker threads.

import pkg/iface

import
  workspace/data_structures/src/small_seqs,
  workspace/tensorcery/src/raw/dtypes,
  workspace/tensorcery/src/ctx

iface *AnyTensorImpl:
  proc getNumel(): int
  proc getShape(): SmallSeq[4, int32]
  proc getStrides(): SmallSeq[4, int32]
  proc add(other: AnyTensorImpl): AnyTensorImpl
  proc mulScalar(scalar: float64): AnyTensorImpl
  proc readbackBytes(): seq[byte]

type DeviceHandle* = distinct uint64
  ## Opaque 8-byte device-buffer handle, owned by the device backend.

type TensorImpl*[T; Hw: static BackendKind] = ref object
  ## Typed tensor state behind one entry of the erased face.
  ctx*: Ctx
  seqNo*: int     ## 0 when materialized at construction, else the op id of the record entry that produces this storage
  shape*: SmallSeq[4, int32]
  strides*: SmallSeq[4, int32]
  when Hw == bkCpu:
    storage*: seq[T]
  elif Hw == bkMetal:
    mem*: DeviceHandle

func numel*(shape: openArray[int]): int =
  ## Returns the element count as a product over all shape sizes.
  result = 1
  for dim in shape:
    result *= dim

proc contiguousStrides*(shape: SmallSeq[4, int32]): SmallSeq[4, int32] =
  ## Returns the contiguous strides of an already rank-checked shape, one step per dimension in elements.
  var stride = 1
  var strides: array[4, int32]
  for i in countdown(int(shape.len) - 1, 0):
    strides[i] = int32(stride)
    stride *= int(shape[i])
  result = SmallSeq[4, int32].initFrom(strides.toOpenArray(0, int(shape.len) - 1))

proc contiguousMeta*(shape: openArray[int]): tuple[shape, strides: SmallSeq[4, int32]] =
  ## Rank-checked contiguous layout, last dimension varying fastest.
  ##
  ## Raises AssertionDefect when `shape` spans more than 4 dimensions.
  doAssert shape.len in 1 .. 4,
      "rank " & $shape.len & " construction refused, tensorcery supports rank 1 to 4"

  var extents: array[4, int32]
  for i in 0 ..< shape.len:
    extents[i] = int32(shape[i])
  result.shape = SmallSeq[4, int32].initFrom(extents.toOpenArray(0, shape.len - 1))

  result.strides = contiguousStrides(result.shape)

proc toTensorImpl*[T; Hw: static BackendKind](
    ctx: Ctx, data: openArray[T], shape: openArray[int]
): TensorImpl[T, Hw] =
  ## Builds a TensorImpl from host data over `shape`, contiguous, last dimension varying fastest.
  ##
  ## Raises AssertionDefect when `shape` spans more than 4 dimensions or when `data` does not hold exactly the element count of `shape`.
  when Hw != bkCpu:
    {.error: "device storage is allocated by the device backend, construct device tensors through the Tensor face".}

  let meta = contiguousMeta(shape)
  let numel = numel(shape)

  doAssert data.len == numel,
      "data holds " & $data.len & " elements, shape " & $shape & " needs " & $numel

  result = TensorImpl[T, Hw](ctx: ctx)
  result.shape = meta.shape
  result.strides = meta.strides
  result.storage = @data

proc getNumel*[T; Hw: static BackendKind](s: TensorImpl[T, Hw]): int =
  ## Returns the element count as a product over all shape sizes.
  result = 1
  for i in 0 ..< int(s.shape.len):
    result *= int(s.shape[i])

proc getShape*[T; Hw: static BackendKind](s: TensorImpl[T, Hw]): SmallSeq[4, int32] =
  s.shape

proc getStrides*[T; Hw: static BackendKind](s: TensorImpl[T, Hw]): SmallSeq[4, int32] =
  s.strides

