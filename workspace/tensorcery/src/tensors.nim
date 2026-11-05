# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Public Tensor face over the erased AnyTensorImpl implementation.
##
## Tensor is a thin value: context, erased implementation, runtime dtype, runtime device.
## Creation copies host data into backend storage, reads dispatch through the erased vtable.

import
  pkg/iface,
  workspace/data_structures/src/small_seqs,
  workspace/tensorcery/src/raw/dtypes,
  workspace/tensorcery/src/raw/impl,
  workspace/tensorcery/src/ctx

type Tensor* = object
  ## Tensor handle: the erased implementation plus its runtime dtype and device.
  ctx: Ctx
  impl: AnyTensorImpl
  dtype: Dtype
  device: DeviceKind

proc toTensor*[T](
    ctx: Ctx, data: openArray[T], Hw: static BackendKind, shape: varargs[int]
): Tensor =
  ## Creates a Tensor over `shape` holding a copy of `data`, contiguous, last dimension varying fastest, on the backend `Hw`.
  ##
  ## Raises AssertionDefect when `shape` spans more than 4 dimensions or when `data` does not hold exactly the element count of `shape`.
  let typedImpl = toTensorImpl[T, Hw](data, shape)
  Tensor(
    ctx: ctx,
    impl: typedImpl.to(AnyTensorImpl),
    dtype: toDtype(T),
    device: deviceKind(Hw),
  )

proc toTensor*[T](ctx: Ctx, data: openArray[T], shape: varargs[int]): Tensor =
  ## CPU form of `toTensor`: the backend defaults to CPU.
  ##
  ## Raises AssertionDefect under the same conditions as the backend form.
  toTensor[T](ctx, data, bkCpu, shape)

proc getNumel*(t: Tensor): int =
  ## Returns the element count, the product of the shape sizes, read through the erased implementation.
  t.impl.getNumel()

proc getShape*(t: Tensor): SmallSeq[4, int32] =
  ## Returns the shape, one size per dimension, read through the erased implementation.
  t.impl.getShape()

proc getStrides*(t: Tensor): SmallSeq[4, int32] =
  ## Returns the strides, one step per dimension in elements, read through the erased implementation.
  t.impl.getStrides()

proc getDtype*(t: Tensor): Dtype =
  ## Returns the runtime dtype, fixed at creation from the storage element type.
  t.dtype

proc getDevice*(t: Tensor): DeviceKind =
  ## Returns the runtime device, fixed at creation from the backend kind.
  t.device
