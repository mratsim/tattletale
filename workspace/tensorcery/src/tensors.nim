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
##
## Compute ops refuse cross-context, cross-device, mixed-dtype and shape-mismatched operands naming the combo.
## The record evaluates on eval() or on read, toSeq and to() eval the context first when the tensor is pending.

import pkg/iface

import
  workspace/data_structures/src/small_seqs,
  workspace/tensorcery/src/raw/dtypes,
  workspace/tensorcery/src/raw/impl,
  workspace/tensorcery/src/ctx,
  workspace/tensorcery/src/backends/metal

type Tensor* = object
  ## Tensor handle: the erased implementation plus its runtime dtype and device.
  ctx: Ctx
  impl: AnyTensorImpl
  dtype: Dtype
  device: DeviceKind

proc toTensorShaped[T](ctx: Ctx, data: openArray[T], Hw: static BackendKind, shape: openArray[int]): Tensor =
  ## Creates a Tensor over `shape` holding a copy of `data`, contiguous, last dimension varying fastest, on the backend `Hw`.
  ##
  ## Raises AssertionDefect when `shape` spans more than 4 dimensions or when `data` does not hold exactly the element count of `shape`.
  when Hw == bkCpu:
    let typedImpl = toTensorImpl[T, Hw](ctx, data, shape)
  elif Hw == bkMetal:
    ensureMetal(ctx)
    when T is float32:
      let typedImpl = toMetalTensorImpl(ctx, data, shape)
    else:
      doAssert false, "Metal construction supports float32, got " & $toDtype(T)

  Tensor(
    ctx: ctx,
    impl: typedImpl.to(AnyTensorImpl),
    dtype: toDtype(T),
    device: toDeviceKind(Hw),
  )

proc toTensor*[T](
    ctx: Ctx, data: openArray[T], Hw: static BackendKind, shape: varargs[int]
): Tensor =
  ## Creates a Tensor over `shape` holding a copy of `data`, contiguous, last dimension varying fastest, on the backend `Hw`.
  ##
  ## Raises AssertionDefect under the same conditions as the openArray shape form.
  toTensorShaped(ctx, data, Hw, shape)

proc toTensor*[T](ctx: Ctx, data: openArray[T], shape: varargs[int]): Tensor =
  ## CPU form of `toTensor`: the backend defaults to CPU.
  ##
  ## Raises AssertionDefect under the same conditions as the backend form.
  toTensorShaped[T](ctx, data, bkCpu, shape)

proc getNumel*(t: Tensor): int =
  ## Returns the element count as a product over all shape sizes.
  t.impl.getNumel()

proc getShape*(t: Tensor): SmallSeq[4, int32] =
  ## Returns the shape, one size per dimension.
  t.impl.getShape()

proc getStrides*(t: Tensor): SmallSeq[4, int32] =
  ## Returns the strides, one step per dimension in elements.
  t.impl.getStrides()

proc getDtype*(t: Tensor): Dtype =
  ## Returns the runtime dtype, fixed at creation from the storage element type.
  t.dtype

proc getDevice*(t: Tensor): DeviceKind =
  ## Returns the runtime device, fixed at creation from the backend kind.
  t.device

func sameShape(a, b: SmallSeq[4, int32]): bool =
  if a.len != b.len:
    return false
  for i in 0 ..< int(a.len):
    if a[i] != b[i]:
      return false
  result = true

proc `+`*(a, b: Tensor): Tensor =
  ## Elementwise add of same-shape tensors, recorded on the context.
  ##
  ## Refuses, naming the combo:
  ##
  ## - operands from different contexts
  ## - operands on different devices
  ## - mixed dtypes
  ## - shape mismatches, no broadcasting
  doAssert a.ctx == b.ctx, "add: operands from different contexts refused"
  doAssert a.device == b.device,
      "add: cross-device operands " & $a.device & " x " & $b.device & " refused, copy to one device first"
  doAssert a.dtype == b.dtype,
      "add: mixed dtypes " & $a.dtype & " x " & $b.dtype & " refused"
  doAssert sameShape(a.getShape(), b.getShape()),
      "add: shape mismatch " & $a.getShape() & " vs " & $b.getShape() &
      " refused, no broadcasting"

  let outImpl = a.impl.add(b.impl)
  Tensor(ctx: a.ctx, impl: outImpl, dtype: a.dtype, device: a.device)

proc `*`*(a: Tensor, scalar: float64): Tensor =
  ## Elementwise multiply by a scalar, recorded on the context.
  ## The scalar narrows to the tensor's dtype at enqueue.
  let outImpl = a.impl.mulScalar(scalar)
  Tensor(ctx: a.ctx, impl: outImpl, dtype: a.dtype, device: a.device)

proc toSeq*[T](t: Tensor): seq[T] =
  ## Values in storage order, last axis fastest.
  ##
  ## Reading materializes: a pending tensor evals the context first (the MLX rule).
  ## Raises AssertionDefect when T does not match the tensor's dtype.
  doAssert toDtype(T) == t.dtype,
      "toSeq[" & $toDtype(T) & "] on a " & $t.dtype & " tensor refused"

  let raw = t.impl.readbackBytes()
  result = newSeq[T](t.getNumel())
  if result.len > 0:
    copyMem(result[0].addr, raw[0].unsafeAddr, raw.len)

proc to*(t: Tensor, Hw: static BackendKind): Tensor =
  ## Copies the tensor to backend `Hw`, materializing pending ops first.
  ## Same-device input returns unchanged.
  ## Raises AssertionDefect for non-float32 tensors. This increment moves float32 only.
  if toDeviceKind(Hw) == t.device:
    return t

  doAssert t.dtype == dtFloat32, "to(" & $Hw & "): float32 only, got " & $t.dtype

  let raw = t.impl.readbackBytes()
  var host = newSeq[float32](t.getNumel())
  copyMem(host[0].addr, raw[0].unsafeAddr, raw.len)

  let sh = t.getShape()
  var shape: array[4, int]
  for i in 0 ..< int(sh.len):
    shape[i] = int(sh[i])
  toTensorShaped(t.ctx, host, Hw, shape.toOpenArray(0, int(sh.len) - 1))
