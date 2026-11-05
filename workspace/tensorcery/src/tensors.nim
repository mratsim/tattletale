# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Public Tensor face over the erased AnyTensorImpl implementation.
##
## Tensor is a thin value: context, erased implementation, runtime dtype, runtime device.
## Creation copies host data into backend storage, registers the tensor for the backward walk and dispatches reads through the erased vtable.
##
## Compute ops refuse cross-context, cross-device, mixed-dtype and shape-mismatched operands naming the combo.
## Ops propagate the grad flag:
## an output requires grad when any input does.
## The record evaluates on eval() or on read, toSeq and to() eval the context first when the tensor is pending.

import std/math
import pkg/iface

import
  workspace/data_structures/src/small_seqs,
  workspace/tensorcery/src/raw/dtypes,
  workspace/tensorcery/src/raw/impl,
  workspace/tensorcery/src/ctx,
  workspace/tensorcery/src/tape,
  workspace/tensorcery/src/kernels/k_sgd_adam,
  workspace/tensorcery/src/backends/metal,
  workspace/tensorcery/src/backends/cpu

# iface resolves the vtable's mixins at the outermost instantiation scope,
# so the typed conformant procs must be visible to module users of the Tensor face.
# Without this re-export the mixin binds the erased signature and the conformance check fails.
export metal.add, metal.mul, metal.mulScalar
export metal.addNoGrad, metal.mulNoGrad, metal.mulScalarNoGrad, metal.onesNoGrad
export metal.gather, metal.layernorm, metal.gelu, metal.softmax
export metal.crossEntropy, metal.where, metal.sgdUpdate, metal.adamUpdate
export metal.softmaxBwdNoGrad, metal.geluBwdNoGrad
export metal.layernormBwdDxNoGrad, metal.layernormBwdWNoGrad, metal.layernormBwdBNoGrad
export metal.crossEntropyBwdNoGrad, metal.gatherBwdNoGrad, metal.sliceBwdNoGrad
export metal.readbackBytes
export cpu.add, cpu.mul, cpu.mulScalar
export cpu.addNoGrad, cpu.mulNoGrad, cpu.mulScalarNoGrad, cpu.onesNoGrad
export cpu.gather, cpu.layernorm, cpu.gelu, cpu.softmax
export cpu.crossEntropy, cpu.where, cpu.sgdUpdate, cpu.adamUpdate
export cpu.softmaxBwdNoGrad, cpu.geluBwdNoGrad
export cpu.layernormBwdDxNoGrad, cpu.layernormBwdWNoGrad, cpu.layernormBwdBNoGrad
export cpu.crossEntropyBwdNoGrad, cpu.gatherBwdNoGrad, cpu.sliceBwdNoGrad
export cpu.readbackBytes

# aliasStrided is an impl.nim proc: a metadata view over explicit shape and strides,
# used by the backward walk to build shape-op cotangents
export impl.aliasStrided

type Tensor* = object
  ## Tensor handle: the erased implementation plus its runtime dtype and device.
  ctx: Ctx
  impl: AnyTensorImpl
  dtype: Dtype
  device: DeviceKind

proc toTensorShaped[T](ctx: Ctx, data: openArray[T], Hw: static BackendKind, shape: openArray[int], requiresGrad: bool): Tensor =
  ## Creates a Tensor over `shape` holding a copy of `data`, contiguous, last dimension varying fastest, on the backend `Hw`.
  ##
  ## Raises AssertionDefect when `shape` spans more than 4 dimensions or when `data` does not hold exactly the element count of `shape`.
  when Hw == bkCpu:
    ensureCpu(ctx)
    let erasedImpl = toTensorImpl[T, Hw](ctx, data, shape, requiresGrad).toErased()
  elif Hw == bkMetal:
    ensureMetal(ctx)
    when T is float32 or T is int32:
      let erasedImpl = toMetalTensorImpl(ctx, data, shape, requiresGrad).toErased()
    else:
      {.error: "Metal construction supports float32 and int32 only, got " & $toDtype(T).}

  tape.register(ctx, erasedImpl.getTid(), erasedImpl)
  Tensor(
    ctx: ctx,
    impl: erasedImpl,
    dtype: toDtype(T),
    device: toDeviceKind(Hw),
  )

proc toTensor*[T](
    ctx: Ctx, data: openArray[T], Hw: static BackendKind, shape: varargs[int]
): Tensor =
  ## Creates a Tensor over `shape` holding a copy of `data`, contiguous, last dimension varying fastest, on the backend `Hw`.
  ##
  ## Raises AssertionDefect under the same conditions as the openArray shape form.
  toTensorShaped(ctx, data, Hw, shape, false)

proc toTensor*[T](ctx: Ctx, data: openArray[T], shape: varargs[int]): Tensor =
  ## CPU form of `toTensor`: the backend defaults to CPU.
  ##
  ## Raises AssertionDefect under the same conditions as the backend form.
  toTensorShaped[T](ctx, data, bkCpu, shape, false)

proc toGradTensor*[T](
    ctx: Ctx, data: openArray[T], Hw: static BackendKind, shape: varargs[int]
): Tensor =
  ## Creates a Tensor that requires grad: backward() accumulates its gradient.
  ## Otherwise the backend form of `toTensor`.
  ##
  ## Raises AssertionDefect under the same conditions as `toTensor`.
  toTensorShaped(ctx, data, Hw, shape, true)

proc toGradTensor*[T](ctx: Ctx, data: openArray[T], shape: varargs[int]): Tensor =
  ## CPU form of `toGradTensor`: the backend defaults to CPU.
  ##
  ## Raises AssertionDefect under the same conditions as `toTensor`.
  toTensorShaped[T](ctx, data, bkCpu, shape, true)

proc toTensor*(ctx: Ctx, impl: AnyTensorImpl, dtype: Dtype, device: DeviceKind): Tensor =
  ## Wraps an already-created erased implementation into the Tensor face.
  ## autograd.nim uses it to hand out gradients recorded on the tape.
  Tensor(ctx: ctx, impl: impl, dtype: dtype, device: device)

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

proc getCtx*(t: Tensor): Ctx =
  ## Returns the recording context of the tensor.
  t.ctx

proc getImpl*(t: Tensor): AnyTensorImpl =
  ## Returns the erased implementation behind the tensor.
  t.impl

proc getRequiresGrad*(t: Tensor): bool =
  ## Returns the grad flag: true when backward() accumulates a gradient for the tensor.
  t.impl.getRequiresGrad()

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
  tape.register(a.ctx, outImpl.getTid(), outImpl)
  Tensor(ctx: a.ctx, impl: outImpl, dtype: a.dtype, device: a.device)

proc `*`*(a, b: Tensor): Tensor =
  ## Elementwise product of same-shape tensors, recorded on the context.
  ## Refusals match `+`: cross-context, cross-device, mixed dtypes, shape mismatches.
  doAssert a.ctx == b.ctx, "mul: operands from different contexts refused"
  doAssert a.device == b.device,
      "mul: cross-device operands " & $a.device & " x " & $b.device & " refused, copy to one device first"
  doAssert a.dtype == b.dtype,
      "mul: mixed dtypes " & $a.dtype & " x " & $b.dtype & " refused"
  doAssert sameShape(a.getShape(), b.getShape()),
      "mul: shape mismatch " & $a.getShape() & " vs " & $b.getShape() &
      " refused, no broadcasting"

  let outImpl = a.impl.mul(b.impl)
  tape.register(a.ctx, outImpl.getTid(), outImpl)
  Tensor(ctx: a.ctx, impl: outImpl, dtype: a.dtype, device: a.device)

proc `*`*(a: Tensor, scalar: float64): Tensor =
  ## Elementwise multiply by a scalar, recorded on the context.
  ## The scalar narrows to the tensor's dtype at enqueue.
  let outImpl = a.impl.mulScalar(scalar)
  tape.register(a.ctx, outImpl.getTid(), outImpl)
  Tensor(ctx: a.ctx, impl: outImpl, dtype: a.dtype, device: a.device)

proc toSeq*[T](t: Tensor): seq[T] =
  ## Values in logical order, last axis fastest, views gathered through their strides.
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
  ##
  ## A copy is a materialization, not a record op, so gradient does not flow through it:
  ## the copy never requires grad.
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
  toTensorShaped(t.ctx, host, Hw, shape.toOpenArray(0, int(sh.len) - 1), false)

# Shape ops: pure metadata, never enqueue, never touch submission counts.
# ------------------------------------------------------------

proc toShapeSeq(dims: openArray[int]): SmallSeq[4, int32] =
  ## Rank-checked dims as a SmallSeq.
  ##
  ## Raises AssertionDefect when the rank leaves 1 to 4.
  doAssert dims.len in 1 .. 4,
      "rank " & $dims.len & " refused, tensorcery supports rank 1 to 4"
  var extents: array[4, int32]
  for i in 0 ..< dims.len:
    extents[i] = int32(dims[i])
  result = SmallSeq[4, int32].initFrom(extents.toOpenArray(0, dims.len - 1))

proc wrapView(t: Tensor, outImpl: AnyTensorImpl): Tensor =
  ## Face wrapper over a metadata view: registers the new tensor id and keeps the parent's dtype and device.
  tape.register(t.ctx, outImpl.getTid(), outImpl)
  Tensor(ctx: t.ctx, impl: outImpl, dtype: t.dtype, device: t.device)

proc reshape*(t: Tensor, shape: varargs[int]): Tensor =
  ## Reshape as a pure-metadata view over the same storage, never enqueues.
  ##
  ## Raises AssertionDefect when the element count changes or the rank leaves 1 to 4.
  wrapView(t, t.impl.reshape(toShapeSeq(shape)))

proc flatten*(t: Tensor): Tensor =
  ## Collapses to one axis over the same storage, a reshape to the element count.
  wrapView(t, t.impl.reshape(toShapeSeq([t.getNumel()])))

proc permute*(t: Tensor, perm: varargs[int]): Tensor =
  ## Dimension permutation as a pure-metadata view: output axis i reads source axis perm[i].
  ##
  ## Raises AssertionDefect when `perm` is not a permutation of the axes.
  wrapView(t, t.impl.permuteDims(toShapeSeq(perm)))

proc transpose*(t: Tensor): Tensor =
  ## Rank-2 transpose as a pure-metadata view.
  ##
  ## Raises AssertionDefect when the tensor is not rank 2.
  doAssert int(t.getShape().len) == 2,
      "transpose: rank " & $t.getShape().len & " refused, rank 2 only"
  wrapView(t, t.impl.permuteDims(toShapeSeq([1, 0])))

proc slice*(t: Tensor, sel: varargs[Slice[int]]): Tensor =
  ## Range selection as a pure-metadata view: axis i keeps `sel[i]`.
  ## Unselected trailing axes keep every element.
  ##
  ## Raises AssertionDefect when a selection is empty or reaches beyond the axis end.
  let rank = int(t.getShape().len)
  doAssert sel.len <= rank,
      "slice: " & $sel.len & " selections on a rank " & $rank & " tensor refused"
  var start, count: array[4, int32]
  for i in 0 ..< rank:
    if i < sel.len:
      let lo = sel[i].a
      let hi = sel[i].b
      start[i] = int32(lo)
      count[i] = int32(hi - lo + 1)
    else:
      start[i] = 0
      count[i] = t.getShape()[i]
  wrapView(t, t.impl.sliceRange(
    SmallSeq[4, int32].initFrom(start.toOpenArray(0, rank - 1)),
    SmallSeq[4, int32].initFrom(count.toOpenArray(0, rank - 1))))


# Microgpt op set
# ------------------------------------------------------------

proc checkSameCtxDevice(a, b: Tensor, op: string) =
  ## Refuses cross-context and cross-device operands, naming the combo.
  doAssert a.ctx == b.ctx, op & ": operands from different contexts refused"
  doAssert a.device == b.device,
      op & ": cross-device operands " & $a.device & " x " & $b.device &
      " refused, copy to one device first"

proc checkFloat32(t: Tensor, op: string) =
  ## Refuses non-float32 tensors on the differentiable op set.
  doAssert t.dtype == dtFloat32, op & ": " & $t.dtype & " refused, float32 only"

proc wrapOp(t: Tensor, outImpl: AnyTensorImpl): Tensor =
  ## Face wrapper over a computed output: registers the tensor id and the producing op's dtype and device.
  tape.register(t.ctx, outImpl.getTid(), outImpl)
  Tensor(ctx: t.ctx, impl: outImpl, dtype: t.dtype, device: t.device)

proc gather*(w: Tensor, indices: Tensor): Tensor =
  ## Embedding lookup: the embedding vector at index `indices[i]` in the rank-2 table
  ## `w` for each index, output shape indices.shape + [embedding dim], recorded on the context.
  ##
  ## Raises AssertionDefect for a float32 table with non-int32 indices or any
  ## cross-context, cross-device, mixed-dtype combo.
  checkSameCtxDevice(w, indices, "gather")
  checkFloat32(w, "gather table")
  doAssert indices.dtype == dtInt32,
      "gather: int32 indices refused, got " & $indices.dtype
  wrapOp(w, w.impl.gather(indices.impl))

proc layernorm*(x, w, b: Tensor, eps: float64): Tensor =
  ## Last-axis layernorm over `x` with the learned scale `w` and shift `b`,
  ## recorded on the context.
  checkSameCtxDevice(x, w, "layernorm")
  checkSameCtxDevice(x, b, "layernorm")
  checkFloat32(x, "layernorm input")
  checkFloat32(w, "layernorm scale")
  checkFloat32(b, "layernorm shift")
  wrapOp(x, x.impl.layernorm(w.impl, b.impl, eps))

proc gelu*(x: Tensor): Tensor =
  ## Tanh-approximate gelu over `x`, recorded on the context.
  checkFloat32(x, "gelu")
  wrapOp(x, x.impl.gelu())

proc softmax*(x: Tensor): Tensor =
  ## Last-axis softmax over `x`, recorded on the context.
  checkFloat32(x, "softmax")
  wrapOp(x, x.impl.softmax())

proc crossEntropy*(logits, targets: Tensor): Tensor =
  ## Cross entropy over the last axis of `logits`, one loss element per logit set,
  ## `targets` names the class per set, recorded on the context.
  checkSameCtxDevice(logits, targets, "cross entropy")
  checkFloat32(logits, "cross entropy logits")
  doAssert targets.dtype == dtInt32,
      "cross entropy: int32 targets refused, got " & $targets.dtype
  wrapOp(logits, logits.impl.crossEntropy(targets.impl))

proc where*(cond, a, b: Tensor): Tensor =
  ## Elementwise select: `a` where `cond` is nonzero, `b` elsewhere, recorded on the context.
  ## Same shape only, no broadcasting.
  checkSameCtxDevice(cond, a, "where")
  checkSameCtxDevice(cond, b, "where")
  checkFloat32(cond, "where condition")
  checkFloat32(a, "where true branch")
  checkFloat32(b, "where false branch")
  doAssert sameShape(cond.getShape(), a.getShape()) and
      sameShape(cond.getShape(), b.getShape()),
      "where: shape mismatch " & $cond.getShape() & " vs " & $a.getShape() &
      " vs " & $b.getShape() & " refused, no broadcasting"
  wrapOp(a, a.impl.where(cond.impl, b.impl))

proc sgdUpdate*(p, g: Tensor, lr: float64): void =
  ## In-place sgd parameter update: `p` moves by `-lr * g`, recorded on the context.
  ## Reading the parameters materializes the update first (the MLX rule).
  checkSameCtxDevice(p, g, "sgd update")
  checkFloat32(p, "sgd update")
  checkFloat32(g, "sgd update gradient")
  p.impl.sgdUpdate(g.impl, lr)

proc adamUpdate*(p, g, m, v: Tensor, lr, beta1, beta2, eps: float64, step: int): void =
  ## In-place adam parameter update over `p` with gradient `g` and the persistent
  ## moments `m` and `v`, recorded on the context. The bias corrections
  ## bc1 = 1/(1 - beta1^step) and bc2 = 1/(1 - beta2^step) fold in at enqueue.
  ## Reading the parameters materializes the update first (the MLX rule).
  checkSameCtxDevice(p, g, "adam update")
  checkSameCtxDevice(p, m, "adam update")
  checkSameCtxDevice(p, v, "adam update")
  checkFloat32(p, "adam update")
  checkFloat32(g, "adam update gradient")
  checkFloat32(m, "adam update first moment")
  checkFloat32(v, "adam update second moment")
  doAssert step >= 1, "adam update: step must count from 1, got " & $step
  let bc1 = 1.0 / (1.0 - pow(beta1, float64(step)))
  let bc2 = 1.0 / (1.0 - pow(beta2, float64(step)))
  p.impl.adamUpdate(g.impl, m.impl, v.impl, AdamScalars(
    lr: float32(lr), beta1: float32(beta1), beta2: float32(beta2), eps: float32(eps),
    bc1: float32(bc1), bc2: float32(bc2)))
