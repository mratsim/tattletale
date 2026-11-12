# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import std/typetraits
import workspace/crucible
import workspace/ceramic
import workspace/math
import workspace/iface
import workspace/design_by_contract
import ./tensors_internal_types
import ./compute_graph/batching

export tensors_internal_types

# ############################################################
#
#                   Tensor internals
#
# ############################################################
#
#
# Design:
#
# Tensorcery is lazy/pull-based, evaluation is on-demand.
# A graph is built and only on explicit .eval() or conversions (string, writing to disk)
# is the compute scheduled.
#
# This allows batching compute all-at-once, reducing the CPU<->GPU synchronization overhead.
# Furthermore, when backprop is added, the graph can be reused for backward propagation.

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - AnyTensorImpl interface
# ═══════════════════════════════════════════════════════════════════════

func shape*[DeviceMem; Rank: static int](t: TensorImpl[DeviceMem, Rank]): Span[int] {.inline.} =
  Span[int](p: toOpenArray(t.shape, 0, Rank - 1))

func strides*[DeviceMem; Rank: static int](t: TensorImpl[DeviceMem, Rank]): Span[int] {.inline.} =
  Span[int](p: toOpenArray(t.stride, 0, Rank - 1))

func dtype*[H, T; Rank: static int](t: TensorImpl[DeviceMem[H, T], Rank]): Dtype {.inline.} =
  toDtype(T)

func offset*[DeviceMem; Rank: static int](t: TensorImpl[DeviceMem, Rank]): int {.inline.} =
  t.offset

func size*[DeviceMem; Rank: static int](t: TensorImpl[DeviceMem, Rank]): int {.inline.} =
  result = 1
  for i in 0 ..< Rank:
    result *= t.shape[i]

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - TensorNode interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*(t: TensorImpl): AnyTensorImpl {.inline.} =
  t.to(AnyTensorImpl)

func opName*(t: TensorImpl): string {.inline.} =
  ## Name in the computation graph
  "Tensor"

# ═══════════════════════════════════════════════════════════════════════
#  Host<->Device copies
# ═══════════════════════════════════════════════════════════════════════

template withDeviceImpl*[E: HwEngine](engine: E, impl: AnyTensorImpl, T: typedesc, concreteTensor, body: untyped) =
  ## Casts the type-erased impl tensor to its concrete TensorImpl[T, Rank] type
  ## and make it available as the symbol passed as "concreteTensor"
  ##
  ## Usage:
  ##   engine.withDeviceImpl(node.getImpl(), float32, t):
  ##     engine.copyToHostFromDevice(dst, t.data[], t.offset)
  case impl.shape().p.len
  of 1:
    let concreteTensor = impl.to(engine.getTensorImplType(T, 1))
    body
  of 2:
    let concreteTensor = impl.to(engine.getTensorImplType(T, 2))
    body
  of 3:
    let concreteTensor = impl.to(engine.getTensorImplType(T, 3))
    body
  of 4:
    let concreteTensor = impl.to(engine.getTensorImplType(T, 4))
    body
  of 5:
    let concreteTensor = impl.to(engine.getTensorImplType(T, 5))
    body
  else:
    raiseAssert("Tensorcery internal error: rank out of range")

proc download1*[E: HwEngine](engine: E, dst: pointer, node: TensorNode) =
  let impl = node.getImpl()
  match(ElemType, impl.dtype(), Rank, impl.shape().p.len):
    let ci = impl.to(engine.getTensorImplType(ElemType, Rank))
    engine.copyToHostFromDevice(cast[ptr ElemType](dst)[], ci.data[], ci.offset)

proc upload*[E: HwEngine](engine: E, node: TensorNode, src: openArray[float32]) =
  engine.withDeviceImpl(node.getImpl(), float32, t):
    engine.copyToDeviceFromHost(t.data[], src)

proc upload*[E: HwEngine](engine: E, node: TensorNode, src: openArray[float16]) =
  engine.withDeviceImpl(node.getImpl(), float16, t):
    engine.copyToDeviceFromHost(t.data[], src)

proc upload*[E: HwEngine](engine: E, node: TensorNode, src: openArray[bfloat16]) =
  engine.withDeviceImpl(node.getImpl(), bfloat16, t):
    engine.copyToDeviceFromHost(t.data[], src)

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl
# ═══════════════════════════════════════════════════════════════════════

func newPlaceholderImpl[E: HwEngine, T](
        engine: E,
        _: typedesc[T],
        rank: static int,
        shape: openArray[int]): engine.getTensorImplType(T, rank) {.inline.} =
  result = (engine.getTensorImplType(T, rank))()
  result.shape[0 ..< rank] = shape
  rowMajorStrides(result.stride, result.shape)

proc newTensorImpl[E: HwEngine, T](
      engine: E,
      _: typedesc[T],
      rank: static int,
      shape: openArray[int]): engine.getTensorImplType(T, rank) {.inline.} =
  result = (engine.getTensorImplType(T, rank))()
  result.shape[0 ..< rank] = shape
  rowMajorStrides(result.stride, result.shape)
  result.data.new()
  result.data[] = engine.allocDevice(T, result.size())

# ═══════════════════════════════════════════════════════════════════════
#  Graph
# ═══════════════════════════════════════════════════════════════════════

proc preAlloc*[E: HwEngine](engine: E, impl: TensorImpl) {.inline.} =
  ## Allocates the impl's device data at enqueue time, before the kernel enqueue.
  ##
  ## Contract:
  ## - every op handler calls this line: enqueue-time allocation is uniform
  engine.allocDataIn(impl)

proc newPlaceholderNode*[E: HwEngine](engine: E, dtype: Dtype, shape: openArray[int]): TensorNode =
  match(ElemType, dtype, Rank, shape.len):
    engine.newPlaceholderImpl(ElemType, Rank, shape).to(TensorNode)

proc newViewNode*[E: HwEngine](engine: E, src: TensorNode, shape, strides: openArray[int], offset: int, srcOpName: string): TensorNode =
  ## View over src's data: the view rank is independent of the source rank,
  ## shape/strides/offset describe the view, data binds to the source.
  let si = src.getImpl()
  match(ElemType, si.dtype(), Rank, shape.len):
    let r = engine.getTensorImplType(ElemType, Rank)()
    r.shape[0 ..< Rank] = shape
    r.stride[0 ..< Rank] = strides
    r.offset = offset
    engine.withDeviceImpl(si, ElemType, s):
      r.data = s.data
      if s.isPreAllocated():
        r.offset = s.offset + offset
    ViewNode(r: r.to(AnyTensorImpl), src: src, srcOpName: srcOpName).to(TensorNode)

proc newTensorNode*[E: HwEngine](engine: E, dtype: Dtype, shape: openArray[int]): TensorNode =
  match(ElemType, dtype, Rank, shape.len):
    engine.newTensorImpl(ElemType, Rank, shape).to(TensorNode)

proc enqueueView[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T], rank: static int): bool =
  let v = n.to(ViewNode)
  if v.isNil():
    return false
  engine.enqueueNode(batch, v.src)
  let r = v.r.to(engine.getTensorImplType(T, rank))
  engine.withDeviceImpl(v.src.getImpl(), T, s):
    r.data = s.data
    r.offset = r.offset + s.offset
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("View", enqueueView)
