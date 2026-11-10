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
import tensors_internal_types

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

template getTensorImplType*(engine: HwEngine, T: typedesc, rank: static int): typedesc =
  TensorImpl[engine.getMemHandleType(T), rank]

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - AnyTensorImpl interface
# ═══════════════════════════════════════════════════════════════════════

func shape*[DeviceMem; Rank: static int](t: TensorImpl[DeviceMem, Rank]): Span[int] {.inline.} =
  Span[int](p: toOpenArray(t.shape, 0, Rank - 1))

func strides*[DeviceMem; Rank: static int](t: TensorImpl[DeviceMem, Rank]): Span[int] {.inline.} =
  Span[int](p: toOpenArray(t.stride, 0, Rank - 1))

func dtype*[H, T; Rank: static int](t: TensorImpl[DeviceMem[H, T], Rank]): Dtype {.inline.} =
  toDtype(T)

func isMaterialized*(t: TensorImpl): bool {.inline.} =
  not t.data.isNil

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
  ""

proc isMaterialized*(n: TensorNode): bool {.inline.} =
  n.getImpl().isMaterialized()

# ═══════════════════════════════════════════════════════════════════════
#  Host<->Device copies
# ═══════════════════════════════════════════════════════════════════════

proc upload*[E: HwEngine](engine: E, node: TensorNode, src: openArray[float32]) {.inline.} =
  let impl = node.getImpl()
  let t = impl.to(engine.getTensorImplType(float32, 2))
  engine.copyToDeviceFromHost(t.data[], src)

proc upload*[E: HwEngine](engine: E, node: TensorNode, src: openArray[bfloat16]) {.inline.} =
  let t = node.getImpl().to(engine.getTensorImplType(bfloat16, 2))
  engine.copyToDeviceFromHost(t.data[], src)

proc download*[E: HwEngine](engine: E, dst: var openArray[float32], node: TensorNode) {.inline.} =
  engine.download(dst, node, 0)

proc download*[E: HwEngine](engine: E, dst: var openArray[bfloat16], node: TensorNode) {.inline.} =
  engine.download(dst, node, 0)

proc download*[E: HwEngine](engine: E, dst: var openArray[float32], node: TensorNode, offsetElems: int) {.inline.} =
  let t = node.getImpl().to(engine.getTensorImplType(float32, 2))
  engine.copyToHostFromDevice(dst, t.data[], t.offset + offsetElems)

proc download*[E: HwEngine](engine: E, dst: var openArray[bfloat16], node: TensorNode, offsetElems: int) {.inline.} =
  let t = node.getImpl().to(engine.getTensorImplType(bfloat16, 2))
  engine.copyToHostFromDevice(dst, t.data[], t.offset + offsetElems)

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

proc materialize*[E: HwEngine](engine: E, impl: TensorImpl) {.inline.} =
  engine.allocDataIn(impl)

proc newPlaceholderNode*[E: HwEngine](engine: E, dtype: Dtype, shape: openArray[int]): TensorNode =
  match(ElemType, dtype, Rank, shape.len):
    engine.newPlaceholderImpl(ElemType, Rank, shape).to(TensorNode)

proc newTensorNode*[E: HwEngine](engine: E, dtype: Dtype, shape: openArray[int]): TensorNode =
  match(ElemType, dtype, Rank, shape.len):
    engine.newTensorImpl(ElemType, Rank, shape).to(TensorNode)
