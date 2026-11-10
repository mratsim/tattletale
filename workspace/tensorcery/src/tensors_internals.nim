# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

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

template getLayoutType(rank: static int): typedesc =
  when rank == 1:
    (int,)
  elif rank == 2:
    (int, int)
  elif rank == 3:
    (int, int, int)
  elif rank == 4:
    (int, int, int, int)
  elif rank == 5:
    (int, int, int, int, int)
  else:
    {.error: "getLayoutType: rank " & $rank & " unsupported".}

template getTensorImplType*(engine: HwEngine, T: typedesc, rank: static int): typedesc =
  TensorImpl[engine.getMemHandleType(T), Layout[getLayoutType(rank), getLayoutType(rank)]]

template getTensorImplType*(engine: HwEngine, T: typedesc, layout: Layout): typedesc =
  TensorImpl[engine.getMemHandleType(T), typeof(layout)]

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - AnyTensorImpl interface
# ═══════════════════════════════════════════════════════════════════════

func shape*[DeviceMem, L](t: TensorImpl[DeviceMem, L]): Span[int] {.inline.} =
  toSpan(t.layout.shape)

func strides*[DeviceMem, L](t: TensorImpl[DeviceMem, L]): Span[int] {.inline.} =
  toSpan(t.layout.stride)

func dtype*[H, T, L](t: TensorImpl[DeviceMem[H, T], L]): Dtype {.inline.} =
  toDtype(T)

func isMaterialized*[DeviceMem, L](t: TensorImpl[DeviceMem, L]): bool {.inline.} =
  not t.data.isNil

func size*[DeviceMem, L](t: TensorImpl[DeviceMem, L]): int {.inline.} =
  t.layout.size()

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - TensorNode interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*[DeviceMem, L](t: TensorImpl[DeviceMem, L]): AnyTensorImpl {.inline.} =
  t.to(AnyTensorImpl)

func opName*[DeviceMem, L](t: TensorImpl[DeviceMem, L]): string {.inline.} =
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

func newPlaceholderImpl[E: HwEngine, T](engine: E, _: typedesc[T], layout: Layout):
    engine.getTensorImplType(T, layout) {.inline.} =
  (engine.getTensorImplType(T, layout))(data: nil, offset: 0, layout: layout)

proc newTensorImpl[E: HwEngine, T](engine: E, _: typedesc[T], layout: Layout):
    engine.getTensorImplType(T, layout) {.inline.} =
  result = (engine.getTensorImplType(T, layout))()
  result.data.new()
  result.data[] = engine.allocDevice(T, layout.size())

# ═══════════════════════════════════════════════════════════════════════
#  Graph
# ═══════════════════════════════════════════════════════════════════════

proc materialize*[E: HwEngine](engine: E, impl: AnyTensorImpl, layout: Layout) =
  case impl.dtype()
  of kFp32:
    let t = impl.to(engine.getTensorImplType(float32, layout))
    t.data.new()
    t.data[] = engine.allocDevice(float32, t.layout.size())
  of kBf16:
    let t = impl.to(engine.getTensorImplType(bfloat16, layout))
    t.data.new()
    t.data[] = engine.allocDevice(bfloat16, t.layout.size())
  else:
    raiseAssert("materialize: dtype unsupported")

proc newTensorNode*[E: HwEngine, T](engine: E, _: typedesc[T], layout: Layout): TensorNode =
  engine.newTensorImpl(T, layout).to(TensorNode)

proc newPlaceholderNode*[E: HwEngine](engine: E, dtype: Dtype, shape: Span[int]): TensorNode =
  case dtype
  of kFp32:
    var sh: (int, int)
    fillShape(sh, shape)
    engine.newPlaceholderImpl(float32, make_layout(sh, rowMajorStrides(sh))).to(TensorNode)
  of kBf16:
    var sh: (int, int)
    fillShape(sh, shape)
    engine.newPlaceholderImpl(bfloat16, make_layout(sh, rowMajorStrides(sh))).to(TensorNode)
  else:
    raise ValueError.newException("newPlaceholderNode: dtype " & $dtype & " unsupported")

proc newTensorNode*[E: HwEngine](engine: E, dtype: Dtype, shape: Span[int]): TensorNode =
  case shape.len
  of 2:
    var sh: (int, int)
    fillShape(sh, shape)
    let layout = make_layout(sh, rowMajorStrides(sh))
    case dtype
    of kFp32:
      engine.newTensorNode(float32, layout)
    of kBf16:
      engine.newTensorNode(bfloat16, layout)
    else:
      raise ValueError.newException("newTensorNode: dtype " & $dtype & " unsupported")
  else:
    raise ValueError.newException("newTensorNode: rank " & $shape.len & " unsupported")
