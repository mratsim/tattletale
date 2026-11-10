# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import workspace/crucible/src/builtins/builtins_gpu_types
export builtins_gpu_types
import workspace/crucible/src/runtime/engines/engines_api
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

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - AnyTensorImpl interface
# ═══════════════════════════════════════════════════════════════════════

func shape*[Mem, L](t: TensorImpl[Mem, L]): Span[int] {.inline.} =
  toSpan(t.layout.shape)

func strides*[Mem, L](t: TensorImpl[Mem, L]): Span[int] {.inline.} =
  toSpan(t.layout.stride)

func dtype*[H, T, L](t: TensorImpl[DeviceMem[H, T], L]): Dtype {.inline.} =
  toDtype(T)

func isMaterialized*[Mem, L](t: TensorImpl[Mem, L]): bool {.inline.} =
  not t.data.isNil

func size*[Mem, L](t: TensorImpl[Mem, L]): int {.inline.} =
  t.layout.size()

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - TensorNode interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*[Mem, L](t: TensorImpl[Mem, L]): AnyTensorImpl {.inline.} =
  t.to(AnyTensorImpl)

func opName*[Mem, L](t: TensorImpl[Mem, L]): string {.inline.} =
  ""

proc isMaterialized*(n: TensorNode): bool {.inline.} =
  n.getImpl().isMaterialized()

# ═══════════════════════════════════════════════════════════════════════
#  Host<->Device copies
# ═══════════════════════════════════════════════════════════════════════

proc upload*[E: HwEngine](engine: E, node: TensorNode, src: openArray[float32]) {.inline.} =
  let impl = node.getImpl()
  type Mem = typeof(engine.allocDevice(float32, 0))
  let t = impl.to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
  engine.copyToDeviceFromHost(t.data[], src)

proc upload*[E: HwEngine](engine: E, node: TensorNode, src: openArray[bfloat16]) {.inline.} =
  type Mem = typeof(engine.allocDevice(bfloat16, 0))
  let t = node.getImpl().to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
  engine.copyToDeviceFromHost(t.data[], src)

proc download*[E: HwEngine](engine: E, dst: var openArray[float32], node: TensorNode) {.inline.} =
  engine.download(dst, node, 0)

proc download*[E: HwEngine](engine: E, dst: var openArray[bfloat16], node: TensorNode) {.inline.} =
  engine.download(dst, node, 0)

proc download*[E: HwEngine](engine: E, dst: var openArray[float32], node: TensorNode, offsetElems: int) {.inline.} =
  type Mem = typeof(engine.allocDevice(float32, 0))
  let t = node.getImpl().to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
  engine.copyToHostFromDevice(dst, t.data[], t.offset + offsetElems)

proc download*[E: HwEngine](engine: E, dst: var openArray[bfloat16], node: TensorNode, offsetElems: int) {.inline.} =
  type Mem = typeof(engine.allocDevice(bfloat16, 0))
  let t = node.getImpl().to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
  engine.copyToHostFromDevice(dst, t.data[], t.offset + offsetElems)

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl
# ═══════════════════════════════════════════════════════════════════════

func newPlaceholderImpl[E: HwEngine, Mem](engine: E, layout: Layout, _: typedesc[Mem]): TensorImpl[Mem, Layout] {.inline.} =
  TensorImpl[Mem, Layout](data: nil, offset: 0, layout: layout)

proc newTensorImpl[E: HwEngine, T, Mem](engine: E, _: typedesc[T], layout: Layout, _: typedesc[Mem]): TensorImpl[Mem, Layout] {.inline.} =
  result.data.new()
  result.data[] = engine.allocDevice(T, layout.size())
  result.layout = layout

# ═══════════════════════════════════════════════════════════════════════
#  Graph
# ═══════════════════════════════════════════════════════════════════════

proc materialize*[E: HwEngine](engine: E, impl: AnyTensorImpl, layout: Layout) =
  case impl.dtype()
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    let t = impl.to(TensorImpl[Mem, Layout])
    t.data.new()
    t.data[] = engine.allocDevice(float32, t.layout.size())
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    let t = impl.to(TensorImpl[Mem, Layout])
    t.data.new()
    t.data[] = engine.allocDevice(bfloat16, t.layout.size())
  else:
    raiseAssert("materialize: dtype unsupported")

proc newTensorNode*[E: HwEngine, T, Mem](engine: E, _: typedesc[T], layout: Layout, _: typedesc[Mem]): TensorNode =
  engine.newTensorImpl(T, layout, Mem).to(TensorNode)

proc newPlaceholderNode*[E: HwEngine](engine: E, dtype: Dtype, shape: Span[int]): TensorNode =
  case dtype
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    var sh: (int, int)
    fillShape(sh, shape)
    engine.newPlaceholderImpl(make_layout(sh, rowMajorStrides(sh)), Mem).to(TensorNode)
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    var sh: (int, int)
    fillShape(sh, shape)
    engine.newPlaceholderImpl(make_layout(sh, rowMajorStrides(sh)), Mem).to(TensorNode)
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
      type Mem = typeof(engine.allocDevice(float32, 0))
      engine.newTensorNode(float32, layout, Mem)
    of kBf16:
      type Mem = typeof(engine.allocDevice(bfloat16, 0))
      engine.newTensorNode(bfloat16, layout, Mem)
    else:
      raise ValueError.newException("newTensorNode: dtype " & $dtype & " unsupported")
  else:
    raise ValueError.newException("newTensorNode: rank " & $shape.len & " unsupported")
