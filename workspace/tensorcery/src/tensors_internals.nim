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
#  Output slot - a node's deferred output, metadata only
# ═══════════════════════════════════════════════════════════════════════

type SlotImpl* = ref object
  dt*: Dtype
  sh*: seq[int]
  st*: seq[int]

func dtype*(s: SlotImpl): Dtype =
  s.dt

func shape*(s: SlotImpl): Span[int] =
  s.sh.toSpan()

func strides*(s: SlotImpl): Span[int] =
  s.st.toSpan()

func toSeq*(s: Span[int]): seq[int] =
  result.newSeq(s.len)
  for i in 0 ..< s.len:
    result[i] = s[i]

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - AnyTensorImpl implementation
# ═══════════════════════════════════════════════════════════════════════

func shape*[Sh: tuple, Mem](t: TensorImpl[Mem, Layout[Sh, Sh]]): Span[int] =
  toSpan(t.layout.shape)

func strides*[Sh: tuple, Mem](t: TensorImpl[Mem, Layout[Sh, Sh]]): Span[int] =
  toSpan(t.layout.stride)

func dtype*[Sh: tuple, H, T](t: TensorImpl[DeviceMem[H, T], Layout[Sh, Sh]]): Dtype =
  toDtype(T)

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - TensorNode implementation (materialized tensors)
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*[Mem, L](t: TensorImpl[Mem, L]): AnyTensorImpl =
  t.to(AnyTensorImpl)

func opName*[Mem, L](t: TensorImpl[Mem, L]): string =
  ## A materialized tensor is a node with no op, the walk leaves it alone
  ""

# ═══════════════════════════════════════════════════════════════════════
#  Typed data path - AnyEngine implementation
# ═══════════════════════════════════════════════════════════════════════

proc upload*[E: HwEngine](engine: E, node: TensorNode, src: openArray[float32]) =
  let impl = node.getImpl()
  type Mem = typeof(engine.allocDevice(float32, 0))
  let t = impl.to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
  engine.copyToDeviceFromHost(t.data[], src)

proc upload*[E: HwEngine](engine: E, node: TensorNode, src: openArray[bfloat16]) =
  type Mem = typeof(engine.allocDevice(bfloat16, 0))
  let t = node.getImpl().to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
  engine.copyToDeviceFromHost(t.data[], src)

proc download*[E: HwEngine](engine: E, dst: var openArray[float32], node: TensorNode) =
  engine.download(dst, node, 0)

proc download*[E: HwEngine](engine: E, dst: var openArray[bfloat16], node: TensorNode) =
  engine.download(dst, node, 0)

proc download*[E: HwEngine](engine: E, dst: var openArray[float32], node: TensorNode, offsetElems: int) =
  type Mem = typeof(engine.allocDevice(float32, 0))
  let t = node.getImpl().to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
  engine.copyToHostFromDevice(dst, t.data[], t.offset + offsetElems)

proc download*[E: HwEngine](engine: E, dst: var openArray[bfloat16], node: TensorNode, offsetElems: int) =
  type Mem = typeof(engine.allocDevice(bfloat16, 0))
  let t = node.getImpl().to(TensorImpl[Mem, Layout[(int, int), (int, int)]])
  engine.copyToHostFromDevice(dst, t.data[], t.offset + offsetElems)

# ═══════════════════════════════════════════════════════════════════════
#  The erased engine
# ═══════════════════════════════════════════════════════════════════════

proc toAnyEngine*[E: HwEngine](engine: E): AnyEngine =
  engine.to(AnyEngine)

proc materializeImpl[E: HwEngine, Sh: tuple, St: tuple, T, Mem](engine: E, _: typedesc[T], layout: Layout[Sh, St]): TensorImpl[Mem, Layout[Sh, St]] =
  ## Mem is the engine's device memory for T, bound by the caller's dtype case
  var t = TensorImpl[Mem, Layout[Sh, St]](data: nil, offset: 0, layout: layout)
  t.data.new()
  t.data[] = engine.allocDevice(T, layout.shape.getNumElems())
  t

proc materializeNode[E: HwEngine, Sh: tuple, St: tuple, T, Mem](engine: E, _: typedesc[T], layout: Layout[Sh, St]): TensorNode =
  ## Erased node of a materialized tensor, the impl itself is the node
  materializeImpl[E, Sh, St, T, Mem](engine, T, layout).to(TensorNode)

proc makeTensorImpl[E: HwEngine, Sh: tuple, St: tuple](engine: E, dtype: Dtype, layout: Layout[Sh, St]): AnyTensorImpl =
  case dtype
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    materializeImpl[E, Sh, St, float32, Mem](engine, float32, layout).to(AnyTensorImpl)
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    materializeImpl[E, Sh, St, bfloat16, Mem](engine, bfloat16, layout).to(AnyTensorImpl)
  else:
    raise ValueError.newException("makeTensorImpl: dtype " & $dtype & " unsupported")

proc makeTensorImpl*[E: HwEngine](engine: E, dtype: Dtype, shape: Span[int]): AnyTensorImpl =
  case shape.len
  of 2:
    var sh: (int, int)
    fillShape(sh, shape)
    engine.makeTensorImpl(dtype, make_layout(sh, rowMajorStrides(sh)))
  else:
    raise ValueError.newException("makeTensorImpl: rank " & $shape.len & " unsupported")

proc makeTensorImpl*[E: HwEngine](engine: E, dtype: Dtype, shape: openArray[int]): AnyTensorImpl =
  engine.makeTensorImpl(dtype, shape.toSpan())

proc newViewNode*[E: HwEngine](engine: E, dtype: Dtype, shape: Span[int]): TensorNode =
  ## A graph node shell without a buffer, view ops bind the data at enqueue
  case dtype
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    var sh: (int, int)
    fillShape(sh, shape)
    TensorImpl[Mem, Layout[(int, int), (int, int)]](
      data: nil, offset: 0,
      layout: make_layout(sh, rowMajorStrides(sh))).to(TensorNode)
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    var sh: (int, int)
    fillShape(sh, shape)
    TensorImpl[Mem, Layout[(int, int), (int, int)]](
      data: nil, offset: 0,
      layout: make_layout(sh, rowMajorStrides(sh))).to(TensorNode)
  else:
    raise ValueError.newException("newViewNode: dtype " & $dtype & " unsupported")

proc newTensorNode*[E: HwEngine](engine: E, dtype: Dtype, shape: Span[int]): TensorNode =
  case shape.len
  of 2:
    var sh: (int, int)
    fillShape(sh, shape)
    let layout = make_layout(sh, rowMajorStrides(sh))
    case dtype
    of kFp32:
      type Mem = typeof(engine.allocDevice(float32, 0))
      materializeNode[E, (int, int), (int, int), float32, Mem](engine, float32, layout)
    of kBf16:
      type Mem = typeof(engine.allocDevice(bfloat16, 0))
      materializeNode[E, (int, int), (int, int), bfloat16, Mem](engine, bfloat16, layout)
    else:
      raise ValueError.newException("newTensorNode: dtype " & $dtype & " unsupported")
  else:
    raise ValueError.newException("newTensorNode: rank " & $shape.len & " unsupported")
