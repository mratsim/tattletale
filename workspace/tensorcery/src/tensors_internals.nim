# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import std/[macros, typetraits]
import workspace/crucible
import workspace/ceramic
import workspace/ceramic/src/macros/static_for
import workspace/math
import workspace/iface
import workspace/design_by_contract
import ./tensors_internal_types
import ./tensors_basic_views
import ./compute_graph/batching

export tensors_internal_types
export tensors_basic_views

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

func shape*[DeviceMem](t: TensorImpl[DeviceMem, LayoutBasic]): Span[int] {.inline.} =
  Span[int](p: toOpenArray(t.layout.shape, 0, t.layout.rank - 1))

func strides*[DeviceMem](t: TensorImpl[DeviceMem, LayoutBasic]): Span[int] {.inline.} =
  Span[int](p: toOpenArray(t.layout.strides, 0, t.layout.rank - 1))

func dtype*[H, T](t: TensorImpl[DeviceMem[H, T], LayoutBasic]): Dtype {.inline.} =
  toDtype(T)

func offset*[DeviceMem](t: TensorImpl[DeviceMem, LayoutBasic]): int {.inline.} =
  t.offset

func size*[DeviceMem](t: TensorImpl[DeviceMem, LayoutBasic]): int {.inline.} =
  result = 1
  for i in 0 ..< t.layout.rank:
    result *= t.layout.shape[i]

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
  ## Casts the type-erased impl tensor to its concrete TensorImpl type
  ## and make it available as the symbol passed as "concreteTensor"
  ##
  ## Usage:
  ##   engine.withDeviceImpl(node.getImpl(), float32, t):
  ##     engine.copyToHostFromDevice(dst, t.data[], t.offset)
  let concreteTensor = impl.to(engine.getTensorImplType(T))
  body

proc download1*[E: HwEngine](engine: E, dst: pointer, node: TensorNode) =
  let impl = node.getImpl()
  match(ElemType, impl.dtype()):
    let ci = impl.to(engine.getTensorImplType(ElemType))
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

proc newPlaceholderImpl[E: HwEngine, T](engine: E, _: typedesc[T], shape: openArray[int]): engine.getTensorImplType(T) {.inline.} =
  result = (engine.getTensorImplType(T))()
  result.layout = toLayoutBasic(shape)

proc newTensorImpl[E: HwEngine, T](engine: E, _: typedesc[T], shape: openArray[int]): engine.getTensorImplType(T) {.inline.} =
  result = (engine.getTensorImplType(T))()
  result.layout = toLayoutBasic(shape)
  result.data.new()
  result.data[] = engine.allocDevice(T, result.size())

# ═══════════════════════════════════════════════════════════════════════
#  Graph
# ═══════════════════════════════════════════════════════════════════════

proc preAlloc*[E: HwEngine](engine: E, impl: TensorImpl) {.inline.} =
  engine.allocDataIn(impl)

proc newPlaceholderNode*[E: HwEngine](engine: E, dtype: Dtype, shape: openArray[int]): TensorNode =
  match(ElemType, dtype):
    engine.newPlaceholderImpl(ElemType, shape).to(TensorNode)

proc newViewNode*[E: HwEngine](engine: E, src: TensorNode, shape, strides: openArray[int], offset: int, srcOpName: string): TensorNode =
  let si = src.getImpl()
  match(ElemType, si.dtype()):
    let r = engine.getTensorImplType(ElemType)()
    r.layout = toLayoutBasic(shape, strides)
    r.offset = offset
    engine.withDeviceImpl(si, ElemType, s):
      r.data = s.data
      if s.isPreAllocated():
        r.offset = s.offset + offset
    ViewNode(r: r.to(AnyTensorImpl), src: src, srcOpName: srcOpName).to(TensorNode)

proc newTensorNode*[E: HwEngine](engine: E, dtype: Dtype, shape: openArray[int]): TensorNode =
  match(ElemType, dtype):
    engine.newTensorImpl(ElemType, shape).to(TensorNode)

proc collectView*(n: TensorNode, stack: var seq[WalkFrame]) =
  ## View collect phase: the source node is the only operand.
  let v = n.to(ViewNode)
  stack.add WalkFrame(n: v.src, kernelPending: false)

proc enqueueView[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: TensorNode, _: typedesc[T]) =
  ## View kernel phase: binds the view's data and offset to its source.
  let v = n.to(ViewNode)
  let r = v.r.to(engine.getTensorImplType(T))
  engine.withDeviceImpl(v.src.getImpl(), T, s):
    r.data = s.data
    r.offset = r.offset + s.offset

macro dispatchViewKernel*(engine, n, batch: typed): untyped =
  ## Kernel phase of one view node. False when `n` is not a view.
  ## The walk wires views outside the marked op registry.
  result = quote do:
    block:
      let v = `n`.to(ViewNode)
      if v.isNil():
        false
      else:
        case `n`.getImpl().dtype()
        of kFp16:
          type T = float16
          enqueueView(`engine`, `batch`, `n`, T)
          true
        of kFp32:
          type T = float32
          enqueueView(`engine`, `batch`, `n`, T)
          true
        of kBf16:
          type T = bfloat16
          enqueueView(`engine`, `batch`, `n`, T)
          true
        else:
          raise ValueError.newException("eval: dtype unsupported")

# ═══════════════════════════════════════════════════════════════════════
#  Basic -> Ranked metadata relayout (M4 groundwork)
# ═══════════════════════════════════════════════════════════════════════
#
# Critical op kernels (gemm, linear, rmsnorm) keep static-rank ceramic Layout signatures.
# Handlers rebuild bridge arguments from basic metadata. Nothing moves and no ranked impl materializes.

func bridgeRank*[H, T; Sh, St: tuple](b: TensorBridge[DeviceMem[H, T], Layout[Sh, St]]): int {.inline.} =
  ## Static rank of the bridge's layout tuple.

func relayoutBridge*[H, T; Sh: tuple, St: tuple](
    impl: TensorImpl[DeviceMem[H, T], LayoutBasic],
    _: typedesc[Layout[Sh, St]]): TensorBridge[DeviceMem[H, T], Layout[Sh, St]] {.inline.} =
  ## Bridge of `impl` retyped to the static-rank `Layout[Sh, St]` whose
  ## shape and stride leaves carry the impl's runtime values.
  ##
  ## Precondition: the impl's runtime rank equals the layout's rank.
  static:
    doAssert tupleLen(Sh) == tupleLen(St)
  doAssert impl.layout.rank == tupleLen(Sh),
    "relayoutBridge: rank " & $impl.layout.rank & " does not match the target layout rank " & $tupleLen(Sh)
  var sh: Sh
  var st: St
  staticFor i, 0, tupleLen(Sh):
    sh[i] = impl.layout.shape[i]
    st[i] = impl.layout.strides[i]
  initTensorBridge(impl.data, impl.offset, make_layout(sh, st))

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

