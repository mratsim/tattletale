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

func shape*(t: TensorImpl): Span[int] {.inline.} =
  Span[int](p: toOpenArray(t.layout.shape, 0, t.layout.rank - 1))

func strides*(t: TensorImpl): Span[int] {.inline.} =
  Span[int](p: toOpenArray(t.layout.strides, 0, t.layout.rank - 1))

func dtype*[H, T](t: TensorImpl[DeviceMem[H, T], LayoutBasic]): Dtype {.inline.} =
  toDtype(T)

func offset*(t: TensorImpl): int {.inline.} =
  t.offset

func size*(t: TensorImpl): int {.inline.} =
  result = 1
  for i in 0 ..< t.layout.rank:
    result *= t.layout.shape[i]

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl - TensorGraph interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*(t: TensorImpl): AnyTensorImpl {.inline.} =
  t.to(AnyTensorImpl)

func opName*(t: TensorImpl): string {.inline.} =
  ## Name in the computation graph
  "Tensor"

# ═══════════════════════════════════════════════════════════════════════
#  Host<->Device copies
# ═══════════════════════════════════════════════════════════════════════

proc download1*[E: HwEngine](engine: E, dst: pointer, node: TensorGraph) =
  let impl = node.getImpl()
  match(ElemType, impl.dtype()):
    let ci = impl.to(engine.getTensorImplType(ElemType))
    engine.copyToHostFromDevice(cast[ptr ElemType](dst)[], ci.data[], ci.offset)

proc upload*[E: HwEngine](engine: E, node: TensorGraph, src: openArray[float32]) =
  let t = node.getImpl().to(engine.getTensorImplType(float32))
  engine.copyToDeviceFromHost(t.data[], src)

proc upload*[E: HwEngine](engine: E, node: TensorGraph, src: openArray[float16]) =
  let t = node.getImpl().to(engine.getTensorImplType(float16))
  engine.copyToDeviceFromHost(t.data[], src)

proc upload*[E: HwEngine](engine: E, node: TensorGraph, src: openArray[bfloat16]) =
  let t = node.getImpl().to(engine.getTensorImplType(bfloat16))
  engine.copyToDeviceFromHost(t.data[], src)

# ═══════════════════════════════════════════════════════════════════════
#  TensorImpl
# ═══════════════════════════════════════════════════════════════════════

proc newPlaceholderImpl[E: HwEngine, T](engine: E, _: typedesc[T], shape: openArray[int]): engine.getTensorImplType(T) {.inline.} =
  result = (engine.getTensorImplType(T))()
  result.layout = makeLayoutBasic(shape)

proc newTensorImpl[E: HwEngine, T](engine: E, _: typedesc[T], shape: openArray[int]): engine.getTensorImplType(T) {.inline.} =
  result = (engine.getTensorImplType(T))()
  result.layout = makeLayoutBasic(shape)
  result.data.new()
  result.data[] = engine.allocDevice(T, result.size())

# ═══════════════════════════════════════════════════════════════════════
#  Graph
# ═══════════════════════════════════════════════════════════════════════

proc newPlaceholderNode*[E: HwEngine](engine: E, dtype: Dtype, shape: openArray[int]): TensorGraph =
  match(ElemType, dtype):
    engine.newPlaceholderImpl(ElemType, shape).to(TensorGraph)

proc newViewNode*[E: HwEngine](engine: E, src: TensorGraph, shape, strides: openArray[int], offset: int, srcOpName: string): TensorGraph =
  let si = src.getImpl()
  match(ElemType, si.dtype()):
    let r = engine.getTensorImplType(ElemType)()
    r.layout = makeLayoutBasic(shape, strides)
    r.offset = offset
    let s = si.to(engine.getTensorImplType(ElemType))
    r.data = s.data
    if s.isPreAllocated():
      r.offset = s.offset + offset
    ViewNode(r: r.to(AnyTensorImpl), src: src, srcOpName: srcOpName).to(TensorGraph)

proc newTensorNode*[E: HwEngine](engine: E, dtype: Dtype, shape: openArray[int]): TensorGraph =
  match(ElemType, dtype):
    engine.newTensorImpl(ElemType, shape).to(TensorGraph)

proc enqueueView[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: TensorGraph, _: typedesc[T]) =
  let v = n.to(ViewNode)
  let r = v.r.to(engine.getTensorImplType(T))
  let s = v.src.getImpl().to(engine.getTensorImplType(T))
  r.data = s.data
  r.offset = r.offset + s.offset

macro dispatchViewKernel*(engine, n, batch: typed): untyped =
  ## Kernel phase of one view node. False when `n` is not a view.
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
#  Basic -> Ranked metadata relayout
# ═══════════════════════════════════════════════════════════════════════

func relayoutBridge*[H, T; L: Layout](
    impl: TensorImpl[DeviceMem[H, T], LayoutBasic],
    _: typedesc[L]): TensorBridge[DeviceMem[H, T], L] {.inline.} =
  ## Bridge of `impl` retyped to the static-rank layout `L` whose
  ## shape and stride leaves carry the impl's runtime values.
  ##
  ## Precondition: the impl's runtime rank equals L's rank.
  if impl.isNil():
    return TensorBridge[DeviceMem[H, T], L]()
  precondition((impl.layout.rank == L.rank(), "relayoutBridge: rank " & $impl.layout.rank & " does not match the target layout rank " & $L.rank()))
  var lay: L
  staticFor i, 0, L.rank():
    lay.shape[i] = impl.layout.shape[i]
    lay.stride[i] = impl.layout.strides[i]
  initTensorBridge(impl.data, impl.offset, lay)
