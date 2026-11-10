# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/typetraits
import workspace/crucible
import workspace/crucible/src/builtins/builtins_gpu_types
export builtins_gpu_types
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/ceramic/src/macros/static_for
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
#  TensorImpl - AnyTensorImpl implementation
# ═══════════════════════════════════════════════════════════════════════

func shape*[Sh: tuple, Mem](t: TensorImpl[Mem, Layout[Sh, Sh]]): Span[int] =
  toSpan(t.layout.shape)

func strides*[Sh: tuple, Mem](t: TensorImpl[Mem, Layout[Sh, Sh]]): Span[int] =
  toSpan(t.layout.stride)

func offset*[Sh: tuple, Mem](t: TensorImpl[Mem, Layout[Sh, Sh]]): int =
  t.offset

func capacity*[Mem, L](t: TensorImpl[Mem, L]): int =
  ## Returns the backing buffer's element count:
  ##
  ## - a materialized tensor owns exactly its layout's element count
  ## - a view impl reads the source's buffer, so its capacity is the source's
  ## - an op-output shell's capacity equals the dense element count of its shape
  t.capacity

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

func crd2Offset*(impl: AnyTensorImpl, crd: openArray[int]): int =
  ## Flat element offset of a coordinate in an erased impl's layout.
  ## crd2idx over the impl's runtime shape and stride tuples, one branch per rank.
  case impl.shape().len
  of 1:
    var sh, st, c: (int,)
    fillShape(sh, impl.shape())
    fillShape(st, impl.strides())
    fillShape(c, crd.toSpan())
    crd2idxImpl(c, sh, st)
  of 2:
    var sh, st, c: (int, int)
    fillShape(sh, impl.shape())
    fillShape(st, impl.strides())
    fillShape(c, crd.toSpan())
    crd2idxImpl(c, sh, st)
  of 3:
    var sh, st, c: (int, int, int)
    fillShape(sh, impl.shape())
    fillShape(st, impl.strides())
    fillShape(c, crd.toSpan())
    crd2idxImpl(c, sh, st)
  of 4:
    var sh, st, c: (int, int, int, int)
    fillShape(sh, impl.shape())
    fillShape(st, impl.strides())
    fillShape(c, crd.toSpan())
    crd2idxImpl(c, sh, st)
  of 5:
    var sh, st, c: (int, int, int, int, int)
    fillShape(sh, impl.shape())
    fillShape(st, impl.strides())
    fillShape(c, crd.toSpan())
    crd2idxImpl(c, sh, st)
  else:
    raise ValueError.newException(
      "crd2Offset: rank " & $impl.shape().len & " unsupported")

# ═══════════════════════════════════════════════════════════════════════
#  Typed data path - AnyEngine implementation
# ═══════════════════════════════════════════════════════════════════════

template withRankedImpl(Mem: typedesc, impl: AnyTensorImpl, body: untyped) =
  ## Casts an erased impl to its concrete TensorImpl at the impl's own rank,
  ## binding `t` over `body`. The layout tuple types fill from the erased
  ## shape and strides spans, so views with custom strides cast too.
  case impl.shape().len
  of 1:
    var sh, st: (int,)
    fillShape(sh, impl.shape())
    fillShape(st, impl.strides())
    let t {.inject.} = impl.to(TensorImpl[Mem, Layout[typeof(sh), typeof(st)]])
    body
  of 2:
    var sh, st: (int, int)
    fillShape(sh, impl.shape())
    fillShape(st, impl.strides())
    let t {.inject.} = impl.to(TensorImpl[Mem, Layout[typeof(sh), typeof(st)]])
    body
  of 3:
    var sh, st: (int, int, int)
    fillShape(sh, impl.shape())
    fillShape(st, impl.strides())
    let t {.inject.} = impl.to(TensorImpl[Mem, Layout[typeof(sh), typeof(st)]])
    body
  of 4:
    var sh, st: (int, int, int, int)
    fillShape(sh, impl.shape())
    fillShape(st, impl.strides())
    let t {.inject.} = impl.to(TensorImpl[Mem, Layout[typeof(sh), typeof(st)]])
    body
  of 5:
    var sh, st: (int, int, int, int, int)
    fillShape(sh, impl.shape())
    fillShape(st, impl.strides())
    let t {.inject.} = impl.to(TensorImpl[Mem, Layout[typeof(sh), typeof(st)]])
    body
  else:
    raise ValueError.newException("rank " & $impl.shape().len & " unsupported")

proc upload*[E: HwEngine](engine: E, node: TensorNode, src: openArray[float32]) =
  type Mem = typeof(engine.allocDevice(float32, 0))
  withRankedImpl(Mem, node.getImpl()):
    engine.copyToDeviceFromHost(t.data[], src)

proc upload*[E: HwEngine](engine: E, node: TensorNode, src: openArray[bfloat16]) =
  type Mem = typeof(engine.allocDevice(bfloat16, 0))
  withRankedImpl(Mem, node.getImpl()):
    engine.copyToDeviceFromHost(t.data[], src)

proc download*[E: HwEngine](engine: E, dst: var openArray[float32], node: TensorNode) =
  engine.download(dst, node, 0)

proc download*[E: HwEngine](engine: E, dst: var openArray[bfloat16], node: TensorNode) =
  engine.download(dst, node, 0)

proc download*[E: HwEngine](engine: E, dst: var openArray[float32], node: TensorNode, offsetElems: int) =
  type Mem = typeof(engine.allocDevice(float32, 0))
  withRankedImpl(Mem, node.getImpl()):
    engine.copyToHostFromDevice(dst, t.data[], t.offset + offsetElems)

proc download*[E: HwEngine](engine: E, dst: var openArray[bfloat16], node: TensorNode, offsetElems: int) =
  type Mem = typeof(engine.allocDevice(bfloat16, 0))
  withRankedImpl(Mem, node.getImpl()):
    engine.copyToHostFromDevice(dst, t.data[], t.offset + offsetElems)

proc downloadItem*[E: HwEngine](engine: E, dst: var float32, node: TensorNode, offsetElems: int) =
  var one: array[1, float32]
  engine.download(one, node, offsetElems)
  dst = one[0]

proc downloadItem*[E: HwEngine](engine: E, dst: var bfloat16, node: TensorNode, offsetElems: int) =
  var one: array[1, bfloat16]
  engine.download(one, node, offsetElems)
  dst = one[0]

proc gatherToHost[E: HwEngine, Sh: tuple, St: tuple, Mem, T](engine: E, t: TensorImpl[Mem, Layout[Sh, St]], dst: var openArray[T]) =
  ## Reads back a strided or offset view: one flat DMA over the layout's reachable window,
  ## then a host-side gather by the layout.
  ##
  ## The engine DMA is flat over the whole buffer. The view's reachable window
  ## copies into a host scratch first:
  ##
  ## - window: buffer addresses 0 .. maxAddr, maxAddr being the highest
  ##   element address the layout reaches
  ## - dst[i]: the window entry at element i's layout address (elements ordered by ascending flat index)
  ## - addr(i) = t.offset + sum_a crd_a(i) * stride_a, crd decomposed so the last axis varies fastest
  ##
  ## A negative stride reaches below t.offset, so the window starts at the buffer base.
  ## The view contracts keep every reachable address inside [0, capacity).
  let numel = t.layout.shape.getNumElems()
  var maxAddr = t.offset
  staticFor a, 0, tupleLen(Sh):
    let reach = t.layout.stride[a] * (t.layout.shape[a] - 1)
    if reach > 0:
      maxAddr += reach
  var window = newSeq[T](maxAddr + 1)
  engine.copyToHostFromDevice(window, t.data[], 0)
  var crd: Sh
  for i in 0 ..< numel:
    dst[i] = window[t.offset + crd2idxImpl(crd, t.layout.shape, t.layout.stride)]
    # carry the coordinate: the last axis varies fastest
    var carry = true
    staticFor a, 0, tupleLen(Sh):
      const ax = tupleLen(Sh) - 1 - a
      if carry:
        inc crd[ax]
        if crd[ax] < t.layout.shape[ax]:
          carry = false
        else:
          crd[ax] = 0

proc gatherToHost*[E: HwEngine](engine: E, dst: var openArray[float32], node: TensorNode) =
  type Mem = typeof(engine.allocDevice(float32, 0))
  withRankedImpl(Mem, node.getImpl()):
    engine.gatherToHost(t, dst)

proc gatherToHost*[E: HwEngine](engine: E, dst: var openArray[bfloat16], node: TensorNode) =
  type Mem = typeof(engine.allocDevice(bfloat16, 0))
  withRankedImpl(Mem, node.getImpl()):
    engine.gatherToHost(t, dst)

# ═══════════════════════════════════════════════════════════════════════
#  The erased engine
# ═══════════════════════════════════════════════════════════════════════

proc toAnyEngine*[E: HwEngine](engine: E): AnyEngine =
  engine.to(AnyEngine)

proc materializeImpl[E: HwEngine, Sh: tuple, St: tuple, T, Mem](engine: E, _: typedesc[T], layout: Layout[Sh, St]): TensorImpl[Mem, Layout[Sh, St]] =
  ## Mem is the engine's device memory for T, bound by the caller's dtype case
  var t = TensorImpl[Mem, Layout[Sh, St]](
    data: nil, offset: 0, capacity: layout.shape.getNumElems(), layout: layout)
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
  of 1:
    var sh: (int,)
    fillShape(sh, shape)
    engine.makeTensorImpl(dtype, make_layout(sh, rowMajorStrides(sh)))
  of 2:
    var sh: (int, int)
    fillShape(sh, shape)
    engine.makeTensorImpl(dtype, make_layout(sh, rowMajorStrides(sh)))
  of 3:
    var sh: (int, int, int)
    fillShape(sh, shape)
    engine.makeTensorImpl(dtype, make_layout(sh, rowMajorStrides(sh)))
  of 4:
    var sh: (int, int, int, int)
    fillShape(sh, shape)
    engine.makeTensorImpl(dtype, make_layout(sh, rowMajorStrides(sh)))
  of 5:
    var sh: (int, int, int, int, int)
    fillShape(sh, shape)
    engine.makeTensorImpl(dtype, make_layout(sh, rowMajorStrides(sh)))
  else:
    raise ValueError.newException("makeTensorImpl: rank " & $shape.len & " unsupported")

proc makeTensorImpl*[E: HwEngine](engine: E, dtype: Dtype, shape: openArray[int]): AnyTensorImpl =
  engine.makeTensorImpl(dtype, shape.toSpan())

proc newViewNode[E: HwEngine, Sh: tuple, St: tuple](engine: E, dtype: Dtype, layout: Layout[Sh, St]): TensorNode =
  case dtype
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    TensorImpl[Mem, Layout[Sh, St]](
      data: nil, offset: 0, capacity: layout.shape.getNumElems(), layout: layout).to(TensorNode)
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    TensorImpl[Mem, Layout[Sh, St]](
      data: nil, offset: 0, capacity: layout.shape.getNumElems(), layout: layout).to(TensorNode)
  else:
    raise ValueError.newException("newViewNode: dtype " & $dtype & " unsupported")

proc newViewNode*[E: HwEngine](engine: E, dtype: Dtype, shape: Span[int]): TensorNode =
  ## A graph node shell without a buffer, view ops bind the data at enqueue
  case shape.len
  of 1:
    var sh: (int,)
    fillShape(sh, shape)
    engine.newViewNode(dtype, make_layout(sh, rowMajorStrides(sh)))
  of 2:
    var sh: (int, int)
    fillShape(sh, shape)
    engine.newViewNode(dtype, make_layout(sh, rowMajorStrides(sh)))
  of 3:
    var sh: (int, int, int)
    fillShape(sh, shape)
    engine.newViewNode(dtype, make_layout(sh, rowMajorStrides(sh)))
  of 4:
    var sh: (int, int, int, int)
    fillShape(sh, shape)
    engine.newViewNode(dtype, make_layout(sh, rowMajorStrides(sh)))
  of 5:
    var sh: (int, int, int, int, int)
    fillShape(sh, shape)
    engine.newViewNode(dtype, make_layout(sh, rowMajorStrides(sh)))
  else:
    raise ValueError.newException("newViewNode: rank " & $shape.len & " unsupported")

proc spanViewNode[Mem](src: AnyTensorImpl, shape, strides: Span[int], offset: int): TensorNode =
  ## Body of newViewImpl: one rank branch per tuple shape
  case shape.len
  of 1:
    var sh: (int,)
    var st: (int,)
    fillShape(sh, shape)
    fillShape(st, strides)
    TensorImpl[Mem, Layout[typeof(sh), typeof(st)]](
      data: nil, offset: offset, capacity: src.capacity(),
      layout: make_layout(sh, st)).to(TensorNode)
  of 2:
    var sh: (int, int)
    var st: (int, int)
    fillShape(sh, shape)
    fillShape(st, strides)
    TensorImpl[Mem, Layout[typeof(sh), typeof(st)]](
      data: nil, offset: offset, capacity: src.capacity(),
      layout: make_layout(sh, st)).to(TensorNode)
  of 3:
    var sh: (int, int, int)
    var st: (int, int, int)
    fillShape(sh, shape)
    fillShape(st, strides)
    TensorImpl[Mem, Layout[typeof(sh), typeof(st)]](
      data: nil, offset: offset, capacity: src.capacity(),
      layout: make_layout(sh, st)).to(TensorNode)
  of 4:
    var sh: (int, int, int, int)
    var st: (int, int, int, int)
    fillShape(sh, shape)
    fillShape(st, strides)
    TensorImpl[Mem, Layout[typeof(sh), typeof(st)]](
      data: nil, offset: offset, capacity: src.capacity(),
      layout: make_layout(sh, st)).to(TensorNode)
  of 5:
    var sh: (int, int, int, int, int)
    var st: (int, int, int, int, int)
    fillShape(sh, shape)
    fillShape(st, strides)
    TensorImpl[Mem, Layout[typeof(sh), typeof(st)]](
      data: nil, offset: offset, capacity: src.capacity(),
      layout: make_layout(sh, st)).to(TensorNode)
  else:
    raise ValueError.newException("newViewImpl: rank " & $shape.len & " unsupported")

proc newViewImpl*[E: HwEngine](engine: E, src: AnyTensorImpl, shape, strides: Span[int], offset: int): TensorNode =
  ## Shared-buffer view node over `src`. The new impl aliases the source's
  ## buffer cell with its own layout, element offset and capacity. Nothing allocates and nothing copies.
  ##
  ## The buffer cell binds at enqueue, after the source's own eval walked.
  ## The layout, offset and capacity are final here.
  ##
  ## Args:
  ## - offset: the view origin's element offset in the backing buffer,
  ##   including the source tensor's own offset
  ## - shape, strides: any rank 1 to 5, any strides, reshape changes the rank
  case src.dtype()
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    spanViewNode[Mem](src, shape, strides, offset)
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    spanViewNode[Mem](src, shape, strides, offset)
  else:
    raise ValueError.newException("newViewImpl: dtype " & $src.dtype() & " unsupported")

proc newTensorNode[E: HwEngine, Sh: tuple, St: tuple](engine: E, dtype: Dtype, layout: Layout[Sh, St]): TensorNode =
  case dtype
  of kFp32:
    type Mem = typeof(engine.allocDevice(float32, 0))
    materializeNode[E, Sh, St, float32, Mem](engine, float32, layout)
  of kBf16:
    type Mem = typeof(engine.allocDevice(bfloat16, 0))
    materializeNode[E, Sh, St, bfloat16, Mem](engine, bfloat16, layout)
  else:
    raise ValueError.newException("newTensorNode: dtype " & $dtype & " unsupported")

proc newTensorNode*[E: HwEngine](engine: E, dtype: Dtype, shape: Span[int]): TensorNode =
  case shape.len
  of 1:
    var sh: (int,)
    fillShape(sh, shape)
    engine.newTensorNode(dtype, make_layout(sh, rowMajorStrides(sh)))
  of 2:
    var sh: (int, int)
    fillShape(sh, shape)
    engine.newTensorNode(dtype, make_layout(sh, rowMajorStrides(sh)))
  of 3:
    var sh: (int, int, int)
    fillShape(sh, shape)
    engine.newTensorNode(dtype, make_layout(sh, rowMajorStrides(sh)))
  of 4:
    var sh: (int, int, int, int)
    fillShape(sh, shape)
    engine.newTensorNode(dtype, make_layout(sh, rowMajorStrides(sh)))
  of 5:
    var sh: (int, int, int, int, int)
    fillShape(sh, shape)
    engine.newTensorNode(dtype, make_layout(sh, rowMajorStrides(sh)))
  else:
    raise ValueError.newException("newTensorNode: rank " & $shape.len & " unsupported")
