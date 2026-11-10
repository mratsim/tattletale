# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
import workspace/ceramic
import workspace/iface
import ../tensors_internals
import dispatch
import workspace/design_by_contract

# #######################################################################
#
#                   Shared-buffer views
#
# #######################################################################
#
# A view is a shape, a stride tuple and an element offset over the source's buffer.
# One impl aliases the source's buffer cell, no allocation, no copy.
# Narrow, transpose, permute, reshape, expand and the general view face all
# build the same node.
#
# Invariants:
#
# - A view impl's metadata (layout, offset, capacity) is final at the face.
#   Each face reads the source's erased metadata, which is final by induction.
# - A materialized tensor owns its real values. An op-output shell names
#   what its buffer will have (offset 0, a dense shape, promised capacity).
#   A view built earlier already has final values.
# - Only the buffer cell binds at enqueue, after the source's own eval walked.
# - A view stays inside the backing buffer. Each face contract checks
#   reachability against the source's capacity, in both stride signs.
#   Every face either enforces this or builds a layout structurally inside
#   its source's own window.

# #######################################################################
#  Defunctionalization
# #######################################################################

type
  ViewNode* = ref object
    src: TensorNode      # walked before the view's buffer binds
    v: TensorNode        # view impl shell, metadata-complete at the face

proc getImpl*(n: ViewNode): AnyTensorImpl =
  n.v.getImpl()

func opName*(n: ViewNode): string =
  "view"

# #######################################################################
#  Graph - Public
# #######################################################################

func nonNegativeDims*(shape: openArray[int]): bool =
  for d in shape:
    if d < 0:
      return false
  true

func reachWindow(shape, strides: Span[int], base: int): tuple[lo, hi: int] =
  ## Lowest and highest element addresses a layout reaches. Axis a reaches
  ## stride[a]·(shape[a]-1) elements from the origin, upward for a positive
  ## stride and downward for a negative one.
  result = (base, base)
  for i in 0 ..< shape.len:
    let reach = if shape[i] > 0: strides[i] * (shape[i] - 1) else: 0
    if reach >= 0:
      result.hi += reach
    else:
      result.lo += reach

func inBackingCapacity(src: AnyTensorImpl, shape, strides: Span[int], relOffset: int): bool =
  ## Checks that the view's reachable addresses stay inside the backing buffer.
  ## The base is the source tensor's own origin plus the view's relative offset.
  let (lo, hi) = reachWindow(shape, strides, src.offset() + relOffset)
  lo >= 0 and hi < src.capacity()

func denseContiguous(shape, strides: Span[int]): bool =
  ## Last axis stride 1, every earlier stride the product of the later
  ## strides and shapes: the layout is one dense run over its window.
  if strides[shape.len - 1] != 1:
    return false
  for i in 0 ..< shape.len - 1:
    if strides[i] != strides[i + 1] * shape[i + 1]:
      return false
  true

func isPermutation(axes: openArray[int], rank: int): bool =
  ## Every axis in 0 ..< rank appears exactly once.
  if axes.len != rank:
    return false
  for m in 0 ..< rank:
    var count = 0
    for n in axes:
      if n == m:
        inc count
    if count != 1:
      return false
  true

func expandable(src: Span[int], target: openArray[int]): bool =
  ## Every target axis equals the source axis, or the source axis is 1:
  ## a size-1 axis stretches to any target size at stride 0.
  if src.len != target.len:
    return false
  for i in 0 ..< src.len:
    if src[i] != target[i] and src[i] != 1:
      return false
  true

proc tn_view*(engine: AnyEngine, x: TensorNode, shape, strides: openArray[int], offset: int): TensorNode {.contracts.} =
  ## Layout view of `x` over the same buffer cell: any rank, any strides.
  ##
  ## Args:
  ## - shape, strides: the view layout, in elements
  ## - offset: the view origin's element offset from the tensor's origin,
  ##   including the tensor's own offset
  ##
  ## The general case of narrow, transpose, permute, reshape and expand:
  ## one shared buffer, no allocation, no copy.
  ##
  ## A stride-0 axis is allowed here and on device: the serializer packs
  ## the static shape leaves, so the kernel reads the true layout.
  let ix = x.getImpl()
  {.precondition: shape.len == strides.len.}
  {.precondition: shape.len in 1 .. 5.}
  {.precondition: nonNegativeDims(shape).}
  {.precondition: inBackingCapacity(ix, shape.toSpan(), strides.toSpan(), offset).}
  ViewNode(
    src: x,
    v: engine.newViewImpl(ix, shape.toSpan(), strides.toSpan(), ix.offset() + offset)
  ).to(TensorNode)

proc tn_reshape*(engine: AnyEngine, x: TensorNode, shape: openArray[int]): TensorNode {.contracts.} =
  ## View of `x` with a new shape, same element order.
  ##
  ## Args:
  ## - shape: the new shape, its element count must match the source's
  ##
  ## Requires a dense contiguous source: the new layout is contiguous over
  ## the same buffer window, and a strided source would need a copy op.
  ## The source's offset is part of the new view, which allocates nothing.
  let ix = x.getImpl()
  let s = ix.shape()
  let st = ix.strides()
  {.precondition: shape.len in 1 .. 5.}
  {.precondition: nonNegativeDims(shape).}
  {.precondition: getNumElems(shape.toSpan()) == s.getNumElems().}
  {.precondition: denseContiguous(s, st).}
  var rt = newSeq[int](shape.len)
  var acc = 1
  for j in 1 .. shape.len:
    rt[shape.len - j] = acc
    acc *= shape[shape.len - j]
  ViewNode(
    src: x,
    v: engine.newViewImpl(ix, shape.toSpan(), rt.toSpan(), ix.offset())
  ).to(TensorNode)

proc tn_transpose*(engine: AnyEngine, x: TensorNode): TensorNode =
  ## View of `x` with every axis reversed, matching MLX transpose with no axes argument.
  ##
  ## Returns:
  ## - a view of `x` over the same buffer with the source's offset, no allocation, no copy
  let ix = x.getImpl()
  let s = ix.shape()
  let st = ix.strides()
  var rs = newSeq[int](s.len)
  var rt = newSeq[int](s.len)
  for i in 0 ..< s.len:
    rs[i] = s[s.len - 1 - i]
    rt[i] = st[st.len - 1 - i]
  ViewNode(
    src: x,
    v: engine.newViewImpl(ix, rs.toSpan(), rt.toSpan(), ix.offset())
  ).to(TensorNode)

proc tn_permute*(engine: AnyEngine, x: TensorNode, axes: openArray[int]): TensorNode {.contracts.} =
  ## View of `x` with the axes reordered: result axis i reads source axis axes[i].
  ##
  ## Args:
  ## - axes: the source axis each result axis reads, one per source axis
  ##
  ## Returns:
  ## - a view of `x` over the same buffer with the source's offset, no allocation, no copy
  let ix = x.getImpl()
  let s = ix.shape()
  let st = ix.strides()
  {.precondition: isPermutation(axes, s.len).}
  var rs = newSeq[int](axes.len)
  var rt = newSeq[int](axes.len)
  for i in 0 ..< axes.len:
    rs[i] = s[axes[i]]
    rt[i] = st[axes[i]]
  ViewNode(
    src: x,
    v: engine.newViewImpl(ix, rs.toSpan(), rt.toSpan(), ix.offset())
  ).to(TensorNode)

proc tn_narrow*(engine: AnyEngine, x: TensorNode, axis, start, width: int): TensorNode {.contracts.} =
  ## Window view of `x` with the other axes unchanged.
  ##
  ## Args:
  ## - axis: the narrowed dimension
  ## - start, width: the window inside that dimension
  ##
  ## The element offset shifts by start·stride[axis] down the narrowed axis,
  ## and a narrowed source keeps its offset under further views and splits.
  ## The view allocates nothing.
  let ix = x.getImpl()
  let s = ix.shape()
  let st = ix.strides()
  {.precondition: axis >= 0 and axis < s.len.}
  {.precondition: start >= 0 and width >= 0 and start + width <= s[axis].}
  var ns = newSeq[int](s.len)
  for i in 0 ..< s.len:
    ns[i] = s[i]
  ns[axis] = width
  ViewNode(
    src: x,
    v: engine.newViewImpl(ix, ns.toSpan(), st, ix.offset() + start * st[axis])
  ).to(TensorNode)

proc tn_expand*(engine: AnyEngine, x: TensorNode, shape: openArray[int]): TensorNode {.contracts.} =
  ## Broadcast view of `x`: a size-1 axis stretches to any target size at stride 0, and every other axis keeps its size and stride.
  ##
  ## Args:
  ## - shape: the target shape, one target axis per source axis
  ##
  ## A stride-0 axis reads correctly everywhere: on the host (item, toSeq),
  ## and ops consume the broadcast view per position on the device.
  let ix = x.getImpl()
  let s = ix.shape()
  let st = ix.strides()
  {.precondition: nonNegativeDims(shape).}
  {.precondition: expandable(s, shape).}
  var es = newSeq[int](s.len)
  for i in 0 ..< s.len:
    es[i] = if s[i] == 1: 0 else: st[i]
  ViewNode(
    src: x,
    v: engine.newViewImpl(ix, shape.toSpan(), es.toSpan(), ix.offset())
  ).to(TensorNode)

# #######################################################################
#  Graph - Internal
# #######################################################################

proc srcBufferCell[Mem](src: AnyTensorImpl): ref Mem =
  ## Shared `ref DeviceMem` cell of an erased impl, one branch per rank.
  ## The cell is nil until the impl's own eval materializes it.
  case src.shape().len
  of 1:
    src.to(TensorImpl[Mem, Layout[(int,), (int,)]]).data
  of 2:
    src.to(TensorImpl[Mem, Layout[(int, int), (int, int)]]).data
  of 3:
    src.to(TensorImpl[Mem, Layout[(int, int, int), (int, int, int)]]).data
  of 4:
    src.to(TensorImpl[Mem, Layout[(int, int, int, int), (int, int, int, int)]]).data
  of 5:
    src.to(TensorImpl[Mem, Layout[(int, int, int, int, int), (int, int, int, int, int)]]).data
  else:
    raise ValueError.newException(
      "srcBufferCell: rank " & $src.shape().len & " unsupported")

proc enqueueView[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: var TensorNode, visited: var VisitedNodes): bool =
  let asView = n.to(ViewNode)
  if asView.isNil:
    return false
  # source materialization comes first. The view then binds the source's buffer cell.
  engine.evalNode(batch, asView.src, visited)
  let rv = asView.v.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
  rv.data = srcBufferCell[Mem](asView.src.getImpl())
  true

# #######################################################################
#  Registration
# #######################################################################

registerOp("view", enqueueView)
