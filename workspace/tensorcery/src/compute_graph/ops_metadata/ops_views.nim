# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/iface
import workspace/design_by_contract
import ../../tensors_internals

# ############################################################
#
#                     Views
#
# ############################################################

func sameAddressing(shape, strides: openArray[int]): bool =
  if strides[shape.len - 1] != 1:
    return false
  for i in 0 ..< shape.len - 1:
    if strides[i] != strides[i + 1] * shape[i + 1]:
      return false
  true

func reachWindow(shape, strides: openArray[int], base: int): tuple[lo, hi: int] =
  result = (base, base)
  for i in 0 ..< shape.len:
    let reach = if shape[i] > 0: strides[i] * (shape[i] - 1) else: 0
    if reach >= 0:
      result.hi += reach
    else:
      result.lo += reach

proc viewInBounds(shape, strides: openArray[int], relOffset: int, src: TensorNode): bool =
  var
    base = relOffset
    cur = src
    offsetsFolded = false
  while true:
    let impl = cur.getImpl()
    if not offsetsFolded:
      base += impl.offset()
      offsetsFolded = impl.isPreAllocated()
    let v = cur.to(ViewNode)
    if v.isNil():
      break
    cur = v.src
  let (lo, hi) = reachWindow(shape, strides, base)
  lo >= 0 and hi < cur.getImpl().size()

func broadcastable(src, target: openArray[int]): bool =
  if target.len < src.len:
    return false
  let lead = target.len - src.len
  for i in 0 ..< src.len:
    if target[lead + i] != src[i] and src[i] != 1:
      return false
  true

proc tn_view(engine: AnyEngine, x: TensorNode, shape, strides: openArray[int], offset: int): TensorNode {.contracts.} =
  {.precondition: shape.len == strides.len.}
  {.precondition: shape.len in 1 .. 5.}
  {.precondition: viewInBounds(shape, strides, offset, x).}
  engine.newViewNode(x, shape, strides, offset, "View")

proc tn_reshape*(engine: AnyEngine, x: TensorNode, shape: openArray[int]): TensorNode {.contracts.} =
  let ix = x.getImpl()
  {.precondition: shape.len in 1 .. 5.}
  var
    inferIdx = -1
    known = 1
  for i in 0 ..< shape.len:
    if shape[i] == -1:
      {.precondition: inferIdx < 0.}
      inferIdx = i
    else:
      known *= shape[i]
  var resolved = @shape
  if inferIdx >= 0:
    {.precondition: known > 0.}
    resolved[inferIdx] = ix.size() div known
  var total = 1
  for d in resolved:
    total *= d
  {.precondition: total == ix.size().}
  let s = @(ix.shape().p)
  {.precondition: sameAddressing(s, ix.strides().p).}
  var outStrides = newSeq[int](resolved.len)
  var acc = 1
  for j in 1 .. resolved.len:
    outStrides[resolved.len - j] = acc
    acc *= resolved[resolved.len - j]
  engine.newViewNode(x, resolved, outStrides, 0, "Reshape")

proc tn_transpose*(engine: AnyEngine, x: TensorNode, axes: openArray[int]): TensorNode {.contracts.} =
  let ix = x.getImpl()
  let s = @(ix.shape().p)
  let st = @(ix.strides().p)
  {.precondition: axes.len == s.len.}
  for i in 0 ..< axes.len:
    {.precondition: axes[i] >= 0 and axes[i] < s.len.}
    for j in 0 ..< i:
      {.precondition: axes[i] != axes[j].}
  var outShape = newSeq[int](axes.len)
  var outStrides = newSeq[int](axes.len)
  for i in 0 ..< axes.len:
    outShape[i] = s[axes[i]]
    outStrides[i] = st[axes[i]]
  engine.newViewNode(x, outShape, outStrides, 0, "Transpose")

proc tn_transpose*(engine: AnyEngine, x: TensorNode): TensorNode =
  let ix = x.getImpl()
  let s = @(ix.shape().p)
  let st = @(ix.strides().p)
  var outShape = newSeq[int](s.len)
  var outStrides = newSeq[int](s.len)
  for i in 0 ..< s.len:
    outShape[i] = s[s.len - 1 - i]
    outStrides[i] = st[s.len - 1 - i]
  engine.newViewNode(x, outShape, outStrides, 0, "Transpose")

proc tn_broadcastTo*(engine: AnyEngine, x: TensorNode, shape: openArray[int]): TensorNode {.contracts.} =
  let ix = x.getImpl()
  let s = @(ix.shape().p)
  {.precondition: shape.len >= s.len and shape.len <= 5.}
  for i in 0 ..< shape.len:
    {.precondition: shape[i] >= 0.}
  {.precondition: broadcastable(s, shape).}
  let st = @(ix.strides().p)
  let lead = shape.len - s.len
  var outStrides = newSeq[int](shape.len)
  for i in 0 ..< lead:
    outStrides[i] = 0
  for i in 0 ..< s.len:
    outStrides[lead + i] = if s[i] == 1: 0 else: st[i]
  engine.newViewNode(x, shape, outStrides, 0, "Broadcast_to")

proc tn_squeeze*(engine: AnyEngine, x: TensorNode, axis: int): TensorNode {.contracts.} =
  let ix = x.getImpl()
  let s = @(ix.shape().p)
  let st = @(ix.strides().p)
  {.precondition: axis >= 0 and axis < s.len.}
  {.precondition: s[axis] == 1.}
  {.precondition: s.len > 1.}
  var outShape = newSeq[int](s.len - 1)
  var outStrides = newSeq[int](s.len - 1)
  for i in 0 ..< s.len:
    if i < axis:
      outShape[i] = s[i]
      outStrides[i] = st[i]
    elif i > axis:
      outShape[i - 1] = s[i]
      outStrides[i - 1] = st[i]
  engine.newViewNode(x, outShape, outStrides, 0, "Squeeze")

proc tn_expandDims*(engine: AnyEngine, x: TensorNode, axis: int): TensorNode {.contracts.} =
  let ix = x.getImpl()
  let s = @(ix.shape().p)
  let st = @(ix.strides().p)
  {.precondition: axis >= 0 and axis <= s.len.}
  {.precondition: s.len < 5.}
  var outShape = newSeq[int](s.len + 1)
  var outStrides = newSeq[int](s.len + 1)
  var trailing = 1
  for i in axis ..< s.len:
    trailing *= s[i]
  for i in 0 ..< axis:
    outShape[i] = s[i]
    outStrides[i] = st[i]
  outShape[axis] = 1
  outStrides[axis] = trailing
  for i in axis ..< s.len:
    outShape[i + 1] = s[i]
    outStrides[i + 1] = st[i]
  engine.newViewNode(x, outShape, outStrides, 0, "Expand_dims")

proc tn_narrow*(engine: AnyEngine, x: TensorNode, axis, start, width: int): TensorNode {.contracts.} =
  let ix = x.getImpl()
  let s = @(ix.shape().p)
  let st = @(ix.strides().p)
  {.precondition: axis >= 0 and axis < s.len.}
  {.precondition: start >= 0 and width >= 0 and start + width <= s[axis].}
  var outShape = @(s)
  outShape[axis] = width
  let relOffset = start * st[axis]
  {.precondition: viewInBounds(outShape, st, relOffset, x).}
  engine.newViewNode(x, outShape, st, relOffset, "Narrow")
