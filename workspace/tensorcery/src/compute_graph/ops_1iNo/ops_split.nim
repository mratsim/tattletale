# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/crucible
import workspace/ceramic
import workspace/iface
import ../../tensors_internals
import ../dispatch
import workspace/design_by_contract

# ############################################################
#
#                   Split
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  SplitCore* = ref object
    r: seq[TensorNode]    # placeholder parts, bound at finalization
    a: TensorNode
    axis: int

proc getOutput(n: SplitCore, i: int): AnyTensorImpl =
  n.r[i].getImpl()

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*[C](n: PosNode[C]): AnyTensorImpl =
  n.core.getOutput(n.pos)

func opName*(n: PosNode[SplitCore]): string =
  "Split"

func size*(n: PosNode[SplitCore]): int =
  n.core.r[n.pos].size()

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_split*(engine: AnyEngine, x: TensorNode, axis: int, k: int): seq[TensorNode] {.contracts.} =
  let ix = x.getImpl()
  let s = ix.shape()
  {.precondition: axis >= 0 and axis < s.len.}
  {.precondition: s[axis] mod k == 0.}

  var partShape = newSeq[int](s.len)
  for i in 0 ..< s.len:
    partShape[i] = s[i]
  partShape[axis] = s[axis] div k
  let core = SplitCore(a: x, axis: axis)
  for i in 0 ..< k:
    core.r.add(engine.newPlaceholderNode(ix.dtype(), partShape.toSpan()))
  for i in 0 ..< k:
    result.add(PosNode[SplitCore](core: core, pos: i).to(TensorNode))

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueSplit[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T], rank: static int): bool =
  let asSplit = n.to(PosNode[SplitCore])
  if asSplit.isNil:
    return false
  let core = asSplit.core
  engine.evalNode(batch, core.a)
  let src = core.a.getImpl().to(engine.getTensorImplType(T, rank))
  let
    axis = core.axis
    partLen = src.layout.shape.atAxis(axis) div core.r.len
    partStride = src.layout.stride.atAxis(axis)
  for i in 0 ..< core.r.len:
    let v = core.r[i].getImpl().to(engine.getTensorImplType(T, rank))
    v.data = src.data
    v.offset = src.offset + i * partLen * partStride
    v.layout = make_layout(src.layout.shape.withAxis(axis, partLen), src.layout.stride)
  n = core.r[asSplit.pos]
  true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("Split", enqueueSplit)
