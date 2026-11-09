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
    r*: seq[AnyTensorImpl]
    a*: TensorNode
    axis*: int

func getOutput*(n: SplitCore, i: int): AnyTensorImpl =
  n.r[i]

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

proc getImpl*[C](n: PosNode[C]): AnyTensorImpl =
  n.core.getOutput(n.pos)

func opName*(n: PosNode[SplitCore]): string =
  "split"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_split*(engine: AnyEngine, x: TensorNode, axis: int, k: int): seq[TensorNode] =
  let ix = x.getImpl()
  let s = ix.shape()
  # erased impls carry contiguous layouts only, a split on an inner
  # axis needs the inherited strides of the source layout
  precondition: (axis == 0, "split: axis " & $axis & " unsupported, only the outer axis splits densely")
  precondition: s[axis] mod k == 0
  var partShape = newSeq[int](s.len)
  for i in 0 ..< s.len:
    partShape[i] = s[i]
  partShape[axis] = s[axis] div k
  let core = SplitCore(a: x, axis: axis)
  for i in 0 ..< k:
    core.r.add(engine.newTensorNode(ix.dtype(), partShape.toSpan()).getImpl())
  for i in 0 ..< k:
    result.add(PosNode[SplitCore](core: core, pos: i).to(TensorNode))

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueSplit*[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: TensorNode): bool =
  let asSplit = n.to(PosNode[SplitCore])
  if asSplit.isNil:
    return false
  let core = asSplit.core
  engine.evalNode(batch, core.a)
  let src = core.a.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
  for i in 0 ..< core.r.len:
    let v = core.r[i].to(TensorImpl[Mem, Layout[Sh, Sh]])
    v.data = src.data
  true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("split", enqueueSplit)

