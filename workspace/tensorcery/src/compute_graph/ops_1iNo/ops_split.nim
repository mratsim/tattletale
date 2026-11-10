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
    r: seq[TensorNode]    # part view shells, metadata-complete at the face
    a: TensorNode

proc getOutput(n: SplitCore, i: int): AnyTensorImpl =
  n.r[i].getImpl()

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

proc tn_split*(engine: AnyEngine, x: TensorNode, axis: int, k: int): seq[TensorNode] {.contracts.} =
  let ix = x.getImpl()
  let s = ix.shape()
  {.precondition: axis >= 0 and axis < s.len.}
  {.precondition: s[axis] mod k == 0.}

  # parts are shared-buffer views over the source impl:
  # - layout: the source layout narrowed along the split axis
  # - offset: part i starts i part-lengths down the split axis,
  #   from the source tensor's own origin
  var partShape = newSeq[int](s.len)
  for i in 0 ..< s.len:
    partShape[i] = s[i]
  partShape[axis] = s[axis] div k
  let st = ix.strides()
  let core = SplitCore(a: x)
  for i in 0 ..< k:
    let offset = ix.offset() + i * partShape[axis] * st[axis]
    core.r.add(engine.newViewImpl(ix, partShape.toSpan(), st, offset))
  for i in 0 ..< k:
    result.add(PosNode[SplitCore](core: core, pos: i).to(TensorNode))

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueSplit[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: var TensorNode, visited: var VisitedNodes): bool =
  let asSplit = n.to(PosNode[SplitCore])
  if asSplit.isNil:
    return false
  let core = asSplit.core
  engine.evalNode(batch, core.a, visited)
  # part layouts, offsets and capacity were final at the face, so only the shared
  # buffer cell binds here, one cell for every part
  let src = core.a.getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]])
  for i in 0 ..< core.r.len:
    core.r[i].getImpl().to(TensorImpl[Mem, Layout[Sh, Sh]]).data = src.data
  n = core.r[asSplit.pos]
  true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("split", enqueueSplit)
