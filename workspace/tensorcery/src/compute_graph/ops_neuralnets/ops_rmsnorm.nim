# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/ceramic
import workspace/ceramic/src/kernels/k_tile_rmsnorm
import workspace/crucible
import workspace/iface
import workspace/math
import workspace/design_by_contract
import ../../tensors_internals
import ../batching
import ../env

# ############################################################
#
#                   RmsNorm
#
# ############################################################

const Fleet = (rowsPerGroup: 4, maxGroups: 8192)

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  RmsNormNode* = ref object
    r: AnyTensorImpl
    x: TensorNode
    w: TensorNode         # nil when unweighted
    eps: float32

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: RmsNormNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(n: RmsNormNode): string {.inline.} =
  "RmsNorm"


# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_rmsnorm*(engine: AnyEngine, x: TensorNode, weight: TensorNode, eps: float32 = 1e-5): TensorNode {.contracts.} =
  let ix = x.getImpl()
  {.precondition: ix.shape().p.len >= 1.}
  if not weight.isNil():
    let iw = weight.getImpl()
    {.precondition: iw.shape().p.len == 1.}
    {.precondition: ix.dtype() == iw.dtype().}
    {.precondition: iw.shape().p[0] == ix.shape().p[ix.shape().p.len - 1].}
  return RmsNormNode(x: x, w: weight, eps: eps,
      r: engine.newPlaceholderNode(ix.dtype(), ix.shape().p).getImpl()
    ).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc collectRmsnorm[E: HwEngine, T](engine: E, n: var TensorNode, stack: var seq[WalkFrame], _: typedesc[T]): bool =
  let asNorm = n.to(RmsNormNode)
  if asNorm.isNil():
    return false
  engine.preAlloc(asNorm.r.to(engine.getTensorImplType(T)))
  stack.add WalkFrame(n: asNorm.x, kernelPending: false)
  if not asNorm.w.isNil():
    stack.add WalkFrame(n: asNorm.w, kernelPending: false)
  return true

proc enqueueRmsnorm[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T]): bool =
  let asNorm = n.to(RmsNormNode)
  if asNorm.isNil():
    return false
  when T is float32:
    let
      rImpl = asNorm.r.to(engine.getTensorImplType(T))
      xImpl = asNorm.x.getImpl().to(engine.getTensorImplType(T))
      rows = rImpl.size() div rImpl.layout.shape[rImpl.layout.rank - 1]
      grid = min(ceil_div(rows, Fleet.rowsPerGroup), Fleet.maxGroups)
    # Ceramic kernel signature: static rank and static reduce axis, one instantiation per rank. relayoutBridge checks the rank match.
    template enqueueRmsnormAt(lay: typedesc, RK: static int) =
      const axis = RK - 1
      let
        rView = rImpl.relayoutBridge(lay)
        xView = xImpl.relayoutBridge(lay)
      if asNorm.w.isNil():
        batch.enqueue<<(grid, LaneWidth)>> rmsnorm(rView, xView, asNorm.eps, axis)
      else:
        let wView = asNorm.w.getImpl().to(engine.getTensorImplType(T)).relayoutBridge(Layout[(int,), (int,)])
        batch.enqueue<<(grid, LaneWidth)>> rmsnorm(rView, xView, wView, asNorm.eps, axis)
    case rImpl.layout.rank
    of 2: enqueueRmsnormAt(Layout[(int, int), (int, int)], 2)
    of 3: enqueueRmsnormAt(Layout[(int, int, int), (int, int, int)], 3)
    of 4: enqueueRmsnormAt(Layout[(int, int, int, int), (int, int, int, int)], 4)
    of 5: enqueueRmsnormAt(Layout[(int, int, int, int, int), (int, int, int, int, int)], 5)
    else:
      raiseAssert("rmsnorm: rank " & $rImpl.layout.rank & " unsupported, the op is rank 2+ float32")
  else:
    raiseAssert("rmsnorm: dtype unsupported, the op is float32")
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("RmsNorm", collectRmsnorm, enqueueRmsnorm)
