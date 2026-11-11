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

proc enqueueRmsnorm[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T], rank: static int): bool =
  let asNorm = n.to(RmsNormNode)
  if asNorm.isNil():
    return false
  engine.enqueueNode(batch, asNorm.x)

  let
    rImpl = asNorm.r.to(engine.getTensorImplType(T, rank))
    xImpl = asNorm.x.getImpl().to(engine.getTensorImplType(T, rank))
  when rank >= 2 and T is float32:
    let rows = rImpl.size() div rImpl.shape[rank - 1]
    engine.materialize(rImpl)
    if asNorm.w.isNil():
      batch.enqueue<<(min(ceil_div(rows, Fleet.rowsPerGroup), Fleet.maxGroups), LaneWidth)>> rmsnorm(rImpl.bridge(), xImpl.bridge(), asNorm.eps, rank - 1)
    else:
      engine.enqueueNode(batch, asNorm.w)
      let wImpl = asNorm.w.getImpl().to(engine.getTensorImplType(T, 1))
      batch.enqueue<<(min(ceil_div(rows, Fleet.rowsPerGroup), Fleet.maxGroups), LaneWidth)>> rmsnorm(rImpl.bridge(), xImpl.bridge(), wImpl.bridge(), asNorm.eps, rank - 1)
  else:
    raiseAssert("rmsnorm: rank " & $rank & " or dtype unsupported, the op is rank 2+ float32")
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("RmsNorm", enqueueRmsnorm)
