# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import workspace/ceramic
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
#  Device kernel
# ═══════════════════════════════════════════════════════════════════════

proc k_rmsnorm(dst, src, weight: distinct TensorBasicView, eps: float32, axis: int) =
  var reg: tile(float32)
  const
    W = getLaneWidth()
    owned = reg.size() div W
  let thrID = getLocalThreadID()
  var rows = 1
  for d in 0 ..< src.layout.rank:
    if d != axis:
      rows *= src.layout.shape[d]
  let gridStride = getGridSize()
  for g in countup(getLocalTileID(rows), rows - 1, gridStride):
    let
      srcRow = src.alongAxisAt(axis, g)
      rowSize = srcRow.size()
      tiled = make_layout(rowSize).logical_divide(reg.getLayout())
      stripTiles = tiled.getTileGrid().size()
    var dstRow = dst.alongAxisAt(axis, g)
    # pass 1: sum of squares, one register tile at a time
    var acc = zero(float32)
    for tileID in 0 ..< stripTiles:
      reg.load(srcRow, thrID, tileID, padWith = zero(float32))
      var tileSum = zero(float32)
      for i in 0 ..< owned:
        reg.data[i] = reg.data[i].square()
        tileSum += reg.data[i]
      acc += reduceAdd(tileSum)
    let scale = rsqrt(acc / float32(rowSize) + eps)
    # pass 2: scaled apply with the weight broadcast along the axis
    if weight.isNil():
      for r in countup(thrID, rowSize - 1, W):
        dstRow[r] = srcRow[r] * scale
    else:
      let weightStrip = weight.alongAxisAt(0, 0)
      for r in countup(thrID, rowSize - 1, W):
        dstRow[r] = srcRow[r] * scale * weightStrip[r]

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
      axis = rImpl.layout.rank - 1
      rows = rImpl.size() div rImpl.layout.shape[axis]
      grid = min(ceil_div(rows, Fleet.rowsPerGroup), Fleet.maxGroups)
    batch.enqueue<<(grid, LaneWidth)>> k_rmsnorm(rImpl.bridge(), xImpl.bridge(),
      asNorm.w.getImpl().to(engine.getTensorImplType(T)).bridge(), asNorm.eps, axis)
  else:
    raiseAssert("rmsnorm: dtype unsupported, the op is float32")
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("RmsNorm", collectRmsnorm, enqueueRmsnorm)
