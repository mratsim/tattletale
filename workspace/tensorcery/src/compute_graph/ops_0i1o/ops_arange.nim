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
import workspace/design_by_contract
from std/math import ceil
import ../../tensors_internals
import ../batching
import ../env

# ############################################################
#
#                   Arange
#
# ############################################################

const FillFleet = (elemsPerGroup: 128)

proc k_arange(D: distinct TensorView, start, step: float32) =
  ## Writes x[i] = start + step*i over the flat element index i.
  ##
  ## `D` is a fresh dense allocation, so the flat index is the element index.
  let thrID = getLocalThreadID()
  var reg: tile(D.T)
  let tiled = make_layout(D.size()).logical_divide(reg.getLayout())
  let grid = tiled.getTileGrid()
  let
    cSize = grid.size()
    gridStride = getGridSize()
  for tileID in countup(grid.getLocalTileID(), cSize - 1, gridStride):
    let
      sel = ((thrID, _), tileID)
      base = tiled.crd2idx(sel)
      box = tiled.slice(sel)
    var rD: tile(D.T)
    for e in 0 ..< box.size():
      let idx = base + box.crd2idx(e)
      rD.data[e] = (start + step * idx.float32).to(D.T)
    D.store(rD, thrID, tileID)

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  ArangeNode* = ref object
    r: AnyTensorImpl
    start, step: float32

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: ArangeNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(n: ArangeNode): string {.inline.} =
  "Arange"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_arange*(engine: AnyEngine, start, stop, step: float32, dtype: Dtype): TensorNode {.contracts.} =
  ## A rank-1 tensor of ceil((stop - start) / step) elements,
  ## element i = start + step*i, materialized on eval.
  {.precondition: step != 0.0'f32.}
  {.precondition: (stop - start) * step > 0.0'f32.}
  let n = ceil((stop - start) / step).int
  return ArangeNode(r: engine.newPlaceholderNode(dtype, [n]).getImpl(), start: start, step: step).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueArange[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T], rank: static int): bool =
  let asArange = n.to(ArangeNode)
  if asArange.isNil():
    return false
  let rImpl = asArange.r.to(engine.getTensorImplType(T, rank))
  engine.materialize(rImpl)
  batch.enqueue<<(ceil_div(rImpl.size(), FillFleet.elemsPerGroup), LaneWidth)>> k_arange(rImpl.bridge(), asArange.start, asArange.step)
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("Arange", enqueueArange)
