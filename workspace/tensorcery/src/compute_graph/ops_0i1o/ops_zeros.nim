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
import ../../tensors_internals
import ../eval
import ../env

# ############################################################
#
#                   Zeros (fill kernel)
#
# ############################################################

const FillFleet* = (elemsPerGroup: 128)

proc k_fill*(D: distinct TensorBasicView, val: float32) =
  let thrID = getLocalThreadID()
  var reg: tile(D.T)
  let tiled = make_layout(D.size()).logical_divide(reg.getLayout())
  let grid = tiled.getTileGrid()
  let
    cSize = grid.size()
    gridStride = getGridSize()
  for tileID in countup(getLocalTileID(grid), cSize - 1, gridStride):
    var rD: tile(D.T)
    for i in 0 ..< rD.size():
      rD[i] = val.to(D.T)
    D.store(rD, thrID, tileID)

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  ZerosNode* = ref object of TensorNode
    r: AnyTensorImpl

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: ZerosNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(_: typedesc[ZerosNode]): static string {.inline.} =
  "Zeros"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_zeros*(engine: AnyEngine, dtype: Dtype, shape: openArray[int]): TensorGraph {.contracts.} =
  {.precondition: shape.len in 1 .. 5.}
  return ZerosNode(r: engine.newPlaceholderNode(dtype, shape).getImpl()).to(TensorGraph)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc enqueueZeros[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorGraph, _: typedesc[T]): bool =
  let asZeros = n.to(ZerosNode)
  if asZeros.isNil():
    return false
  let rImpl = asZeros.r.to(engine.getTensorImplType(T))
  batch.enqueue<<(ceil_div(rImpl.size(), FillFleet.elemsPerGroup), LaneWidth)>> k_fill(rImpl.bridge(), 0.0'f32)
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp(ZerosNode, enqueueZeros, opName(ZerosNode))
