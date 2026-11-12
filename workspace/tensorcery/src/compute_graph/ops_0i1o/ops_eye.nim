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
import ../batching
import ../env


const FillFleet = (elemsPerGroup: 128)

proc k_eye(D: distinct TensorView, ncols, kDiag: float32) =
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
      let idxF = (base + box.crd2idx(e)).float32
      var rowF = 0.0'f32
      var colF = idxF
      while colF >= ncols:
        colF -= ncols
        rowF += 1.0'f32
      rD.data[e] = if colF - rowF == kDiag: 1.0'f32.to(D.T) else: zero(D.T)
    D.store(rD, thrID, tileID)


type
  EyeNode* = ref object
    r: AnyTensorImpl
    n, m, k: int


func getImpl*(n: EyeNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(n: EyeNode): string {.inline.} =
  "Eye"


proc tn_eye*(engine: AnyEngine, n, m, k: int, dtype: Dtype): TensorNode {.contracts.} =
  return EyeNode(r: engine.newPlaceholderNode(dtype, [n * m]).getImpl(), n: n, m: m, k: k).to(TensorNode)


proc enqueueEye[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T], rank: static int): bool =
  let asEye = n.to(EyeNode)
  if asEye.isNil():
    return false
  let rImpl = asEye.r.to(engine.getTensorImplType(T, rank))
  engine.preAlloc(rImpl)
  batch.enqueue<<(ceil_div(rImpl.size(), FillFleet.elemsPerGroup), LaneWidth)>> k_eye(rImpl.bridge(), asEye.m.float32, asEye.k.float32)
  return true


registerOp("Eye", enqueueEye)
