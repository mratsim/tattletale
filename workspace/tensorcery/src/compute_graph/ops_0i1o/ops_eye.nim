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
#                   Eye
#
# ############################################################

const FillFleet = (elemsPerGroup: 128)

proc k_eye(D: distinct TensorBasicView, ncols, kDiag: float32) =
  # int32 kernel args lose stores through the enqueue path, so ncols and k are float32 and convert to int32 here
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
      let row = idx div ncols.int32
      let col = idx mod ncols.int32
      rD.data[e] = if col - row == kDiag.int32: 1.0'f32.to(D.T) else: zero(D.T)
    D.store(rD, thrID, tileID)

# ═══════════════════════════════════════════════════════════════════════
#  Defunctionalization
# ═══════════════════════════════════════════════════════════════════════

type
  EyeNode* = ref object
    r: AnyTensorImpl
    n, m, k: int

# ═══════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════

func getImpl*(n: EyeNode): AnyTensorImpl {.inline.} =
  n.r

func opName*(n: EyeNode): string {.inline.} =
  "Eye"

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Public
# ═══════════════════════════════════════════════════════════════════════

proc tn_eye*(engine: AnyEngine, n, m, k: int, dtype: Dtype): TensorNode {.contracts.} =
  # size 0 is refused at allocation, not here
  {.precondition: n >= 0 and m >= 0.}
  return EyeNode(r: engine.newPlaceholderNode(dtype, [n * m]).getImpl(), n: n, m: m, k: k).to(TensorNode)

# ═══════════════════════════════════════════════════════════════════════
#  Graph - Internal
# ═══════════════════════════════════════════════════════════════════════

proc collectEye[E: HwEngine, T](engine: E, n: var TensorNode, stack: var seq[WalkFrame], _: typedesc[T]): bool =
  let asEye = n.to(EyeNode)
  if asEye.isNil():
    return false
  engine.preAlloc(asEye.r.to(engine.getTensorImplType(T)))
  return true

proc enqueueEye[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T]): bool =
  let asEye = n.to(EyeNode)
  if asEye.isNil():
    return false
  let rImpl = asEye.r.to(engine.getTensorImplType(T))
  batch.enqueue<<(ceil_div(rImpl.size(), FillFleet.elemsPerGroup), LaneWidth)>> k_eye(rImpl.bridge(), asEye.m.float32, asEye.k.float32)
  return true

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

registerOp("Eye", collectEye, enqueueEye)
