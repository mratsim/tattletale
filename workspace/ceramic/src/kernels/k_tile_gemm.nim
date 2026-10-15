# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tiled matmul on the tile algebra, one squadron per (MT, NT) tile.
##
## - `matmul`: D = A·B
## - `gemm`: D = alpha·A·B + beta·C, beta = 0 skips the C read
## - per-lane operand register tiles load through the atom's thread->value map
## - ragged edge tiles go through guardedLoad / guardedStore, interior tiles take the unchecked flat path

import workspace/crucible
import workspace/ceramic/src/tensors
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layouts
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/hardware/h_mma_dispatch
import workspace/ceramic/src/kernels/k_layout_gemm

# ═══════════════════════════════════════════════════════════════
#  Launch mapping
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  Tile-level matmul
# ═══════════════════════════════════════════════════════════════

proc gemm*(mma: static TiledMma, D: distinct TensorView, alpha: float32, A, B: distinct TensorView, beta: float32, C: distinct TensorView) =
  ## D = alpha·A·B + beta·C.
  ## The tile grid is cdiv(D.shape, (MT, NT)), excess squadrons exit,
  ## the tile comes from getLocalTileID, the lane from getLocalThreadID.
  ##
  ## Args:
  ##   - mma: the tile schedule, one squadron covers an (MT, NT) D tile
  ##   - D: destination, receives alpha·A·B + beta·C
  ##   - A, B, C: operand tensor views
  ##   - beta = 0 skips the C read, D may hold uninitialized memory
  const tvmA = mma.getTVCrdLayout(opA)
  const tvmB = mma.getTVCrdLayout(opB)
  const tvmC = mma.getTVCrdLayout(opC)
  const cLayout = mma.getMmaLayout(opC)

  let cGeom = D.tileWith((mma.MT, mma.NT))
  let aGeom = A.tileWith((mma.MT, mma.KT))

  let cGrid = cGeom.getTileGrid()
  let (_, kTiles) = aGeom.getTileGrid()
  let tileID = cGrid.getLocalTileID()
  let thrID = mma.getLocalThreadID()

  if tileID < cGrid.size() and thrID < mma.getThreadCount():
    let (m, n) = idx2crd(cGrid, tileID)
    var rD: rt(D.T, cLayout)
    var rA: rt(A.T, mma.getMmaLayout(opA))
    var rB: rt(B.T, mma.getMmaLayout(opB))
    for i in 0 ..< cLayout.size():
      rD[i] = zero(D.T)
    for k in 0 ..< kTiles:
      rA.loadTile(A, tvmA, thrID, (m, k))
      rB.loadTile(B, tvmB, thrID, (n, k))
      mma.atom.gemm_tile(rD, rA, rB, rD)

    if alpha != 1.0'f32:
      for i in 0 ..< cLayout.size():
        rD[i] = alpha * rD[i]

    if beta != 0.0'f32:
      var rC: rt(C.T, cLayout)
      rC.loadTile(C, tvmC, thrID, (m, n))
      for i in 0 ..< cLayout.size():
        rD[i] += beta * rC[i]
    D.storeTile(rD, tvmC, thrID, (m, n))

proc matmul*(mma: static TiledMma, D, A, B: distinct TensorView) =
  ## D = A·B, same tile grid and launch mapping as gemm.
  gemm(mma, D, 1.0'f32, A, B, 0.0'f32, D)
