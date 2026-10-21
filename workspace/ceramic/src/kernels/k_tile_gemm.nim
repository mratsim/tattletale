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
## - ragged edge tiles go through paddedLoad / guardedStore, interior tiles take the unchecked flat path

import workspace/ceramic/src/tile_algebra

# ═══════════════════════════════════════════════════════════════
#  Tile-level matmul
# ═══════════════════════════════════════════════════════════════

proc gemm*(mma: static TiledMma, D: distinct TensorView, alpha: float32, A, B: distinct TensorView, beta: float32, C: distinct TensorView) =
  ## D = alpha·A·B + beta·C.
  ##
  ## Args:
  ##   - mma: the tile schedule, one squadron covers an (MT, NT) D tile
  ##   - D: destination, receives alpha·A·B + beta·C
  ##   - A, B, C: operand tensor views
  ##   - beta = 0 skips the C read, D may hold uninitialized memory
  const tvmA = mma.getThreadValueMap(opA)
  const tvmB = mma.getThreadValueMap(opB)
  const tvmC = mma.getThreadValueMap(opC)
  const cLayout = mma.getMmaLayout(opC)

  let cGeom = D.tileWith((mma.MT, mma.NT))
  let aGeom = A.tileWith((mma.MT, mma.KT))

  let cGrid = cGeom.getTileGrid()
  let (_, kTiles) = aGeom.getTileGrid()
  let thrID = getLocalThreadID()

  if thrID < getLaneWidth():
    let
      cSize = cGrid.size()
      gridStride = getGridSize()
    for tileID in countup(cGrid.getLocalTileID(), cSize - 1, gridStride):
      let (m, n) = idx2crd(cGrid, tileID)
      var rD: tile(D.T, cLayout)
      var rA: tile(A.T, mma.getMmaLayout(opA))
      var rB: tile(B.T, mma.getMmaLayout(opB))
      rD.fillWith(zero(D.T))
      for k in 0 ..< kTiles:
        rA.load(A, tvmA, thrID, (m, k))
        rB.load(B, tvmB, thrID, (n, k))
        mma.atom.gemm_tile(rD, rA, rB, rD)

      if alpha != 1.0'f32:
        for i in 0 ..< cLayout.size():
          rD[i] = alpha * rD[i]

      if beta != 0.0'f32:
        let rC = tile(C.T, cLayout).load(C, tvmC, thrID, (m, n))
        for i in 0 ..< cLayout.size():
          rD[i] += beta * rC[i]
      D.store(rD, tvmC, thrID, (m, n))

proc matmul*(mma: static TiledMma, D, A, B: distinct TensorView) =
  ## D = A·B, same tile grid and launch mapping as gemm.
  gemm(mma, D, 1.0'f32, A, B, 0.0'f32, D)
