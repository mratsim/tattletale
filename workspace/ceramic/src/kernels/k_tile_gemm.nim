# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tiled matmul on the tile algebra, one squadron per (MT, NT) tile.
##
## - `gemm`: D = alpha·A·B + beta·C, the entry, reforwards on the atom family
## - `gemm_via_mma`: the threadmap family, the atom's (T, V) layouts carry
##   per-lane register tiles (mma.sync, Apple simdgroup, universal FMA)
## - `gemm_via_khr_coopmat`: the opaque-fragment family, per-subgroup
##   `coopmat` objects through the KHR cooperative-matrix builtins
## - beta = 0 skips the C read

import workspace/crucible
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_layout_gemm
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu

# ═══════════════════════════════════════════════════════════════════════
#  Tile-level matmul, the entry
# ═══════════════════════════════════════════════════════════════════════

proc gemm*(mma: static TiledMma, D: distinct TensorView, alpha: float32, A, B: distinct TensorView, beta: float32, C: distinct TensorView) =
  ## D = alpha·A·B + beta·C.
  ##
  ## Args:
  ##   - mma: the tile schedule, one squadron covers an (MT, NT) D tile
  ##   - D: destination, receives alpha·A·B + beta·C
  ##   - A, B, C: operand tensor views
  ##   - beta = 0 skips the C read, D may hold uninitialized memory
  when mma.atom.getFamily() == "coopmat_khr":
    gemm_via_khr_coopmat(mma, D, alpha, A, B, beta, C)
  else:
    gemm_via_mma(mma, D, alpha, A, B, beta, C)

proc matmul*(mma: static TiledMma, D, A, B: distinct TensorView) =
  ## D = A·B, same tile grid and launch mapping as gemm.
  gemm(mma, D, 1.0'f32, A, B, 0.0'f32, D)

# ═══════════════════════════════════════════════════════════════════════
#  Threadmap family: per-lane register tiles
# ═══════════════════════════════════════════════════════════════════════

proc gemm_via_mma*(mma: static TiledMma, D: distinct TensorView, alpha: float32, A, B: distinct TensorView, beta: float32, C: distinct TensorView) =
  ## Per-lane operand register tiles load through the atom's thread->value map.
  ##
  ## - ragged edge tiles go through paddedLoad / guardedStore
  ## - interior tiles take the unchecked flat path
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

# ═══════════════════════════════════════════════════════════════════════
#  Opaque-fragment family: Vulkan KHR cooperative matrices
# ═══════════════════════════════════════════════════════════════════════

func fragData*[T, Sh, St](t: TensorView[T, Sh, St]): ptr UncheckedArray[T] {.inline.} =
  ## Raw data pointer of a tensor view.
  t.data

func fragStride*[T, Sh, St](t: TensorView[T, Sh, St], dim: static int): int {.inline.} =
  ## Per-dimension stride as a plain runtime int.
  when typeof(t.layout.stride[dim]) is Int:
    coopFold(t.layout.stride[dim])
  else:
    t.layout.stride[dim]

func coopFold*[V: static int](x: Int[V]): int {.inline.} =
  ## Compile-time Int folded to its value, the layout algebra's static
  ## dimensions carry their value in the type parameter.
  V

# ═══════════════════════════════════════════════════════════════════════
#  KHR cooperative-matrix tier (Vulkan)
# ═══════════════════════════════════════════════════════════════════════

template coopDimStride*(t: TensorView, dim: static int): int =
  ## Per-dimension stride as a kernel value.
  ##
  ## - literal `make_layout` strides fold to compile-time constants
  ## - dynamic strides read the pushed layout at runtime
  when typeof(t.layout.stride[dim]) is Int:
    coopFold(t.layout.stride[dim])
  else:
    t.fragStride(dim)

template coopFragmentStride*(t: TensorView, rowDim, colDim: static int): int =
  ## `coopMatLoad`/`coopMatStore` stride value for a fragment whose
  ## (row, col) indices walk the tensor's (rowDim, colDim).
  ##
  ## - RowMajor strides by the row dim, ColumnMajor by the col dim
  ## - statically-typed strides fold to constants, dynamic strides read
  ##   the pushed layout
  when typeof(t.layout.stride[colDim]) is Int[1]:
    coopDimStride(t, rowDim)
  elif typeof(t.layout.stride[rowDim]) is Int[1]:
    coopDimStride(t, colDim)
  else:
    {.error: "KHR coopmat tier: one fragment dimension must be statically contiguous (a statically-1 stride, make_layout with literal strides)".}
    0

template coopFragmentLayout*(t: TensorView, rowDim, colDim: static int): int =
  ## `coopMatLoad`/`coopMatStore` layout constant for a fragment whose
  ## (row, col) indices walk the tensor's (rowDim, colDim).
  ##
  ## Args:
  ##   - RowMajor addresses `buf[element + row*stride + col]`, the col dim
  ##     is the contiguous one
  ##   - the KHR SPIR-V requires the layout operand to be a compile-time
  ##     constant
  ##   - the orientation is decided from the statically-typed stride types
  when typeof(t.layout.stride[colDim]) is Int[1]:
    cmLayoutRowMajor
  elif typeof(t.layout.stride[rowDim]) is Int[1]:
    cmLayoutColumnMajor
  else:
    {.error: "KHR coopmat tier: one fragment dimension must be statically contiguous (a statically-1 stride, make_layout with literal strides)".}
    0

proc gemm_via_khr_coopmat*(mma: static TiledMma, D: distinct TensorView, alpha: float32, A, B: distinct TensorView, beta: float32, C: distinct TensorView) =
  ## Opaque-fragment family.
  ## One fragment is a per-subgroup `coopmat<T, gl_ScopeSubgroup, M, N, Use>` object.
  ##
  ## Mechanics:
  ##
  ## - the component distribution is driver-defined
  ## - no threadmap, no staging, no packing
  ## - the fragment acquisition is `coopMatLoad` at the tile origin
  ## - the only arithmetic is `coopMatMulAdd`
  ##
  ## Launch:
  ##
  ## - one squadron per workgroup, `blk = (subgroupSize, 1)`
  ## - the tile grid is iterated by workgroup ID with a grid-stride loop
  ## - every coopmat operand is uniform within the subgroup
  ##
  ## Layout contract:
  ##
  ## - the fragment load/store layout operand must be a compile-time constant
  ##   the KHR SPIR-V admits only constant instructions
  ## - the layout orientation comes from the statically-typed strides:
  ##   a contiguous dimension carries a statically-1 stride
  ##   (`make_layout` with literal strides)
  ##   a strided dimension may stay dynamic
  ## - the B fragment is (K, N) over (N, K) storage, its orientation
  ##   flips accordingly
  ## - the alpha/beta epilogue goes through the fragment operators
  ##   (`coopMatScale`, `coopMatAdd`)

  const
    am = mma.atom.getM()
    an = mma.atom.getN()
    ak = mma.atom.getK()
    fragsM = mma.MT div am
    fragsN = mma.NT div an
    kFrags = mma.KT div ak

  # per-operand strides for the element addressing, constants when static
  let
    aS0 = coopDimStride(A, 0)
    aS1 = coopDimStride(A, 1)
    bS0 = coopDimStride(B, 0)
    bS1 = coopDimStride(B, 1)
    dS0 = coopDimStride(D, 0)
    dS1 = coopDimStride(D, 1)
    cS0 = coopDimStride(C, 0)
    cS1 = coopDimStride(C, 1)
  # the fragment is (rows, cols): A (m, k), B (k, n) over (n, k) storage,
  # D (m, n). The layout constants fold from the stride types
  const aLoadLayout = coopFragmentLayout(A, 0, 1)
  const bLoadLayout = coopFragmentLayout(B, 1, 0)
  const dStoreLayout = coopFragmentLayout(D, 0, 1)
  let
    aLoadStride = coopFragmentStride(A, 0, 1)
    bLoadStride = coopFragmentStride(B, 1, 0)
    dStoreStride = coopFragmentStride(D, 0, 1)

  let cGeom = D.tileWith((mma.MT, mma.NT))
  let aGeom = A.tileWith((mma.MT, mma.KT))

  let cGrid = cGeom.getTileGrid()
  let (_, kTiles) = aGeom.getTileGrid()
  let
    cSize = coopFold(cGrid.size())
    gridStride = getGridSize()

  # one squadron per workgroup, the tile ID is the workgroup ID:
  # uniform across the whole squadron, as every coopmat operand must be
  for tileID in countup(cGrid.getLocalTileID(), cSize - 1, gridStride):
    let (tm, tn) = idx2crd(cGrid, tileID)
    var rD: array[fragsM * fragsN, coopmatAcc[D.T, am, an]]
    block:
      var rDinit: coopmatAcc[D.T, am, an]
      rDinit = coopMatFill(am, an, zero(D.T))
      for i in 0 ..< fragsM * fragsN:
        rD[i] = rDinit

    for k in 0 ..< kTiles:
      for kf in 0 ..< kFrags:
        let kCol = k * mma.KT + kf * ak
        for fm in 0 ..< fragsM:
          let
            row = tm * mma.MT + fm * am
            element = row * aS0 + kCol * aS1
          var matA: coopmatA[A.T, am, ak]
          matA = coopMatLoadA(am, ak, A.data, element, aLoadStride, aLoadLayout)
          for fn in 0 ..< fragsN:
            let
              brow = tn * mma.NT + fn * an
              belement = brow * bS0 + kCol * bS1
            var matBload: coopmatB[B.T, ak, an]
            matBload = coopMatLoadB(ak, an, B.data, belement, bLoadStride, bLoadLayout)
            rD[fm * fragsN + fn] = coopMatMulAdd(matA, matBload, rD[fm * fragsN + fn])

    if alpha != 1.0'f32:
      for i in 0 ..< fragsM * fragsN:
        rD[i] = coopMatScale(rD[i], alpha)

    if beta != 0.0'f32:
      for fm in 0 ..< fragsM:
        for fn in 0 ..< fragsN:
          let
            row = tm * mma.MT + fm * am
            col = tn * mma.NT + fn * an
            element = row * dS0 + col * dS1
          var rC: coopmatAcc[D.T, am, an]
          rC = coopMatLoadAcc(am, an, C.data, element, dStoreStride, dStoreLayout)
          rC = coopMatScale(rC, beta)
          rD[fm * fragsN + fn] = coopMatAdd(rD[fm * fragsN + fn], rC)

    for fm in 0 ..< fragsM:
      for fn in 0 ..< fragsN:
        let
          row = tm * mma.MT + fm * am
          col = tn * mma.NT + fn * an
          element = row * dS0 + col * dS1
        coopMatStore(rD[fm * fragsN + fn], D.data, element, dStoreStride, dStoreLayout)
