## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Annotated BLIS-style GEMM using CuTe Layout Algebra
##
## Follows the BLIS 5-loop structure (Van Zee & van de Geijn, IPDPS 2014):
##
##   Loop 5 (outermost): jc — column panels of C/B              (step nc)
##   Loop 4:             pc — rank-k updates over K              (step kc)
##   Loop 3:             ic — row blocks of A                    (step mc)
##   Loop 2:             jr — micro-panels of B in L1 cache      (step nr)
##   Loop 1 (innermost): ir — micro-tiles of A in registers      (step mr)
##
##                ┌──── C[mr, nr] lives in registers
##                │     ┌── B[kc, nr] lives in L1 cache
##                │     │     ┌── A[mc, kc] lives in L2 cache
##                │     │     │     ┌── B[kc, nc] lives in L3 cache
##                │     │     │     │     ┌── C[M, N] in memory
##                │     │     │     │     │
##                jr    ir    ic    pc    jc
##   C[M, N] += α · A[M, K] × B[K, N] + β · C[M, N]
##
## Reference:
##   - BLIS paper: SGSHV14 – Anatomy of High-Performance
##     Many-Threaded Matrix Multiplication
##   - Laser (Nim BLAS): laser/primitives/matrix_multiplication/
##   - CuTe: NVIDIA/cutlass/include/cute/

import std/[macros, math]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors

{.experimental: "callOperator".}

# ═══════════════════════════════════════════════════════════════════════════
#  pack_layout — derive pack-buffer layout from zipped_divide
# ═══════════════════════════════════════════════════════════════════════════
#
#  zipped_divide groups a panel layout into (tile, rest):
#    mode 0 = tile       — e.g. (mr, 1) for A, (1, nr) for B
#    mode 1 = rest       — e.g. (num_ir, kc) for A, (kc, num_jr) for B
#
#  The pack layout nests them:
#    - tile      → LayoutLeft  (tile-contiguous: rows for A, cols for B)
#    - rest      → LayoutRight (A, transposed) or LayoutLeft (B, not)
#    - rest strides are scaled by tile_size = product(tile_shape)
#
#  Result: ((tile_shape), (rest_shape)) : ((tile_stride), (rest_stride))

template pack_layout(zd: Layout; transposed: static bool): auto =
  let tileCompact = make_layout(zd.shape[0], LayoutLeft)
  let tile_size = product(zd.shape[0])
  let restCompact = make_layout(zd.shape[1],
    when transposed: LayoutRight else: LayoutLeft)
  let restScaled = mapLeavesWith(restCompact):
    (it_sh, it_st * tile_size)
  nested_product(tileCompact, restScaled)

# ═══════════════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════════════

type ActivationFn[T] = proc(x: T): T {.nimcall.}

proc relu[T](x: T): T {.nimcall.} =
  if x > 0.T: x else: 0.T

proc identity[T](x: T): T {.nimcall.} = x

type MmaAtom = object
  ## Micro-kernel parameters (register tile shape + inner loop depth).
  mr*: int
  nr*: int
  kc*: int

# ═══════════════════════════════════════════════════════════════════════════
#  Cache & tile sizing
# ═══════════════════════════════════════════════════════════════════════════

const
  L1_CACHE_SIZE = 32 * 1024   # 32 KB — holds B micro-panel kc × nr
  L2_CACHE_SIZE = 256 * 1024  # 256 KB — holds A block mc × kc

proc autoTileParams(atom: static MmaAtom; T: typedesc; M, K: int): tuple[mc, kc: int] =
  ## Choose cache-block dimensions mc (rows) and kc (inner) so that:
  ##   - B micro-panel (kc × nr) fits in half of L1
  ##   - A block     (mc × kc) fits in half of L2
  const mr = atom.mr
  const nr = atom.nr
  const kc_atom = atom.kc

  # kc: L1 cache — B micro-panel is kc × nr elements
  let kc_max = int((L1_CACHE_SIZE div 2) div (nr * sizeof(T)))
  result.kc = min(K, kc_max)
  result.kc = (result.kc div kc_atom) * kc_atom   # round down to atom kc
  if result.kc < kc_atom:
    result.kc = min(K, kc_atom)

  # mc: L2 cache — A block is mc × kc elements
  let mc_max = int((L2_CACHE_SIZE div 2) div (result.kc * sizeof(T)))
  result.mc = min(M, mc_max)
  result.mc = (result.mc div mr) * mr              # round down to mr
  if result.mc < mr:
    result.mc = min(M, mr)

# ═══════════════════════════════════════════════════════════════════════════
#  Loop 1: Micro-kernel (ir loop) — C[mr, nr] += A[mr] · B[nr]
#
#  The innermost loop.  MR × NR elements of C live in registers.
#  For each k-step, one column of A (MR elements) and one row of B
#  (NR elements) are multiplied and accumulated into the register tile.
#
#  Known as the GEBB (Generalized Block-Block) micro-kernel in BLIS.
# ═══════════════════════════════════════════════════════════════════════════

proc gemm_ukernel[T; MR, NR: static int](
    packA, packB: ptr UncheckedArray[T];
    AB: var array[MR, array[NR, T]];
    kc: int) =
  ## Register-level tile: AB[ri][rj] += A[k*MR+ri] × B[k*NR+rj]
  for k in 0 ..< kc:
    for ri in 0 ..< MR:
      let ai = packA[k * MR + ri]
      for rj in 0 ..< NR:
        AB[ri][rj] += ai * packB[k * NR + rj]

# ═══════════════════════════════════════════════════════════════════════════
#  Epilogue — C[mr, nr] += α·f(AB) + β·C[mr, nr]
#
#  Follows the Laser / BLIS epilogue pattern:
#    1. If β == 0:   C ← 0     (clear)
#       If β != 1:   C ×= β    (scale)
#    2. If α == 1:   C += f(AB)       (fused activation)
#       If α != 1:   C += α × f(AB)   (scale + fused activation)
#
#  The C pointer is already positioned at the tile's location
#  (the caller computes `C + cRow + cCol × ldC`).
#  `mr, nr` are the actual tile dimensions (MR when full,
#  smaller at matrix edges).
# ═══════════════════════════════════════════════════════════════════════════

proc gemm_epilogue[T; MR, NR: static int; Sh, St](
    C: TensorView[T, Sh, St];
    AB: array[MR, array[NR, T]];
    mr, nr: int;
    alpha, beta: T;
    activation: ActivationFn[T]) =
  if beta == 0.T:
    for i in 0 ..< mr:
      for j in 0 ..< nr:
        C[i, j] = 0.T
  elif beta != 1.T:
    for i in 0 ..< mr:
      for j in 0 ..< nr:
        C[i, j] *= beta
  if alpha == 1.T:
    for i in 0 ..< mr:
      for j in 0 ..< nr:
        C[i, j] += activation(AB[i][j])
  else:
    for i in 0 ..< mr:
      for j in 0 ..< nr:
        C[i, j] += alpha * activation(AB[i][j])

# ═══════════════════════════════════════════════════════════════════════════
#  Top-level GEMM — C[M, N] += α·A[M, K] × B[K, N] + β·C[M, N]
#
#  Implements the full 5-loop BLIS structure:
#
#    for jc = 0 .. N-1 step nc:     Loop 5 — column panels of C/B
#      pack panel B[jc .. jc+nc] into packB
#      for pc = 0 .. K-1 step kc:   Loop 4 — rank-k updates
#        for ic = 0 .. M-1 step mc: Loop 3 — row blocks of A
#          pack block A[ic .. ic+mc] into packA
#          for jr = 0 .. nc-1 step nr:              Loop 2 — micro-panels of B
#            for ir = 0 .. mc-1 step mr:            Loop 1 — micro-tiles of A
#              gemm_ukernel + gemm_epilogue
#
#  BLIS-style strided interface: each matrix has separate row and column strides.
# ═══════════════════════════════════════════════════════════════════════════

proc gemm_strided*[T: SomeNumber](
    M, N, K: int;
    alpha: T;
    A: ptr UncheckedArray[T]; rowStrideA, colStrideA: int;
    B: ptr UncheckedArray[T]; rowStrideB, colStrideB: int;
    beta: T;
    C: ptr UncheckedArray[T]; rowStrideC, colStrideC: int;
    activation: ActivationFn[T] = identity[T]) =

  # ── Micro-kernel tile dimensions (register-level) ──
  const
    mr = 4      # micro-tile height  (rows of C in registers)
    nr = 4      # micro-tile width   (cols of C in registers)
    kc_atom = 4 # inner-loop step within micro-kernel

  # ── Cache-block dimensions ──
  const atom = MmaAtom(mr: mr, nr: nr, kc: kc_atom)
  let (mc, kc) = autoTileParams(atom, T, M, K)

  if mc < mr or kc < kc_atom:
    # Matrix too small for cache-tiled path — fall back to triple loop
    for i in 0 ..< M:
      for j in 0 ..< N:
        var acc = 0.T
        for k in 0 ..< K:
          acc += A[i * rowStrideA + k * colStrideA] * B[k * rowStrideB + j * colStrideB]
        let ci = i * rowStrideC + j * colStrideC
        C[ci] = beta * C[ci] + alpha * activation(acc)
    return

  # ── Derived quantities ──
  let num_jr = ceil_div(N, nr)     # micro-panels per column panel
  let nc     = num_jr * nr          # column panel width (N — no outer partition)
  let num_ir = mc div mr            # micro-tiles per row block
  let num_ic = ceil_div(M, mc)      # row blocks
  let num_pc = ceil_div(K, kc)      # rank-k panel steps

  # ── Input matrix views (logical layouts, stride-aware) ──
  let vA = make_view(A, make_layout((M, K), (rowStrideA, colStrideA)))
  let vB = make_view(B, make_layout((K, N), (rowStrideB, colStrideB)))
  var vC = make_view(C, make_layout((M, N), (rowStrideC, colStrideC)))

  # ── Panel / block layouts (for zipped_divide + pack) ──
  let panelA_lay = make_layout((mc, kc), (rowStrideA, colStrideA))
  let panelB_lay = make_layout((kc, nc), (rowStrideB, colStrideB))


  # ── Pack buffer layouts (3D LayoutRight — kc-major, mr/nr minor) ──
  var packDataA = newSeq[T](int(cosize(packALay)))
  var packDataB = newSeq[T](int(cosize(packBLay)))
  
  let packALay = make_layout((num_ir, kc, mr), LayoutRight)
  let packBLay = make_layout((num_jr, kc, nr), LayoutRight)
  var packA = make_view(packDataA, packALay)
  var packB = make_view(packDataB, packBLay)

  # ── Grouped (tile, rest) layouts via zipped_divide ──
  #
  #  zipped_divide splits a panel into two groups:
  #    mode 0 = micro-tile       (mr × 1 for A, 1 × nr for B)
  #    mode 1 = how tiles tile   (num_ir × kc for A, kc × num_jr for B)
  #
  #  pack_layout derives the compact pack-buffer layout from this
  #  grouping (see layout_algebra / layout_algebra).
  let srcA_zd = zipped_divide(panelA_lay, (mr, 1))
  let dstA_zd = pack_layout(srcA_zd, transposed = true)

  let srcB_zd = zipped_divide(panelB_lay, (1, nr))
  let dstB_zd = pack_layout(srcB_zd, transposed = false)

  let pA = tiled_divide(vA.layout, (mc, kc))
  let pB = tiled_divide(vB.layout, (kc, nc))

  # ═══════════════════════════════════════════════════════════════════════
  #  Loop 5 (jc):  Column-panel loop over N
  #                (not partitioned — nc = N for now)
  # ═══════════════════════════════════════════════════════════════════════

  # ═══════════════════════════════════════════════════════════════════════
  #  Loop 4 (pc):  Rank-k update loop over K
  # ═══════════════════════════════════════════════════════════════════════
  for pc in 0 ..< num_pc:                           # Loop 4
    let current_kc = min(K - pc * kc, kc)
    if current_kc <= 0:
      continue
    let last_k  = (pc == num_pc - 1) and (current_kc < kc)

    # ═══════════════════════════════════════════════════════════════════
    #  Pack panel B[kc × nc] into contiguous pack buffer
    #  (keeps B micro-panels in L3 cache)
    # ═══════════════════════════════════════════════════════════════════
    #
    #  No jc loop yet — nc = N, so one big panel of B for now.
    #  Future: partition N at socket level for multi-threading.
    #

    for jc in 0 ..< 1:      # placeholder: single jc panel (nc = N)
      let panelB = local_tile(vB, pB, pc, jc)

      if last_k:
        # Edge: kc dimension is smaller than full kc
        packB.fillWith(0.T)
        let srcB_edge = make_view(panelB,
          make_layout(((1, nr), (current_kc, num_jr)), srcB_zd.stride))
        var dstB_edge = make_view(packB,
          make_layout(((1, nr), (current_kc, num_jr)), dstB_zd.stride))
        copyFrom(dstB_edge, srcB_edge)
      else:
        let src4B = make_view(panelB, srcB_zd)
        var dst4B = make_view(packB, dstB_zd)
        copyFrom(dst4B, src4B)

      # ═══════════════════════════════════════════════════════════════
      #  Loop 3 (ic):  Row-block loop over M
      # ═══════════════════════════════════════════════════════════════
      for ic in 0 ..< num_ic:                       # Loop 3
        let current_mc = min(M - ic * mc, mc)
        if current_mc <= 0:
          continue
        let last_m  = (current_mc < mc)
        let num_ir_eff =
          if last_m:
            ceil_div(current_mc, mr)
          else:
            num_ir

        let panelA = local_tile(vA, pA, ic, pc)

        # ── Pack block A[mc × kc] into contiguous pack buffer ──
        #  (keeps A block in L2 cache)
        if last_m or last_k:
          let mr_eff = min(mr, current_mc)
          packA.fillWith(0.T)
          let srcA_edge = make_view(panelA,
            make_layout(((mr_eff, 1), (num_ir_eff, current_kc)), srcA_zd.stride))
          var dstA_edge = make_view(packA,
            make_layout(((mr_eff, 1), (num_ir_eff, current_kc)), dstA_zd.stride))
          copyFrom(dstA_edge, srcA_edge)
        else:
          let src4A = make_view(panelA, srcA_zd)
          var dst4A = make_view(packA, dstA_zd)
          copyFrom(dst4A, src4A)

        # ═══════════════════════════════════════════════════════════
        #  Loop 2 (jr):  Micro-panel loop over nc (B in L1 cache)
        # ═══════════════════════════════════════════════════════════
        for jr in 0 ..< num_jr:                     # Loop 2
          let cCol = jr * nr
          if cCol >= N: break

          let bTile = packB.slice((jr, _, _))

          # ═══════════════════════════════════════════════════════
          #  Loop 1 (ir):  Micro-tile loop over mc (registers)
          # ═══════════════════════════════════════════════════════
          for ir in 0 ..< num_ir_eff:                # Loop 1 (innermost)
            let cRow = ic * mc + ir * mr
            if cRow >= M: break

            let aTile = packA.slice((ir, _, _))
            var AB: array[mr, array[nr, T]]          # register tile, zero-init

            # ── Micro-kernel: AB += A[mr] × B[nr] over kc ──
            gemm_ukernel[T, mr, nr](
              aTile.data, bTile.data, AB, current_kc)

            # ── Epilogue: C += α·f(AB) + β·C ──
            let cTile = displace(vC, (cRow, cCol))
            gemm_epilogue(
              cTile, AB, mr, nr,
              alpha, beta, activation)

# ═══════════════════════════════════════════════════════════════════════════
#  Convenience overload for testing — wraps openArray[T] into ptr + strides
# ═══════════════════════════════════════════════════════════════════════════

proc gemm_strided*[T: SomeNumber](
    M, N, K: int;
    alpha: T;
    A: openArray[T]; rowStrideA, colStrideA: int;
    B: openArray[T]; rowStrideB, colStrideB: int;
    beta: T;
    C: var openArray[T]; rowStrideC, colStrideC: int;
    activation: ActivationFn[T] = identity[T]) =
  gemm_strided(
    M, N, K, alpha,
    cast[ptr UncheckedArray[T]](addr A[0]), rowStrideA, colStrideA,
    cast[ptr UncheckedArray[T]](addr B[0]), rowStrideB, colStrideB,
    beta,
    cast[ptr UncheckedArray[T]](addr C[0]), rowStrideC, colStrideC,
    activation)

# ═══════════════════════════════════════════════════════════════════════════
#  Test driver
# ═══════════════════════════════════════════════════════════════════════════

when isMainModule:
  import std/[random, strutils]

  proc gemm_reference(M, N, K: int; alpha: float32;
      A: openArray[float32]; rsA, csA: int;
      B: openArray[float32]; rsB, csB: int;
      beta: float32; C: var openArray[float32]; rsC, csC: int) =
    for j in 0 ..< N:
      for i in 0 ..< M:
        let ci = i * rsC + j * csC
        if beta == 0.0'f32: C[ci] = 0.0'f32
        elif beta != 1.0'f32: C[ci] *= beta
    for j in 0 ..< N:
      for k in 0 ..< K:
        let bVal = B[k * rsB + j * csB]
        if bVal != 0.0'f32:
          for i in 0 ..< M:
            C[i * rsC + j * csC] += alpha * A[i * rsA + k * csA] * bVal

  proc test(M, N, K: int; rsA, csA, rsB, csB, rsC, csC: int; label: string) =
    echo "\n### ", label
    randomize(42)
    let aLen = max((M-1)*rsA + (K-1)*csA + 1, 0)
    let bLen = max((K-1)*rsB + (N-1)*csB + 1, 0)
    let cLen = max((M-1)*rsC + (N-1)*csC + 1, 0)
    var A = newSeq[float32](aLen)
    var B = newSeq[float32](bLen)
    var C_ref = newSeq[float32](cLen)
    var C_tst = newSeq[float32](cLen)
    for i in 0 ..< A.len:   A[i] = rand(1.0'f32)
    for i in 0 ..< B.len:   B[i] = rand(1.0'f32)
    for i in 0 ..< C_ref.len: C_ref[i] = rand(1.0'f32)
    for i in 0 ..< C_tst.len: C_tst[i] = C_ref[i]

    let alpha = 1.0'f32; let beta = 1.0'f32
    gemm_reference(M, N, K, alpha,
      A, rsA, csA,
      B, rsB, csB, beta,
      C_ref, rsC, csC)

    gemm_strided[float32](
      M, N, K, alpha,
      A, rsA, csA,
      B, rsB, csB, beta,
      C_tst, rsC, csC,
      identity[float32])

    var err: float32 = 0
    for i in 0 ..< cLen:
      err = max(err, abs(C_ref[i] - C_tst[i]))
    echo "  max error: ", err.formatFloat(ffScientific, 2),
         if err < 1e-4: "  v" else: "  x"

  # Column-major (Fortran layout) — default BLIS-style
  test(16, 16, 16, 1, 16, 1, 16, 1, 16, "Square 16x16 col-major")
  # Row-major
  test(16, 16, 16, 16, 1, 16, 1, 16, 1, "Square 16x16 row-major")
  # Non-square strides
  test(8, 8, 4, 1, 10, 1, 10, 1, 10, "Non-square (8x4)")
  # Non-power-of-2
  test(10, 10, 5, 1, 12, 1, 12, 1, 12, "Non-power-of-2 (10x5)")
  # Large
  test(128, 128, 128, 1, 128, 1, 128, 1, 128, "Large 128x128 col-major")
  # Odd ukernel dims
  test(6, 16, 64,  1, 16, 1, 16, 1, 16, "UKernel 6x16")
  test(6, 32, 64,  1, 32, 1, 32, 1, 32, "UKernel 6x32")
  test(14, 32, 64, 1, 32, 1, 32, 1, 32, "UKernel 14x32")
  # Small matrix (falls through to triple-loop path)
  test(2, 2, 2, 1, 2, 1, 2, 1, 2, "Tiny 2x2 (triple-loop path)")
  echo "\nDone."
