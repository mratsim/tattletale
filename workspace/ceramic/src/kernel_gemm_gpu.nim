## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU GEMM
##
## `gemm_atom`:
##    instruction-level matrix-multiplication, either tensor-core or fused-multiply-add in the degenerate case.
##
## `gemm_ukernel`:
##    a microkernel, loop over k micropanels [m, k].[k, n]
##
## `gemm_tiled`:
##    Data partitioning
##
## `gemm_cta`:
##    GEMM at the Cooperative Thread Array (CTA) level
##    Takes ownership of a large arrays,
##    Thread partitioning
##    Lifecycle:
##    - copy async,
##    - computing the matrix multiplication
##    - applying the epilogue
##    - copying the result back

import std/macros
import ./int_tuples
import ./layouts
import ./tensors
import ./ptr_arithmetic
import ./atoms
import ./atoms_mma_partitioning
import ./layout_algebra
import ./kernel_copy_gpu
import ./atoms_copy
import ./kernel_fillwith_gpu
import ./kernel_gemm_epilogues
import ./macros/static_for
import ./kernel_gemm/nvidia_tensor_cores
import workspace/crucible

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  gemm_atom(mma, ...): register-level MMA
# ═════════════════════════════════════════════════════════════════════════

func gemm_atom*[TD, ShD, StD, TA, ShA, StA, TB, ShB, StB](
    mma: static MmaAtom,
    dFrag: var Tensor[TD, ShD, StD],
    aFrag: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA],
    bFrag: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB]) {.inline.} =
  ## Register-level MMA, in-place accumulate: dFrag += aFrag·bFrag.
  ##
  ## Args:
  ##   mma: the atom, passed static. Its instr + per-operand register
  ##        counts (V) produce the inline-asm statement
  ##   dFrag: mutable register fragment tensor, the accumulator (AB).
  ##   aFrag, bFrag: register fragment tensors, the operands
  gemm_mma(mma.instr,
           toIntVal(mma.valuesPerThread(opC)),
           toIntVal(mma.valuesPerThread(opA)),
           toIntVal(mma.valuesPerThread(opB)),
           dFrag, aFrag, bFrag)

# ═════════════════════════════════════════════════════════════════════════
#  gemm_ukernel(mma, ...): the GEBB microkernel (loop over K)
# ═════════════════════════════════════════════════════════════════════════

func gemm_ukernel*[TD, ShD, StD, TA, ShA, StA, TB, ShB, StB](
    mma: static MmaAtom,
    dFrag: var Tensor[TD, ShD, StD],
    aFrag: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA],
    bFrag: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB]) {.inline.} =
  ## a microkernel, loop over k micropanels [m, k].[k, n]
  ##
  ## Naming follows the epilogue:
  ##   D     = the destination fragment (in global memory)
  ##   AB    = the accumulator fragment
  ##   A, B  = the operand fragments
  ##
  ## Args:
  ##   mma: a compile-time MmaAtom
  ##   dFrag: destination register fragment tensor / accumulator (AB)
  ##   aFrag, bFrag: operand fragments
  const
    VA = mma.valuesPerThread(opA)
    VB = mma.valuesPerThread(opB)
    kBlocks = ShA.default[2]
  static:
    doAssert ShA.default[0] === VA,
      "gemm_ukernel: A fragment width (" & $ShA.default[0] & ") != atom valuesPerThread(opA) (" & $VA & ")"
    doAssert ShB.default[0] === VB,
      "gemm_ukernel: B fragment width (" & $ShB.default[0] & ") != atom valuesPerThread(opB) (" & $VB & ")"
    # TODO: relax the Rest == 1 restriction when the GEMM generalizes to
    # multi-rest fragments.
    doAssert ShA.default[1] === 1,
      "gemm_ukernel: A RestM (" & $ShA.default[1] & ") != 1. A k_block slice must be exactly one atom A fragment (V, 1)"
    doAssert ShB.default[1] === 1,
      "gemm_ukernel: B RestN (" & $ShB.default[1] & ") != 1. B k_block slice must be exactly one atom B fragment (V, 1)"
    doAssert ShB.default[2] === kBlocks,
      "gemm_ukernel: B k dimension (" & $ShB.default[2] & ") != A k dimension (" & $kBlocks & "). A and B must agree on the k_block count"
    doAssert dFrag.layout.cosize().toIntVal() === mma.valuesPerThread(opC),
        "gemm_ukernel: accumulator size (" & $dFrag.layout.cosize().toIntVal() &
        ") != atom valuesPerThread(opC) (" & $mma.valuesPerThread(opC) & ")"

  staticFor k_block, 0, kBlocks.toIntVal():
    gemm_atom(mma, dFrag, aFrag(_, _, k_block), bFrag(_, _, k_block))

# ═════════════════════════════════════════════════════════════════════════
#  gemm_tiled(tma, dFrag, sA, sB, TileShape, threadIdx): one k-tile of the GEMM
# ═════════════════════════════════════════════════════════════════════════
func gemm_tiled*[TA, ShA, StA, TB, ShB, StB, TD, ShD, StD](
    tma: static TiledMma,
    dFrag: var Tensor[TD, ShD, StD],
    sA: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA],
    sB: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB],
    TileShape: static tuple[M: int, N: int, K: int],
    threadIdx: int) {.inline.} =
  ## Tile-level data partitioning and bookkeeping
  ##
  ## Args:
  ##   tma: the TiledMma, atom plus (ThrM, ThrN, ThrK) thread tiling
  ##   dFrag: the accumulator fragment (V, RestM, RestN), in/out.
  ##          Assumes zeroed
  ##   sA: shared memory (tileM, tileK) tile of A, col-major,
  ##   sB: shared memory (tileN, tileK) tile of B, col-major,
  ##   TileShape: static (tileM, tileN, tileK), the k-tile dims.
  ##        tileM and tileN must be exactly the thread layout's coverage
  ##        (thrM·atomM, thrN·atomN).
  ##   threadIdx: the flat linear thread id in 0 ..< blockSize
  ##
  ## Preconditions:
  ##   - sA/sB are col-major views with shapes (tileM, tileK),
  ##     (tileN, tileK) over the CTA's staged smem tiles
  ##   - dFrag is the caller's accumulator, zeroed before the loop
  ##   - tileK mod (thrK·atomK) == 0, ThrK == 1
  ##   - threadIdx < blockSize
  ##
  ## Postconditions:
  ##   - dFrag += sA·sB over the k-tile
  ##   - sA and sB are unmodified
  ##
  ## Panic-if:
  ##   - TileShape.M/N != thrM·atomM / thrN·atomN. The thread layout must exactly cover the tile.
  ##   - tileK mod (thrK·atomK) != 0. The k-tile depth is not a multiple of the thread k-depth.
  ##   - ThrK != 1. v1 does not distribute threads along K
  ##   - view shape mismatch. Pass (tileM, tileK), (tileN, tileK) col-major views with tileK
  ##   - accumulator size != tileM·tileN div blockSize.
  const
    tileM = TileShape[0]
    tileN = TileShape[1]
    tileK = TileShape[2]

  static:
    doAssert tileM === tma.thrM * tma.atom.mnk.m,
      "gemm_tiled: TileShape.M (" & $tileM & ") != thrM·atomM (" & $tma.thrM & "·" & $tma.atom.mnk.m &
        "). The thread layout must exactly cover the tile (partition contract)"
    doAssert tileN === tma.thrN * tma.atom.mnk.n,
      "gemm_tiled: TileShape.N (" & $tileN & ") != thrN·atomN (" & $tma.thrN & "·" & $tma.atom.mnk.n &
        "). The thread layout must exactly cover the tile (partition contract)"
    doAssert tileK mod (tma.thrK * tma.atom.mnk.k) == 0,
      "gemm_tiled: TileShape.K (" & $tileK & ") mod (thrK·atomK) (" & $tma.thrK & "·" & $tma.atom.mnk.k &
        ") != 0. Use a k-tile depth multiple of thrK·atomK"
    doAssert tma.thrK == 1,
      "gemm_tiled: ThrK (" & $tma.thrK & ") != 1. v1 does not distribute threads along K"
    doAssert ShA.default[0] === tileM and ShA.default[1] === tileK,
      "gemm_tiled: A shape (" & $ShA.default[0] & ", " & $ShA.default[1] &
        ") != tile (" & $tileM & ", " & $tileK & "). Pass a (tileM, tileK) view"
    doAssert ShB.default[0] === tileN and ShB.default[1] === tileK,
      "gemm_tiled: B shape (" & $ShB.default[0] & ", " & $ShB.default[1] &
        ") != tile (" & $tileN & ", " & $tileK & "). Pass a (tileN, tileK) view"
    doAssert ShD.default[0] * ShD.default[1] * ShD.default[2] === tileM * tileN div tma.threadCount(),
      "gemm_tiled: accumulator size (" & $(ShD.default[0] * ShD.default[1] * ShD.default[2]) &
        ") != tile elements per thread (" & $(tileM * tileN div tma.threadCount()) & ")"

  let thr = tma.get_slice(threadIdx)
  # The thread's operand views for one k-tile (tileM, tileK) smem tile:
  #   tAv = (V, RestM, RestK), A fragment in smem, RestK = tileK div atomK
  let tAv = tma.partition_A(thr, sA)
  let tBv = tma.partition_B(thr, sB)

  # Gather the k-tile's fragments smem → registers, tileK deep.
  var aFragTile = make_fragment_A(tma.atom, tAv)
  var bFragTile = make_fragment_B(tma.atom, tBv)
  aFragTile.copyFrom(tAv)
  bFragTile.copyFrom(tBv)

  # the k_block microkernel: accumulate into dFrag (in/out)
  gemm_ukernel(tma.atom, dFrag, aFragTile, bFragTile)

# ═════════════════════════════════════════════════════════════════════════
#  gemm_cta(tma, D, A, B, M, N, K, epi, TileShape, mCTA, nCTA, threadIdx)
# ═════════════════════════════════════════════════════════════════════════

func gemm_cta*[TA, ShA, StA, TB, ShB, StB, TD, ShD, StD, Epi](
    tma: static TiledMma,
    D: var (TensorView[TD, ShD, StD] or Tensor[TD, ShD, StD]),
    A: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA],
    B: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB],
    M, N, K: int,
    epi: Epi,
    TileShape: static tuple[M: int, N: int, K: int],
    mCTA, nCTA, threadIdx: int) {.inline.} =
  ## GEMM at the Cooperative Thread Array (CTA) level
  ##
  ## The input shape M/N/K runtime values are tiled into static tiles
  ## derived from hardware constraint.
  ##
  ##   (M, kView) view ── local_tile((tileM, tileK), (mCTA, kCTA)) ──▶ (tileM, tileK)
  ##   (N, kView) view ── local_tile((tileN, tileK), (nCTA, kCTA)) ──▶ (tileN, tileK)
  ##
  ## Args:
  ##   tma: the TiledMma (see gemm_tiled)
  ##   D: the thread's destination fragment view,
  ##      written to by the epilogue.
  ##   A: (M, kView)
  ##   B: (N, kView)
  ##   M, N, K: input dimensions
  ##   epi: the fused epilogue op (EpiAXPBY, EpiIdentity, EpiReLU,
  ##        EpiAddBias or user-defined)
  ##   TileShape: static (tileM, tileN, tileK) derived from hardware tensor cores
  ##   mCTA, nCTA:
  ##        the CTA grid coordinates, blockIdx.x/y.
  ##        The tile origin is m0 = mCTA·tileM, n0 = nCTA·tileN
  ##   threadIdx: the flat linear thread id in 0 ..< blockSize
  ##
  ## Preconditions:
  ##   - K <= kView: the problem K does not exceed the allocated K the
  ##     views were built on, the launcher's contract.
  ##     A runtime K above kView would read past the buffer
  ##   - the view K (ShA.default[1]) is a multiple of tileK,
  ##   - the CTA grid covers 0 ..< ceil(M/tileM) by 0 ..< ceil(N/tileN),
  ##     so m0 < M and n0 < N for every launched CTA (the launcher's
  ##     contract)
  ##   - tileK mod (thrK·atomK) == 0
  ##
  ## Postconditions:
  ##   - the (mCTA, nCTA) tile of the destination := the epilogue of the
  ##     accumulated tile.
  ##     Only the elements inside the valid (M, N) range of the tile are stored
  ##   - A and B are unmodified
  ##
  ## K = 0 runs zero k-tiles: the accumulator stays zero and the
  ## epilogue stores β·C
  const kView = toIntVal(ShA.default[1])
  const
    tileM = TileShape[0]
    tileN = TileShape[1]
    tileK = TileShape[2]
    blockSize = tma.threadCount()
    unitsA = (tileM * tileK) div (4 * blockSize)   # the 16-byte chunks per thread
    unitsB = (tileN * tileK) div (4 * blockSize)
  static:
    doAssert ShA.default[1] === ShB.default[1],
      "gemm_cta: the A and B problem views must agree on the allocated K (" & $kView & " vs " &
      $toIntVal(ShB.default[1]) & "). The k-tile grid is sliced with the A view's K"
    doAssert kView mod tileK == 0,
      "gemm_cta: the allocated K (" & $kView & ") mod k-tile depth (" & $tileK & ") != 0." &
      " local_tile needs the view K to tile evenly into tileK-deep tiles"
    doAssert Epi is Epilogue,
      "gemm_cta: the epilogue op must satisfy the Epilogue concept (preflight + 2-arg apply)"

  # Accumulator
  # -----------
  # zeroed once, persists across k-tiles
  var dFrag = make_tensor(TD, D.layout.shape)
  dFrag.fillWith(TD(0))

  # Valid tile range
  let m0 = mCTA * tileM
  let n0 = nCTA * tileN
  let validM = min(M - m0, tileM)
  let validN = min(N - n0, tileN)

  # Out-of-range CTA
  if validM <= 0 or validN <= 0:
    return

  # Shared memory staging buffers
  # -----------------------------
  var smemA {.shared.}: array[tileM * tileK, TA]
  var smemB {.shared.}: array[tileN * tileK, TB]
  var sA = make_view(addr smemA[0], make_layout((tileM, tileK)))
  var sB = make_view(addr smemB[0], make_layout((tileN, tileK)))

  # Compute loop
  # ------------
  #
  #   (M, kView) view ── local_tile((tileM, tileK), (mCTA, kCTA)) ──▶ (tileM, tileK) gmem
  #   (N, kView) view ── local_tile((tileN, tileK), (nCTA, kCTA)) ──▶ (tileN, tileK) gmem
  #                              │                                        │
  #                              ▼                                        ▼
  #              kView chunked into tileK-deep tiles            load cp.async chunks
  #                  (a grid of tiles over M × kView)           + commit, wait, syncthreads
  #                                                             + gemm_tiled (smem → regs)
  #
  #
  #   The last k-tile may be partial (ragged K),
  #   with validK = K - kCTA·tileK < tileK,
  #   and the load zero-fills the k >= validK lanes.

  const tileLA = make_layout((tileM, tileK), (1, tileM))
  const tileLB = make_layout((tileN, tileK), (1, tileN))

  # Create a per-thread table that maps index -> coordinate
  # to avoid runtime division and modulo
  const unitCoordsA = block:
    var a: array[blockSize * unitsA, (int, int)]
    for tid in 0 ..< blockSize:
      let p = thrfrg_copy(tileLA, CpAsyncAtom[TA], blockSize)
      for i in 0 ..< unitsA:
        a[tid * unitsA + i] = idx2crd((tileM, tileK), toIntVal(crd2idx(p, (tid, 0, i))))
    a
  const unitCoordsB = block:
    var a: array[blockSize * unitsB, (int, int)]
    for tid in 0 ..< blockSize:
      let p = thrfrg_copy(tileLB, CpAsyncAtom[TB], blockSize)
      for i in 0 ..< unitsB:
        a[tid * unitsB + i] = idx2crd((tileN, tileK), toIntVal(crd2idx(p, (tid, 0, i))))
    a

  let kTiles = K.ceil_div(tileK)
  for kCTA in 0 ..< kTiles:
    let tA = local_tile(A, (tileM, tileK), (mCTA, kCTA))
    let tB = local_tile(B, (tileN, tileK), (nCTA, kCTA))
    let validK = min(tileK, K - kCTA * tileK)


    # Load tile A into shared memory
    let tAgA = partition_S(tA, CpAsyncAtom[TA], blockSize, threadIdx)
    var tAsA = partition_D(sA, CpAsyncAtom[TA], blockSize, threadIdx)
    var tApA: array[unitsA, bool]
    for i in 0 ..< unitsA:
      let c = unitCoordsA[threadIdx * unitsA + i]
      tApA[i] = c[0] < validM and c[1] < validK
    let tApAv = make_view(addr tApA[0], make_layout((1, unitsA)))
    copyFromIf(tAsA, tAgA, tApAv)

    # Load tile B into shared memory
    let tBgB = partition_S(tB, CpAsyncAtom[TB], blockSize, threadIdx)
    var tBsB = partition_D(sB, CpAsyncAtom[TB], blockSize, threadIdx)
    var tBpB: array[unitsB, bool]
    for i in 0 ..< unitsB:
      let c = unitCoordsB[threadIdx * unitsB + i]
      tBpB[i] = c[0] < validN and c[1] < validK
    let tBpBv = make_view(addr tBpB[0], make_layout((1, unitsB)))
    copyFromIf(tBsB, tBgB, tBpBv)

    # one commit group for both loads
    cp.async.commit_group()
    cp.async.wait_group(0)
    syncthreads() # Wait until all threads have copied their tiles
    tma.gemm_tiled(dFrag, sA, sB, TileShape, threadIdx)
    syncthreads() # Wait until all threads have processed gemm_tiled

  # Fused epilogue
  # --------------
  var o = epi
  o.preflight()
  var res = make_tensor(TD, D.layout.shape)
  o.apply(res, dFrag)
  o.storeMask = cStoreMask(tma, threadIdx, tileM, tileN, validM, validN)
  o.finalStore(res, D)
