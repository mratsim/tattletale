## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GPU Matrix Multiplication.
##
## This file implements the following primitives, ordered bottom-up
##
##   gemm_atom    one Matrix-Multiply-Accumulate (MMA) instruction on a thread's register fragments,
##                accumulated in place.
##   gemm_warp    loop over atoms
##   gemm_tiled   partition the shared memory tiles and move data into registers
##   gemm_cta     Copy data into shared memory, apply gemm_tiled, apply epilogue, store result into destination
##   gemm_kernel  the GEMM with a fused epilogue
##
## SM80 example in matryoshka-like diagram
##   - atom m16n8k8,
##   - thread layout (2, 2, 1),
##   - tile (32, 16, 32),
##   - input (64, 32)
##   - grid (2, 2).
##
## This is the manual gemm_cta configuration: tile and thread layout are passed explicitly (as in the manual gemm_cta tests).
## gemm_kernel derives them from the input instead.
## A (64, 32) input yields thread layout (4, 4, 1), tile (64, 32, 32), grid (1, 1), 512 lanes = 16 warps.
##
## Fragment shapes:
##   a thread's register fragments are arrays, one per operand,
##   each with a V axis and a Repeat axis per tile axis:
##   - A: (V, RepeatM, RepeatK), here (4, 1, 4)
##   - B: (V, RepeatN, RepeatK), here (2, 1, 4)
##   - C: (V, RepeatM, RepeatN), the accumulator, here (4, 1, 1)
## V is the values per thread per atom instruction
## m16n8k8 gives
##   - 4 for A and C
##   - 2 for B.
## Repeat* is the number of atoms the thread owns along the axis,
##     tileM = ThrM·atomM·RepeatM,
## here 1 along M and N, 4 along K.
##
## The axes are register storage, not the GEMM's (M, N, K).
## The thread's output stays 2D, RepeatM·atomM × RepeatN·atomN,
## and K is the contraction dimension.
##
## Thread organization: who owns what (worked example)
## ---------------------------------------------------
## Config (worked example):
##   - atom: m16n8k8
##   - thread layout: (2, 2, 1)
##   - tile: (32, 16, 32)
##   - CTA: one tile, 128 lanes = 4 warps
## The same 32 lanes are the cooperation unit at both gemm_atom and gemm_warp.
##
## Terminology:
##   lane = one thread (threadIdx.x % 32),
##   warp = 32 lanes (the unit that executes mma.sync),
##   CTA = the unit scheduled on one SM (blockIdx).
##
##   CTA (gemm_cta)            128 lanes = 4 warps, owns the (32, 16) output tile
##     └─ warp (tm, tn) ∈ (2, 2)      [gemm_tiled partition]
##        owns the (16, 8) sub-tile: A rows tm·16..tm·16+16, B rows tn·8..tn·8+8
##        └─ lane l ∈ 0..31           [hardware fragment layouts]
##           owns V values per atom: A: 4, B: 2, C: 4
##
##   gemm_atom:  one mma.sync. The warp's 32 lanes jointly perform
##               one 16×8×8 MMA into their C fragment
##               (16×8 = 128 values = 32 lanes × 4).
##   gemm_warp:  the same 32 lanes loop kSlice = 0 ..< 4, issuing
##               one warp-wide mma.sync per k slice
##               (RepeatK = tileK/atomK = 32/8), accumulating in place.
##               Per lane:
##                 A (4, 1, 4)
##                 B (2, 1, 4)
##                 C (4, 1, 1)
##
##   The (2, 2, 1) thread layout never splits an atom across lanes:
##   it replicates the atom 2×2 = 4 times across the 4 warps, so the warps together cover the (32, 16) CTA tile.
##
## State/data flow (per CTA tile, one tileK-sized slice of K at a time):
##   gmem A (32, kView), B (16, kView)
##     ── gemm_cta cp.async ──▶ smem A (32, 32), B (16, 32)
##     ── gemm_tiled gather ──▶ lane registers A (4, 1, 4), B (2, 1, 4)
##     ── gemm_cta zeroes C (4, 1, 1) once, before the K-loop
##     ── gemm_warp 4× mma.sync ──▶ C (4, 1, 1) accumulated
##     ── gemm_cta epilogue ──▶ gmem D (32, 16)
##
##   grid: the (64, 32) input, tiled into (2, 2) CTA tiles of (32, 16)
##   - mCTA = blockIdx.x, rows
##   - nCTA = blockIdx.y, cols
##   ┌────────────────────────────────────────────────────────────┐
##    nCTA →   (0, 0)       (0, 1)
##    mCTA ↓   ┌──────────┐   ┌──────────┐
##             │  CTA     │   │  CTA     │
##             │ (32, 16) │   │ (32, 16) │
##             └──────────┘   └──────────┘
##             ┌──────────┐   ┌──────────┐
##             │  CTA     │   │  CTA     │
##             │ (32, 16) │   │ (32, 16) │
##             └──────────┘   └──────────┘
##   └────────────────────────────────────────────────────────────┘
##
##   One CTA zoomed in.
##   The loop over the K dimension iterates the
##   ceil(K/tileK) tileK-sized slices of K.
##   ┌──────────────────────────────────────────────────────────────┐
##    CTA: one (32, 16) output tile, tiled into (2, 2) warps
##    data: gmem → smem via cp.async:
##            A (32, kView)
##            B (16, kView)
##          epilogue store:
##            D (32, 16) with ragged tile handling
##
##    (0,0)      (0,1)
##      ┌────────┐  ┌────────┐
##      │ warp   │  │ warp   │
##      │ (16, 8)│  │ (16, 8)│
##      └────────┘  └────────┘
##    (1,0)      (1,1)
##      ┌────────┐  ┌────────┐
##      │ warp   │  │ warp   │
##      │ (16, 8)│  │ (16, 8)│
##      └────────┘  └────────┘
##
##   ┌──────────────────────────────────────────────────────────────┐
##   │  one warp (16, 8)                                            │
##   │  data: per slice of K, fragments copied smem → registers     │
##   │         (V, RepeatM, RepeatK) × (V, RepeatN, RepeatK)        │
##   │ │ ┌──────────────────────────────────────────────────────┐ │ │
##   │ │  micropanel (16, 8, 8): one atomK-sized slice,           │ │
##   │ │                      4 per slice of K                    │ │
##   │ │ │ ┌──────────────────────────────────────────────────┐ │ │ │
##   │ │ │  atom m16n8k8: one MMA instruction                   │ │ │
##   │ │ │  dFrag (V, RepeatM, RepeatN) += aFrag·bFrag          │ │ │
##   │ │ │ └──────────────────────────────────────────────────┘ │ │ │
##   │ │ └──────────────────────────────────────────────────────┘ │ │
##   │ └──────────────────────────────────────────────────────────┘ │
##   └──────────────────────────────────────────────────────────────┘
##
## The dtype policy (mmaDTypeOf, atom_selector, threadLayoutOf, tile_shape)
## maps operand types to the atom, thread layout and tile. The host and the
## device run the same policy, so the config is derived on both sides
## instead of passed.

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
import ./kernel_gemm/atoms_nvidia
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
  ## One mma.sync, executed by the 32 lanes of a warp cooperatively.
  ##
  ## Worked example (m16n8k8, (2, 2, 1), tile (32, 16, 32), 128 lanes = 4 warps):
  ##   each lane runs this call on its own register slice.
  ##   The warp's 32 lanes together hold the atom's (16, 8) fragments
  ##   (A: 4 values/lane, B: 2, C: 4), then jointly perform
  ##   one 16×8×8 MMA into the warp's C fragment.
  ##
  ## Args:
  ##   mma: the hardware instruction descriptor
  ##   dFrag: mutable register fragment tensor (V, RepeatM, RepeatN), the accumulator (AB).
  ##   aFrag: the A register fragment tensor (V, RepeatM, 1), one k slice
  ##   bFrag: the B register fragment tensor (V, RepeatN, 1), one k slice
  gemm_mma(mma.instr,
           toIntVal(mma.valuesPerThread(opC)),
           toIntVal(mma.valuesPerThread(opA)),
           toIntVal(mma.valuesPerThread(opB)),
           dFrag, aFrag, bFrag)

# ═════════════════════════════════════════════════════════════════════════
#  gemm_warp(mma, ...): loop over atoms
# ═════════════════════════════════════════════════════════════════════════

func gemm_warp*[TD, ShD, StD, TA, ShA, StA, TB, ShB, StB](
    mma: static MmaAtom,
    dFrag: var Tensor[TD, ShD, StD],
    aFrag: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA],
    bFrag: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB]) {.inline.} =
  ## Loop over atoms, one gemm_atom per k slice, with dFrag += aFrag·bFrag accumulated in place.
  ##
  ## Warp-scoped loop: the 32 lanes that cooperate on one gemm_atom run it
  ## in lockstep, one warp-wide mma.sync per k slice, all accumulating
  ## into the warp's C fragment.
  ##
  ## Worked example (m16n8k8, (2, 2, 1), tile (32, 16, 32), 128 lanes = 4 warps):
  ##   per lane:
  ##     A: (4, 1, 4), 16 values
  ##     B: (2, 1, 4), 8 values
  ##     C: (4, 1, 1), 4 values
  ##   kSlices = RepeatK = tileK/atomK = 32/8 = 4.
  ##   Each warp issues 4 warp-wide mma.syncs, one per k slice of the A/B registers.
  ##
  ## Args:
  ##   mma: the hardware instruction descriptor
  ##   dFrag: mutable register fragment tensor (V, RepeatM, RepeatN), the accumulator (AB).
  ##   aFrag: the A register fragment tensor (V, RepeatM, RepeatK), one atom's worth per k slice
  ##   bFrag: the B register fragment tensor (V, RepeatN, RepeatK), one atom's worth per k slice
  const
    VA = mma.valuesPerThread(opA)
    VB = mma.valuesPerThread(opB)
    kSlices = ShA.default[2]
  static:
    doAssert ShA.default[0] === VA,
      "gemm_warp: A fragment width (" & $ShA.default[0] & ") != atom valuesPerThread(opA) (" & $VA & ")"
    doAssert ShB.default[0] === VB,
      "gemm_warp: B fragment width (" & $ShB.default[0] & ") != atom valuesPerThread(opB) (" & $VB & ")"
    # TODO: relax the Repeat == 1 restriction when the GEMM generalizes to
    # multi-Repeat fragments.
    doAssert ShA.default[1] === 1,
      "gemm_warp: A RepeatM (" & $ShA.default[1] & ") != 1. A k slice must be exactly one atom A fragment (V, 1)"
    doAssert ShB.default[1] === 1,
      "gemm_warp: B RepeatN (" & $ShB.default[1] & ") != 1. B k slice must be exactly one atom B fragment (V, 1)"
    doAssert ShB.default[2] === kSlices,
      "gemm_warp: B k dimension (" & $ShB.default[2] & ") != A k dimension (" & $kSlices & "). A and B must agree on the k slice count"
    doAssert dFrag.layout.cosize().toIntVal() === mma.valuesPerThread(opC),
        "gemm_warp: accumulator size (" & $dFrag.layout.cosize().toIntVal() &
        ") != atom valuesPerThread(opC) (" & $mma.valuesPerThread(opC) & ")"

  staticFor kSlice, 0, kSlices.toIntVal():
    gemm_atom(mma, dFrag, aFrag(_, _, kSlice), bFrag(_, _, kSlice))

# ═════════════════════════════════════════════════════════════════════════
#  gemm_tiled(tma, dFrag, sA, sB, TileShape, threadIdx)
# ═════════════════════════════════════════════════════════════════════════

func gemm_tiled*[TA, ShA, StA, TB, ShB, StB, TD, ShD, StD](
    tma: static TiledMma,
    dFrag: var Tensor[TD, ShD, StD],
    sA: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA],
    sB: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB],
    TileShape: static tuple[M: int, N: int, K: int],
    threadIdx: int) {.inline.} =
  ## Returns the thread's fragment of the tile-level matmul
  ## for one tileK-sized slice of K:
  ##   dTile += aTile·bTile.
  ## with
  ##   dTile of shape (TileM, TileN)
  ##   aTile of shape (TileM, TileK)
  ##   bTile of shape (TileN, TileK)
  ##
  ## In practice dTile, aTile, bTile never materialize as tensors.
  ## The thread tiling (ThrM, ThrN) partitions them across the threads,
  ## according to the hardware MMA requirements
  ## with the shapes (V, RepeatM, RepeatN)+=(V, RepeatM, RepeatK)·(V, RepeatN, RepeatK):
  ##   - V is the values per thread per atom instruction,
  ##   - RepeatM/N/K the number of atoms per threaf along M, N, K
  ##
  ## Worked example (m16n8k8, (2, 2, 1), tile (32, 16, 32), 128 lanes = 4 warps):
  ##   the CTA's 4 warps cover the (32, 16) output tile (C never lives in smem).
  ##   Warp (tm, tn) ∈ (2, 2) owns the (16, 8) C sub-tile, reading from smem:
  ##     A: rows tm·16..tm·16+16
  ##     B: rows tn·8..tn·8+8
  ##   Each of its 32 lanes gathers its V values into registers:
  ##     A: (4, 1, 4)
  ##     B: (2, 1, 4)
  ##   It then hands off to gemm_warp.
  ##   C (4, 1, 1) was pre-zeroed by gemm_cta.
  ##
  ## Args:
  ##   tma: the TiledMma, atom plus (ThrM, ThrN, ThrK) thread tiling
  ##   dFrag: the accumulator fragment (V, RepeatM, RepeatN), in/out.
  ##          The caller zeroes it before the loop over the K dimension
  ##   sA: shared memory (tileM, tileK) tile of A, col-major
  ##   sB: shared memory (tileN, tileK) tile of B, col-major
  ##   TileShape: static (tileM, tileN, tileK). tileM and tileN must be
  ##        exactly the thread layout's coverage (thrM·atomM, thrN·atomN)
  ##   threadIdx: the flat linear thread id in 0 ..< tma.threadCount()
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
        ") != 0. Use a tileK multiple of thrK·atomK"
    doAssert tma.thrK == 1,
      "gemm_tiled: ThrK (" & $tma.thrK & ") != 1. At the moment, threads are not distributed along K"
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
  # The thread's operand views for the (tileM, tileK) smem tile:
  #   tAv = (V, RepeatM, RepeatK), A fragment in smem, RepeatK = tileK div atomK
  let tAv = tma.partition_A(thr, sA)
  let tBv = tma.partition_B(thr, sB)

  # Gather the tileK-sized slice's fragments smem → registers.
  var aFragTile = make_fragment_A(tma.atom, tAv)
  var bFragTile = make_fragment_B(tma.atom, tBv)
  aFragTile.copyFrom(tAv)
  bFragTile.copyFrom(tBv)

  # gemm_warp: accumulate into dFrag (in/out)
  gemm_warp(tma.atom, dFrag, aFragTile, bFragTile)

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
  ## Returns the (mCTA, nCTA) tile of the GEMM D = A·B:
  ## the epilogue of aTile·bTile, accumulated over the tile's slices of K.
  ## One tile per Cooperative Thread Array (CTA) in the grid,
  ## one tileK-sized slice of K prepared in smem per loop iteration.
  ##
  ## Worked example (m16n8k8, (2, 2, 1), tile (32, 16, 32), 128 lanes = 4 warps):
  ##   the CTA owns the (32, 16) output tile.
  ##   Per tileK slice of K it stages the smem tiles via cp.async:
  ##     A: (32, 32)
  ##     B: (16, 32)
  ##   The 4 warps compute the tile through gemm_tiled + gemm_warp,
  ##   and the epilogue stores the (32, 16) D tile.
  ##
  ##   A: (M, kView) ── local_tile((tileM, tileK), (mCTA, kCTA)) ──▶ (tileM, tileK)
  ##      ── cp.async, 16-byte chunks per thread ──▶ smem ── partition_A ──▶ aFrag (V, RepeatM, RepeatK)
  ##   B: (N, kView) ── local_tile((tileN, tileK), (nCTA, kCTA)) ──▶ (tileN, tileK)
  ##      ── cp.async, 16-byte chunks per thread ──▶ smem ── partition_B ──▶ bFrag (V, RepeatN, RepeatK)
  ##                                                                                 │
  ##                                                                                 ▼
  ##                  gemm_warp: one gemm_atom per k slice,
  ##                  dFrag (V, RepeatM, RepeatN) += aFrag(_, _, kSlice)·bFrag(_, _, kSlice),
  ##                  per-slice fragments (V, RepeatM, 1) and (V, RepeatN, 1)
  ##                                                                                 ▲
  ##                                                                                 │
  ##   D: (mCTA, nCTA) tile ◀── finalStore ◀── epilogue ◀── dFrag (V, RepeatM, RepeatN)
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
  ## K = 0 runs no slices of the K dimension.
  ## The accumulator stays zero and the epilogue stores only its own terms:
  ##   for EpiAXPBY, β·C.
  const kView = toIntVal(ShA.default[1])
  const
    tileM = TileShape[0]
    tileN = TileShape[1]
    tileK = TileShape[2]
    blockSize = tma.threadCount()
    unitsA = (tileM * tileK) div (numPacked(CpAsyncAtom[TA]) * blockSize)
    unitsB = (tileN * tileK) div (numPacked(CpAsyncAtom[TB]) * blockSize)
  static:
    doAssert ShA.default[1] === ShB.default[1],
      "gemm_cta: the A and B input views must agree on the allocated K (" & $kView & " vs " &
      $toIntVal(ShB.default[1]) & "). The K slicing uses the A view's K"
    doAssert kView mod tileK == 0,
      "gemm_cta: the allocated K (" & $kView & ") mod tileK (" & $tileK & ") != 0." &
      " local_tile needs the view K to tile evenly into tileK-sized slices of K"
    doAssert Epi is Epilogue,
      "gemm_cta: the epilogue op must satisfy the Epilogue concept (preflight + 2-arg apply)"

  # Accumulator
  # -----------
  # zeroed once, persists across the slices of K
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
  var smemA {.smem.}: array[tileM * tileK, TA]
  var smemB {.smem.}: array[tileN * tileK, TB]
  var sA = make_view(addr smemA[0], make_layout((tileM, tileK)))
  var sB = make_view(addr smemB[0], make_layout((tileN, tileK)))

  # Compute loop
  # ------------
  #
  #   (M, kView) view ── local_tile((tileM, tileK), (mCTA, kCTA)) ──▶ (tileM, tileK) gmem
  #   (N, kView) view ── local_tile((tileN, tileK), (nCTA, kCTA)) ──▶ (tileN, tileK) gmem
  #                              │                                        │
  #                              ▼                                        ▼
  #              the view's K, chunked into               load cp.async chunks
  #              tileK-sized slices of K, one             + commit, wait, syncthreads
  #              tile per (mCTA, kCTA)                    + gemm_tiled (smem → regs)
  #
  #
  #   The last slice of K may be partial (ragged K),
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
    copyFromIfAsync(tAsA, tAgA, tApAv)

    # Load tile B into shared memory
    let tBgB = partition_S(tB, CpAsyncAtom[TB], blockSize, threadIdx)
    var tBsB = partition_D(sB, CpAsyncAtom[TB], blockSize, threadIdx)
    var tBpB: array[unitsB, bool]
    for i in 0 ..< unitsB:
      let c = unitCoordsB[threadIdx * unitsB + i]
      tBpB[i] = c[0] < validN and c[1] < validK
    let tBpBv = make_view(addr tBpB[0], make_layout((1, unitsB)))
    copyFromIfAsync(tBsB, tBgB, tBpBv)

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
  var tmp = make_tensor(TD, D.layout.shape)
  o.apply(tmp, dFrag)
  o.storeMask = cStoreMask(tma, threadIdx, tileM, tileN, validM, validN)
  o.finalStore(D, tmp)

# ═════════════════════════════════════════════════════════════════════════
#  The dtype policy: atom → thread layout → tile
# ═════════════════════════════════════════════════════════════════════════
#
#  gemm_kernel computes its tma config internally.
#  The host and the device run the same policy: the same atom, thread layout and tile.
#  TODO: refactor this whole section: the dtype mapping is a naive first cut.

template mmaDTypeOf(T: typedesc): MmaDType =
  ## The MmaDType for an operand type.
  # TODO: naive mapping: a uint32 operand may pack 4 × fp8 or 2 × fp16,
  # not just TF32. Rework when the packed datatypes land.
  when T is uint32: mdtTF32
  elif T is float32: mdtF32
  else:
    {.error: "mmaDTypeOf: no MmaDType for the operand type " & $T &
      ". At the moment: uint32 (TF32) and float32".}

template atom_selector*(TA, TB, TC: typedesc): auto =
  ## Derive the MMA atom for the operand types.
  # TODO: naive dtype matching: the operand types may pack smaller
  # datatypes (4 × fp8, 2 × fp16). Rework when those land.
  when mmaDTypeOf(TA) == mdtTF32 and mmaDTypeOf(TB) == mdtTF32 and mmaDTypeOf(TC) == mdtF32:
    SM80_16x8x8_F32TF32TF32F32_TN
  else:
    {.error: "atom_selector: no atom defined for the operand types (" & $TA & ", " & $TB & ", " & $TC &
      ").".}

proc make_tiled_mma*[LA, LB, LC, Sh, St](
    a: MmaAtom[LA, LB, LC],
    thread_layout: Layout[Sh, St]): TiledMma[MmaAtom[LA, LB, LC], Layout[Sh, St]] {.inline.} =
  TiledMma[MmaAtom[LA, LB, LC], Layout[Sh, St]](atom: a, threadLayout: thread_layout)

template threadLayoutOf*(atom: static MmaAtom, M, N: static int): auto =
  ## Thread tiling for the input M and N, derived from the atom's mnk dimensions:
  ##   thrM = M div atom.mnk.m, thrN = N div atom.mnk.n
  ## The input M and N must be multiples of the atom.
  # TODO: this needs M, N known at compile-time but they are dynamic
  make_layout((M div atom.mnk.m, N div atom.mnk.n, 1))

template tile_shape*(tma: static TiledMma; tileK: static int): auto =
  ## The tile one CTA computes: (thrM·atomM, thrN·atomN, tileK),
  ## the thread layout times the atom on M and N,
  ## one tileK-sized slice of K.
  ## The grid is M/tileM × N/tileN CTAs.
  (M: tma.thrM * tma.atom.mnk.m, N: tma.thrN * tma.atom.mnk.n, K: tileK)

# ═════════════════════════════════════════════════════════════════════════
#  gemm_kernel(D, A, B, epi)
# ═════════════════════════════════════════════════════════════════════════

proc gemm_kernel*[TA, ShA, StA, TB, ShB, StB, TC, ShC, StC, Epi](
    D: var (TensorView[TC, ShC, StC] or Tensor[TC, ShC, StC]),
    A: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA],
    B: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB],
    epi: Epi) =
  ## Computes D = f(A·B), with f an epilogue.
  ##
  ## Worked example (m16n8k8, tile (32, 16, 32)):
  ##   gemm_kernel derives the thread layout, tile and grid from the input,
  ##   then partitions D per thread (partition_C) and shards the epilogue.
  ##   For a (32, 16) input:
  ##     thread layout: (2, 2, 1)
  ##     tile: (32, 16, 32)
  ##     grid: 1×1
  ##     CTA: 128 lanes = 4 warps
  ##   Each CTA computes the (32, 16) output tile via gemm_cta.
  ##   For a (64, 32) input:
  ##     thread layout: (4, 4, 1)
  ##     tile: (64, 32, 32)
  ##     grid: 1×1
  ##     CTA: 512 lanes = 16 warps
  ##
  ## Example: C += α·AB + β·C is implemented via
  ##   let pA = make_view(A, (M, K), (1, M))
  ##   let pB = make_view(B, (N, K), (1, N))
  ##   var pC = make_view(C, (M, N), (1, M))
  ##   gemm_kernel(pC, pA, pB, initEpiAXPBY(alpha, beta, pC))
  ##
  ## Args:
  ##   D: the (M, N) destination view, written by the epilogue
  ##   A: the (M, K) A view
  ##   B: the (N, K) B view
  ##   epi: the epilogue config, e.g. initEpiAXPBY(alpha, beta, pC)
  ##
  ## The view K must be a multiple of 32 (one tileK-sized slice of K per loop iteration).
  ## The input M/N must be a multiple of the atom: the thread layout times the atom, thrM·atomM == M.
  ## At the moment: 32-bit operands (TF32) with float32 accumulation.
  static:
    doAssert sizeof(TA) == 4 and sizeof(TB) == 4,
      "gemm_kernel: at the moment, 32-bit operands only (the SM80 TF32 atom)"
    doAssert TC is float32,
      "gemm_kernel: at the moment, float32 accumulation only"
  const
    M = toIntVal(ShA.default[0])
    K = toIntVal(ShA.default[1])
    N = toIntVal(ShB.default[0])
    layout = threadLayoutOf(atom_selector(TA, TB, TC), M, N)
    # TODO: temporary SM80 tileK, hardcoded until the dtype policy derives
    # it per architecture and datatype. The smem tile's K size: gemm_cta
    # prepares one tileK-sized slice of K at a time, whatever the input K.
    defaultTileK = 32
  template tma: untyped = make_tiled_mma(atom_selector(TA, TB, TC), layout)
  const
    (tileM, tileN, tileK) = tile_shape(tma, defaultTileK)
  static:
    doAssert M mod tileM == 0 and N mod tileN == 0,
      "gemm_kernel: the input (" & $M & ", " & $N & ") is not a multiple of the tile (" &
      $tileM & ", " & $tileN & "). At the moment, ragged tiles are not expressible through gemm_kernel"
  let mCTA = int(blockIdx.x)
  let nCTA = int(blockIdx.y)
  let thr = tma.get_slice(int(threadIdx.x))
  var tD = local_tile(D, (tileM, tileN), (mCTA, nCTA))
  var tDv = tma.partition_C(thr, tD)
  gemm_cta(tma, tDv, A, B, M, N, K, epi.shard(tma, thr, mCTA, nCTA),
           (M: tileM, N: tileN, K: tileK), mCTA, nCTA, int(threadIdx.x))
