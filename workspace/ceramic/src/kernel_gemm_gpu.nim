## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GEMM fragment operations, the GEBB microkernel, and the tiled GEMM.
##
## `gemm_atom`: the MMA (fragment-level) GEMM, one instruction on the
## thread's register fragments, in-place accumulate. The atom's `instr` +
## fragment counts + register types produce the inline-asm statement
## (NVIDIA mma.sync via kernel_gemm/nvidia_tensor_cores.nim; AMD MFMA and
## Intel AMX construct their instructions differently). No kc loop, no
## epilogue. The caller owns those.
## (the naive whole-tile outer-product reference oracle lives in the
## tests: `gemm_ref` in tests/gemm/gemm_test_lib.nim)
##
## `gemm_ukernel`: the GEBB microkernel, the loop over K on top of
##   `gemm_atom` (one atom instruction per k_block, accumulated in
##   dFrag). Atom-parametric. The same signature serves GPU tensor cores
##   and CPU FMA/AMX atoms.
##
## `gemm_tiled`: one k-tile of the problem GEMM: thread partitioning,
##   fragment gathering (tileK deep), and the k_block loop in
##   `gemm_ukernel`, accumulating into the caller's dFrag (in/out: the
##   persistent accumulator). No accumulator zeroing, no k-tile
##   loop, no epilogue (gemm_cta owns all three).
##
## `gemm_cta`: one CTA tile of the problem GEMM, selected by grid
##   coordinates (blockIdx.x/y). local_tile over the (M, K) / (N, K)
##   problem views yields each k-tile as a static (tileM, tileK) /
##   (tileN, tileK) view (K chunked into tileK-deep tiles); it loops them (the
##   gemm_k_iterations loop), zeroes the accumulator once, and runs
##   the fused epilogue once after the loop.
##
## Data flow: one CTA tile (tileM×tileN) over the full problem K:
##
## ```
## gemm_cta:  init dFrag := 0 before the k-tile loop
##              |  for kCTA in 0 ..< kTiles:          # k-tile loop, static bound
##              |    tA_k = local_tile(A, (tileM, tileK), (mCTA, kCTA))  # static (tileM, tileK) view
##              |    tB_k = local_tile(B, (tileN, tileK), (nCTA, kCTA))  #   (K chunked into tileK-deep tiles)
##              ▼
## gemm_tiled: partition_A/B of one k-tile
##              ▼
##        tAv (V,RestM,tileK div atomK)   tBv (V,RestN,tileK div atomK)
##              │ gather gmem → registers (make_fragment_A/B + copyFrom)
##              └──────────────┬───────────────┘
##                             ▼
##              gemm_ukernel: dFrag += Σ_k aFrag(_,_,k)·bFrag(_,_,k)
##                             │   one gemm_atom per k_block
##                             ▼
##                      dFrag (persistent accumulator)
##              │
##              ▼  after the loop: the fused epilogue (once):
##        D = f(AB)                    (preflight stages the op operands:
##        C (tileM, tileN) gmem         C for AXPBY, bias for AddBias)
## ```
##
## Loop nesting (kernel → collective → microkernel → atom):
##
## ```
## gemm_cta:   tile + k-tile origins from blockIdx and the loop var
##              init dFrag := 0 before the k-tile loop
##              for kCTA in 0 ..< kTiles:              # k-tile loop, static bound
##                gemm_tiled: partition + gather (tileK)    # gmem → registers
##                gemm_ukernel: for k_block in 0 ..< kBlocks  # k_block loop
##                  gemm_atom(mma, dFrag, aFrag(_,_,k), bFrag(_,_,k))  # one mma.sync
##              fused epilogue once after the loop
## ```
##
## Tile/block hierarchy:
##
## ```
## problem (M, N, K)                                  gmem       the whole GEMM
## └─ threadblock tile (tileM, tileN, K)              1 CTA      ← gemm_cta's blockIdx pick
##    ├─ k-tile (tileM, tileN, tileK)                1 k-tile    ← gemm_cta's loop var
##    │   ├─ warp tile (tileM/nW, tileN/nW)          1 warp
##    │   │   └─ thread tile / fragment              1 thread   ← rmem (aFragTile/dFrag)
##    │   └─ k_block (atomK)                         1 warp     ← one gemm_atom
## ```
##
## Memory spaces:
##   gmem = global memory (the problem tensors)
##   smem = shared memory (per-CTA staging: the epilogue's C operand)
##   rmem = register memory (the per-thread fragments)
##
## 1 threadblock tile = K/atomK k_blocks, looped in kTiles = K/tileK
## k-tiles of tileK/atomK k_blocks each (tileK, the k-tile
## depth). Split-K is not used.
##
## NVIDIA mma.sync assembly construction lives in
## kernel_gemm/nvidia_tensor_cores.nim.

import std/macros
import ./int_tuples
import ./layouts
import ./tensors
import ./ptr_arithmetic
import ./atoms
import ./atoms_mma_partitioning
import ./layout_algebra
import ./kernel_copy_gpu
import ./kernel_fillwith_gpu
import ./kernel_gemm_epilogues
import ./macros/static_for
import ./kernel_gemm/nvidia_tensor_cores

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  gemm_atom(mma, ...): register-level MMA
# ═════════════════════════════════════════════════════════════════════════

func gemm_atom*[TD, ShD, StD, TA, ShA, StA, TB, ShB, StB](
    mma: static MmaAtom;
    dFrag: var Tensor[TD, ShD, StD];
    aFrag: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA];
    bFrag: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB]) {.inline.} =
  ## Register-level MMA, in-place accumulate: dFrag += aFrag·bFrag.
  ##
  ## Args:
  ##   mma: the atom, passed static. Its instr + per-operand register
  ##        counts (V) produce the inline-asm statement
  ##   dFrag: var register fragment tensor, the accumulator (AB). The asm
  ##        output writes it in place
  ##   aFrag, bFrag: register fragment tensors, the operands
  ##
  ## The language evaluates the tensor arguments once at the call
  ## boundary, so k_block slices like `aFrag(_, _, k)` pass as arguments.
  ## gemm_mma (kernel_gemm/nvidia_tensor_cores.nim) builds the asm and
  ## hardcodes float32/uint32 (TF32). See the TODO there.
  const
    aV = toIntVal(mma.valuesPerThread(opA))
    bV = toIntVal(mma.valuesPerThread(opB))
    dV = toIntVal(mma.valuesPerThread(opC))
  gemm_mma(mma.instr, dV, aV, bV, dFrag, aFrag, bFrag)

# ═════════════════════════════════════════════════════════════════════════
#  gemm_ukernel(mma, ...): the GEBB microkernel (loop over K)
# ═════════════════════════════════════════════════════════════════════════

func gemm_ukernel*[TD, ShD, StD, TA, ShA, StA, TB, ShB, StB](
    mma: static MmaAtom;
    dFrag: var Tensor[TD, ShD, StD];
    aFrag: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA];
    bFrag: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB]) {.inline.} =
  ## GEBB microkernel: dFrag += Σ_k aFrag(_,_,k)·bFrag(_,_,k), one gemm_atom
  ## per k_block, accumulated in dFrag. The loop over K lives here, not in
  ## gemm_tiled.
  ##
  ## Naming follows the epilogue:
  ##   D     = the destination fragment (in global memory)
  ##   AB    = the accumulator fragment
  ##   A, B  = the operand fragments
  ##
  ## Args:
  ##   mma: a compile-time MmaAtom (bkGPU_TensorCore / bkCPU_X86_AMX, has
  ##        `instr`), passed `static`: the atom is data that monomorphizes
  ##        the kernel
  ##   dFrag: var register fragment tensor, the accumulator (AB) across all
  ##        k_blocks, in-place accumulate
  ##   aFrag, bFrag: operand fragments (make_fragment_A/B), shape
  ##        (VA, RestM, RestK) / (VB, RestN, RestK). V flattened to the
  ##        atom register order. RestK counts k_blocks, each mma.k elements
  ##        deep. A k_block slice is a (VA, RestM) view at data offset
  ##        k·VA·RestM. v1 requires RestM == RestN == 1: each k_block
  ##        slice must be exactly one atom fragment (the static block
  ##        below asserts it; TODO there tracks relaxing this to
  ##        multi-rest fragments).
  ##
  ## The unrolled staticFor binds k_block to each k-block index. The
  ## constant indices keep the asm operands register-resident: a runtime
  ## k would spill the fragment data to local memory and break the
  ## "f"/"r" constraints.
  ## Atom-parametric: the same signature serves GPU tensor-core atoms and
  ## CPU FMA/AMX atoms (the atom decides the per-k_block instruction).
  ##
  ## gemm_tiled calls it once per k-tile: the fragment's RestK mode is
  ## the per-tile k_block count (tileK div atomK), so the
  ## kBlocks asserts below hold per-tile.
  const
    VA = toIntVal(mma.valuesPerThread(opA))
    VB = toIntVal(mma.valuesPerThread(opB))
    kBlocks = toIntVal(ShA.default[2])
  static:
    doAssert toIntVal(ShA.default[0]) === VA,
      "gemm_ukernel: A fragment width (" & $toIntVal(ShA.default[0]) & ") != atom valuesPerThread(opA) (" & $VA & ")"
    doAssert toIntVal(ShB.default[0]) === VB,
      "gemm_ukernel: B fragment width (" & $toIntVal(ShB.default[0]) & ") != atom valuesPerThread(opB) (" & $VB & ")"
    # TODO: relax the Rest == 1 restriction when the GEMM generalizes to
    # multi-rest fragments: the fragment becomes (V, RestM, RestN) per
    # the fragment becomes (V, RestM, RestN) per k_block and the k_block
    # loop issues one gemm_atom per (m, n) rest pair. The gemm_tiled
    # C-fragment size check below already uses the full (V, RestM, RestN)
    # product and needs no change.
    doAssert toIntVal(ShA.default[1]) === 1,
      "gemm_ukernel: A RestM (" & $toIntVal(ShA.default[1]) & ") != 1. A k_block slice must be exactly one atom A fragment (V, 1)"
    doAssert toIntVal(ShB.default[1]) === 1,
      "gemm_ukernel: B RestN (" & $toIntVal(ShB.default[1]) & ") != 1. B k_block slice must be exactly one atom B fragment (V, 1)"
    doAssert toIntVal(ShB.default[2]) === kBlocks,
      "gemm_ukernel: B k dimension (" & $toIntVal(ShB.default[2]) & ") != A k dimension (" & $kBlocks &
        "). A and B must agree on the k_block count"
    doAssert toIntVal(cosize(dFrag.layout)) === toIntVal(mma.valuesPerThread(opC)),
      "gemm_ukernel: accumulator size (" & $(toIntVal(cosize(dFrag.layout))) &
        ") != atom valuesPerThread(opC) (" & $(toIntVal(mma.valuesPerThread(opC))) & ")"
  staticFor k_block, 0, kBlocks:
    gemm_atom(mma, dFrag, aFrag(_, _, k_block), bFrag(_, _, k_block))

# ═════════════════════════════════════════════════════════════════════════
#  gemm_tiled(tma, dFrag, A, B, TileShape, threadIdx): one k-tile of the GEMM
# ═════════════════════════════════════════════════════════════════════════

func gemm_tiled*[TA, ShA, StA, TB, ShB, StB, TD, ShD, StD](
    tma: static TiledMma;
    dFrag: var Tensor[TD, ShD, StD];   # the accumulator, in/out
    A: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA];
    B: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB];
    TileShape: static tuple[M: int, N: int, K: int];
    threadIdx: int) {.inline.} =
  ## One k-tile of the problem GEMM: partition, gather, k_block microkernel.
  ##
  ## This is the per-k-tile compute:
  ##   the static (tileM, tileN, tileK) k-tile views are partitioned into
  ##   this thread's (V, RestM, RestK) operand views, gathered gmem →
  ##   registers once, and gemm_ukernel accumulates into the caller's
  ##   dFrag (in/out).
  ## No accumulator zeroing, no k-tile loop, no epilogue:
  ##   gemm_cta zeroes dFrag once, loops the k-tiles, and runs the fused
  ##   epilogue once after the loop.
  ##
  ## Args:
  ##   tma: the TiledMma, atom plus (ThrM, ThrN, ThrK) thread tiling
  ##   dFrag: the accumulator fragment (V, RestM, RestN), in/out. The
  ##        caller zeroes it before the k-tile loop and reads it in the
  ##        epilogue after. Its shape type is the partition_C shape (the
  ##        epilogue's shared-Sh contract)
  ##   A: col-major (tileM, tileK) view, ONE k-tile of the CTA's A tile,
  ##        element type TA (tf32 uint32 in v1)
  ##   B: col-major (tileN, tileK) view, ONE k-tile of the CTA's B tile,
  ##        element type TB (tf32 uint32 in v1)
  ##   TileShape: static (tileM, tileN, tileK), the k-tile dims. tileM and
  ##        tileN must be exactly the thread layout's coverage
  ##        (thrM·atomM, thrN·atomN). tileK is the k-tile depth
  ##   threadIdx: the flat linear thread id in 0 ..< blockSize (a
  ##        multi-dimensional block must be linearized by the caller)
  ##
  ## are gathered gmem → registers once (0.75·tileK registers per
  ## operand). gemm_ukernel loops the k_blocks into dFrag.
  ##
  ## Preconditions:
  ##   - A/B are col-major views with shapes (tileM, tileK), (tileN, tileK)
  ##   - dFrag is the caller's accumulator, zeroed before the loop
  ##   - tileK mod (thrK·atomK) == 0, ThrK == 1
  ##   - threadIdx < blockSize
  ##   - the backing buffers must address the tile. The ragged underlying
  ##     allocation is the caller's contract, not checked in v1
  ##
  ## Postconditions:
  ##   - dFrag += A·B over the k-tile's K
  ##   - A and B are unmodified
  ##
  ## Panic-if (expansion-time rejections):
  ##   - TileShape.M/N != thrM·atomM / thrN·atomN. The thread layout must
  ##     exactly cover the tile (the partition contract). Fix the TiledMma
  ##     thread layout or the tile
  ##   - tileK mod (thrK·atomK) != 0. The k-tile depth is not a multiple
  ##     of the thread k-depth. Use a tile K multiple of thrK·atomK
  ##   - ThrK != 1. v1 does not distribute threads along K
  ##   - view shape mismatch. Pass (tileM, tileK), (tileN, tileK) col-major
  ##     views with tileK
  ##   - accumulator size != tileM·tileN div blockSize. dFrag must be the
  ##     thread's full C fragment for the tile
  const
    tileM = TileShape[0]
    tileN = TileShape[1]
    tileK = TileShape[2]          # the k-tile depth
    atomM = tma.atom.mnk.m
    atomN = tma.atom.mnk.n
    atomK = tma.atom.mnk.k
    thrM  = toIntVal(tma.threadLayout.shape[0])
    thrN  = toIntVal(tma.threadLayout.shape[1])
    thrK  = toIntVal(tma.threadLayout.shape[2])
    blockSize = toIntVal(tma.atom.threadCount(opA)) * thrM * thrN * thrK

  static:
    doAssert tileM === thrM * atomM,
      "gemm_tiled: TileShape.M (" & $tileM & ") != thrM·atomM (" & $thrM & "·" & $atomM &
        "). The thread layout must exactly cover the tile (partition contract)"
    doAssert tileN === thrN * atomN,
      "gemm_tiled: TileShape.N (" & $tileN & ") != thrN·atomN (" & $thrN & "·" & $atomN &
        "). The thread layout must exactly cover the tile (partition contract)"
    doAssert tileK mod (thrK * atomK) == 0,
      "gemm_tiled: TileShape.K (" & $tileK & ") mod (thrK·atomK) (" & $thrK & "·" & $atomK &
        ") != 0. Use a k-tile depth multiple of thrK·atomK"
    doAssert thrK == 1,
      "gemm_tiled: ThrK (" & $thrK & ") != 1. v1 does not distribute threads along K"
    doAssert ShA.default[0] === tileM and ShA.default[1] === tileK,
      "gemm_tiled: A shape (" & $ShA.default[0] & ", " & $ShA.default[1] &
        ") != tile (" & $tileM & ", " & $tileK & "). Pass a (tileM, tileK) view"
    doAssert ShB.default[0] === tileN and ShB.default[1] === tileK,
      "gemm_tiled: B shape (" & $ShB.default[0] & ", " & $ShB.default[1] &
        ") != tile (" & $tileN & ", " & $tileK & "). Pass a (tileN, tileK) view"
    doAssert toIntVal(ShD.default[0]) * toIntVal(ShD.default[1]) * toIntVal(ShD.default[2]) === tileM * tileN div blockSize,
      "gemm_tiled: accumulator size (" & $(toIntVal(ShD.default[0]) * toIntVal(ShD.default[1]) * toIntVal(ShD.default[2])) &
        ") != tile elements per thread (" & $(tileM * tileN div blockSize) & ")"

  let thr = tma.get_slice(threadIdx)
  # The thread's operand views for ONE k-tile (static (tileM, tileK) tile):
  #   tAv = (V, RestM, RestK), my A fragment in gmem, RestK = tileK div atomK
  let tAv = tma.partition_A(thr, A)
  let tBv = tma.partition_B(thr, B)

  # gather the k-tile's fragments gmem → registers, tileK deep
  var aFragTile = make_fragment_A(tma.atom, tAv)
  aFragTile.copyFrom(tAv)
  var bFragTile = make_fragment_B(tma.atom, tBv)
  bFragTile.copyFrom(tBv)

  # the k_block microkernel: accumulate into dFrag (in/out)
  gemm_ukernel(tma.atom, dFrag, aFragTile, bFragTile)

# ═════════════════════════════════════════════════════════════════════════
#  gemm_cta(tma, D, A, B, epi, TileShape, mCTA, nCTA, threadIdx)
# ═════════════════════════════════════════════════════════════════════════

func gemm_cta*[TA, ShA, StA, TB, ShB, StB, TD, ShD, StD, Epi](
    tma: static TiledMma;
    D: var (TensorView[TD, ShD, StD] or Tensor[TD, ShD, StD]);  # the thread's destination fragment view
    A: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA];  # (M, K) problem view, col-major
    B: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB];  # (N, K) problem view, col-major
    epi: Epi;                           # the fused epilogue op (concept), built by the caller
    TileShape: static tuple[M: int, N: int, K: int];
    mCTA, nCTA, threadIdx: int) {.inline.} =   # CTA grid coords + thread id (blockIdx.x/y + threadIdx.x)
  ## One CTA tile of the problem GEMM: the (mCTA, nCTA) tile of
  ## C = f(A·B) over the full (M, N, K) problem.
  ##
  ## The problem dims and the leading strides live on the A/B view types.
  ## Each CTA tile is a local_tile over the problem views:
  ##
  ##   (M, K) view ── local_tile((tileM, tileK), (mCTA, kCTA)) ──▶ (tileM, tileK)
  ##   (N, K) view ── local_tile((tileN, tileK), (nCTA, kCTA)) ──▶ (tileN, tileK)
  ##
  ##   The K dimension is chunked into tileK-deep tiles:
  ##   the problem layout becomes a grid of (tileM × tileK) tiles.
  ##   The coord slices one tile; the offset comes from crd2idx
  ##   (no manual pointer math).
  ## The epilogue op and its destination view (D) are built by the caller
  ## (the kernel wrapper): the op carries its own state
  ## (alpha/beta, C, bias), the grid never sees it.
  ##
  ## Args:
  ##   tma: the TiledMma (see gemm_tiled)
  ##   D: the thread's destination fragment view (partition_C of the C
  ##        tile), written by the epilogue. Same view the op was built
  ##        with
  ##   A: the (M, K) problem view (col-major gmem, compact: see
  ##        Preconditions). M and K come from its shape type
  ##        (M = ShA[0], K = ShA[1])
  ##   B: the (N, K) problem view (col-major gmem, compact: see
  ##        Preconditions). N comes from its shape type (N = ShB[0])
  ##   epi: the fused epilogue op (EpiAXPBY, EpiIdentity, EpiReLU,
  ##        EpiAddBias or a user op), built by the caller with its state
  ##   TileShape: static (tileM, tileN, tileK), see gemm_tiled. tileK is
  ##        the k-tile depth; the problem K is a multiple of it
  ##        (gemm_cta loops K div tileK k-tiles)
  ##   mCTA, nCTA: the CTA grid coordinates, blockIdx.x/y. The tile
  ##        origin is m0 = mCTA·tileM, n0 = nCTA·tileN
  ##   threadIdx: the flat linear thread id in 0 ..< blockSize
  ##
  ## Preconditions:
  ##   - M mod tileM == 0 and N mod tileN == 0, the v1 divisibility
  ##     contract. Ragged tiles are a documented TODO
  ##   - K mod tileK == 0: the problem K is a multiple of the k-tile
  ##     depth, so the CTA tile (which spans the full K) splits into an
  ##     integer number of k-tiles
  ##   - the CTA grid covers 0 ..< M div tileM by 0 ..< N div tileN, so
  ##     m0 + tileM <= M and n0 + tileN <= N (the launcher's contract)
  ##   - the A view is compact: StA.default[1] === M, the B view compact:
  ##     StB.default[1] === N. The epilogue's C leading dim is M
  ##   - tileK mod (thrK·atomK) == 0, enforced by gemm_tiled
  ##
  ## Postconditions:
  ##   - the (mCTA, nCTA) tile of the destination := the epilogue of the
  ##     accumulated tile
  ##   - A and B are unmodified
  # The problem dims live on the view types: M and K from the A view's
  # shape type, N from the B view's (the (M, K) and (N, K) problem
  # layouts). toIntVal coerces the Int[N] leaves to plain ints for the
  # asserts below (this codebase has no implicit static-int coercion for ==)
  const M = toIntVal(ShA.default[0])
  const K = toIntVal(ShA.default[1])
  const N = toIntVal(ShB.default[0])
  const
    tileM = TileShape[0]
    tileN = TileShape[1]
    tileK = TileShape[2]      # the k-tile depth
    kTiles = K div tileK      # the k-tile count
  static:
    doAssert M mod tileM == 0,
      "gemm_cta: M (" & $M & ") mod tileM (" & $tileM & ") != 0. v1 requires tile-aligned" &
      " problem dims, ragged tiles are a documented TODO"
    doAssert N mod tileN == 0,
      "gemm_cta: N (" & $N & ") mod tileN (" & $tileN & ") != 0. v1 requires tile-aligned" &
      " problem dims, ragged tiles are a documented TODO"
    doAssert K mod tileK == 0,
      "gemm_cta: problem K (" & $K & ") mod k-tile depth (" & $tileK & ") != 0. The CTA" &
      " tile spans the whole problem K; gemm_cta slices it into K div tileK k-tiles," &
      " so the problem K must be a multiple of tileK"
    doAssert ShA.default[1] === ShB.default[1],
      "gemm_cta: the A and B problem views must agree on K (" & $K & " vs " &
      $toIntVal(ShB.default[1]) & "). The k-tile grid is sliced with K from the A view"
    doAssert StA.default[1] === ShA.default[0],
      "gemm_cta: the A view's leading stride (" & $toIntVal(StA.default[1]) & ") != M (" &
      $M & "). v1 requires compact (M, K) problem views"
    doAssert StB.default[1] === ShB.default[0],
      "gemm_cta: the B view's leading stride (" & $toIntVal(StB.default[1]) & ") != N (" &
      $N & "). v1 requires compact (N, K) problem views"
    doAssert Epi is Epilogue,
      "gemm_cta: the epilogue op must satisfy the Epilogue concept (preflight + 2-arg apply)"
    doAssert K > 0,
      "gemm_cta: problem K (" & $K & ") must be positive. K = 0 would give zero k-tiles"

  # Accumulator
  # -----------
  # zeroed once, persists across k-tiles
  var dFrag = make_tensor(TD, D.layout.shape)
  dFrag.fillWith(TD(0))

  # Compute loop
  # ------------
  # Each iteration pulls one k-tile view out of the problem views:
  #
  #   (M, K) view ── local_tile((tileM, tileK), (mCTA, kCTA)) ──▶ (tileM, tileK)
  #   (N, K) view ── local_tile((tileN, tileK), (nCTA, kCTA)) ──▶ (tileN, tileK)
  #                              │                                       │
  #                              ▼                                       ▼
  #              K chunked into tileK-deep tiles              gemm_tiled(dFrag, ...)
  #                  (a grid of tiles over M × K)
  #
  #   The coord slices one tile, the offset comes from crd2idx.
  #   The divide is compile-time,
  #   so only the tile offset is computed per iteration.
  for kCTA in 0 ..< kTiles:
    let tA = local_tile(A, (tileM, tileK), (mCTA, kCTA))
    let tB = local_tile(B, (tileN, tileK), (nCTA, kCTA))
    tma.gemm_tiled(dFrag, tA, tB, TileShape, threadIdx)

  # Fused epilogue
  # --------------
  var o = epi
  o.preflight()
  o.apply(D, dFrag)
