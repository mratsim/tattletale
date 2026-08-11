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
## `gemm_tiled`: one tile of the problem GEMM: thread partitioning,
##   fragment gathering, the k_block loop in `gemm_ukernel`, and the
##   fused epilogue (the Epilogue zoo: EpiAXPBY, EpiIdentity, EpiReLU,
##   EpiAddBias).
##
## `gemm_grid`: one CTA tile of the problem GEMM, selected by grid
##   coordinates (blockIdx.x/y). It divides the global A/B layouts into
##   tile grids (flat_divide) and slices the CTA's tile by coordinate,
##   the CUTLASS local_tile pattern, then calls `gemm_tiled`.
##
## Data flow: one CTA tile (tileM×tileN, K = the tile K):
##
## ```
##   A (tileM, K) gmem          B (tileN, K) gmem
##        │ partition_A              │ partition_B
##        ▼                          ▼
##   tAv (V,RestM,RestK)      tBv (V,RestN,RestK)        C (V,RestM,RestN)
##        │                          │          (partition_C of the tile,
##        └───────────┬──────────────┘           the epilogue destination)
##                    ▼
##              aFragTile     bFragTile     : full-K fragments
##              (registers, make_fragment_A/B)  (one copyFrom each)
##                    │           │
##                    └─────┬─────┘
##                          ▼
##              gemm_ukernel: dFrag += Σ_k aFrag(_,_,k)·bFrag(_,_,k)
##                          │   one gemm_atom per k_block
##                          ▼
##                    dFrag (register fragment, zero-cleared)
##                          │
##                          ▼  fused epilogue (the Epilogue zoo):
##                    D = f(AB)                  (preflight stages the
##                    C (tileM, tileN) gmem       op operands: C for AXPBY,
##                                                bias for AddBias)
## ```
##
## Loop nesting:
##
## ```
## gemm_grid:   tile origin from blockIdx, bake into A/B/C views
##                |
## gemm_tiled:  gather aFragTile/bFragTile         # gmem → registers (full K)
##                gemm_ukernel: for k_block in 0 ..< kBlocks  # k_block loop
##                  gemm_atom(mma, dFrag, aFrag(_,_,k), bFrag(_,_,k))  # one mma.sync
## ```
##
## Tile/block hierarchy:
##
## ```
## problem (M, N, K)                                  gmem       the whole GEMM
## └─ threadblock tile (tileM, tileN, tileK)          1 CTA      ← gemm_grid's blockIdx pick, gemm_tiled's tile
##    ├─ warp tile (tileM/nW, tileN/nW)               1 warp
##    │   └─ thread tile / fragment                   1 thread   ← rmem (aFragTile/dFrag)
##    └─ k_block (atomK)                              1 warp     ← one gemm_atom
## ```
##
## Memory spaces:
##   gmem = global memory (the problem tensors)
##   smem = shared memory (per-CTA staging: the epilogue's C operand)
##   rmem = register memory (the per-thread fragments)
##
## 1 threadblock tile = tileK/atomK k_blocks. Split-K is not used.
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

func gemm_ukernel*[TC, ShC, StC, TA, ShA, StA, TB, ShB, StB](
    mma: static MmaAtom;
    dFrag: var Tensor[TC, ShC, StC];
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
    # multi-rest fragments (CUTLASS MmaIterations::kRow/kColumn pattern):
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
#  gemm_tiled(tma, threadIdx, epi, A, B, TileShape): the tiled GEMM
# ═════════════════════════════════════════════════════════════════════════

func gemm_tiled*[TA, ShA, StA, TB, ShB, StB, TC, ShC, StC, Epi](
    tma: static TiledMma;
    D: var (TensorView[TC, ShC, StC] or Tensor[TC, ShC, StC]);
    A: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA];
    B: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB];
    epi: Epi;
    TileShape: static tuple[M: int, N: int, K: int];
    threadIdx: int) {.inline.} =
  ## One tile of the problem GEMM with a fused epilogue.
  ##
  ## Args:
  ##   tma: the TiledMma, atom plus (ThrM, ThrN, ThrK) thread tiling
  ##   D: the thread's destination fragment view (partition_C of the C
  ##        tile), written by the epilogue. AXPBY reads C_old from the
  ##        same gmem through the op's staged copy
  ##   A: col-major (tileM, tileK) view, element type TA (tf32 uint32 in v1)
  ##   B: col-major (tileN, tileK) view, element type TB (tf32 uint32 in v1)
  ##   epi: the fused epilogue op: any type satisfying the Epilogue
  ##        concept (the shipped EpiAXPBY/EpiIdentity/EpiReLU/EpiAddBias
  ##        or a user-defined op), built by the caller. It carries the
  ##        staged operands (C, bias) and applies D = f(AB)
  ##   TileShape: static (tileM, tileN, tileK), the tile dims. tileM and
  ##        tileN must be exactly the thread layout's coverage
  ##        (thrM·atomM, thrN·atomN). tileK is the views' K
  ##   threadIdx: the flat linear thread id in 0 ..< blockSize (a
  ##        multi-dimensional block must be linearized by the caller)
  ##
  ## Computes the tile: the operand fragments span the full tile K and
  ## are gathered gmem → registers once. gemm_ukernel loops the k_blocks
  ## into a zero-cleared accumulator (dFrag, the epilogue's AB). The
  ## fused epilogue stages its operands into smem (preflight: C for
  ## AXPBY, bias for AddBias, nothing for Identity/ReLU) and applies
  ## D = f(AB) (apply). No TMA, no tile-origin logic. The caller bakes
  ## the origin into the view pointers.
  ##
  ## Preconditions:
  ##   - A/B are col-major views with shapes (tileM, tileK), (tileN, tileK)
  ##   - D is the thread's partition_C view of the tile, built by the
  ##     caller. The op's staged C reads the same gmem through its own
  ##     view (op state)
  ##   - tileK mod (thrK·atomK) == 0, ThrK == 1
  ##   - threadIdx < blockSize
  ##   - the backing buffers must address the tile. The ragged underlying
  ##     allocation is the caller's contract, not checked in v1
  ##   - the epilogue's C is read iff beta != 0. The caller must
  ##     initialize C for beta != 0. C is never read when beta == 0
  ##
  ## Postconditions:
  ##   - the tile's destination := the epilogue of the accumulated tile
  ##   - A and B are unmodified
  ##
  ## Panic-if (expansion-time rejections):
  ##   - TileShape.M/N != thrM·atomM / thrN·atomN. The thread layout must
  ##     exactly cover the tile (the partition contract). Fix the TiledMma
  ##     thread layout or the tile
  ##   - tileK mod (thrK·atomK) != 0. The tile K is not a multiple of the
  ##     thread k-depth. Use a tile K multiple of thrK·atomK
  ##   - ThrK != 1. v1 does not distribute threads along K
  ##   - view shape mismatch. Pass (tileM, tileK), (tileN, tileK)
  ##     col-major views
  const
    tileM = TileShape[0]
    tileN = TileShape[1]
    tileK = TileShape[2]
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
        ") != 0. Use a tile K multiple of thrK·atomK"
    doAssert thrK == 1,
      "gemm_tiled: ThrK (" & $thrK & ") != 1. v1 does not distribute threads along K"
    doAssert ShA.default[0] === tileM,
      "gemm_tiled: A shape M (" & $ShA.default[0] & ") != tile M (" & $tileM &
        "). Pass a (tileM, tileK) view"
    doAssert ShA.default[1] === tileK,
      "gemm_tiled: A shape K (" & $ShA.default[1] & ") != tile K (" & $tileK &
        "). Pass a (tileM, tileK) view"
    doAssert ShB.default[0] === tileN,
      "gemm_tiled: B shape N (" & $ShB.default[0] & ") != tile N (" & $tileN &
        "). Pass a (tileN, tileK) view"
    doAssert ShB.default[1] === tileK,
      "gemm_tiled: B shape K (" & $ShB.default[1] & ") != tile K (" & $tileK &
        "). Pass a (tileN, tileK) view"
    doAssert toIntVal(ShC.default[0]) * toIntVal(ShC.default[1]) * toIntVal(ShC.default[2]) === tileM * tileN div blockSize,
      "gemm_tiled: epilogue C fragment size (" & $(toIntVal(ShC.default[0]) * toIntVal(ShC.default[1]) * toIntVal(ShC.default[2])) &
        ") != tile elements per thread (" & $(tileM * tileN div blockSize) & ")"
    doAssert Epi is Epilogue,
      "gemm_tiled: the epilogue op must satisfy the Epilogue concept (preflight + 2-arg apply)"

  let thr = tma.get_slice(threadIdx)
  # The thread's operand views:
  #   tAv = (V, RestM, RestK), my A fragment in gmem
  #   tBv = (V, RestN, RestK), my B fragment in gmem
  let tAv = tma.partition_A(thr, A)
  let tBv = tma.partition_B(thr, B)

  # D/AB naming follows the epilogue:
  #   D   = the destination fragment, this thread's C view (epi.C_gmem)
  #   AB  = the accumulator fragment, dFrag. dFrag shares the partition
  #         view's shape type with column-major compact strides, so its
  #         flat enumeration is the register order and the epilogue's
  #         shared-Sh contract holds.
  var dFrag = make_tensor(TC, ShC.default)
  dFrag.fillWith(TC(0))  # TC(0) not 0.0'f32: the accumulator dtype derives from the atom's cType

  var o = epi
  # the operand fragments span the full tile K. The k_block loop lives
  # in gemm_ukernel
  var aFragTile = make_fragment_A(tma.atom, tAv)
  aFragTile.copyFrom(tAv)
  var bFragTile = make_fragment_B(tma.atom, tBv)
  bFragTile.copyFrom(tBv)
  gemm_ukernel(tma.atom, dFrag, aFragTile, bFragTile)

  # fused epilogue: preflight stages the op's operands (C for AXPBY,
  # bias for AddBias, skipped for Identity/ReLU), apply writes D = f(AB)
  o.preflight()
  o.apply(D, dFrag)

# ═════════════════════════════════════════════════════════════════════════
#  gemm_grid(tma, D, A, ldA, B, ldB, epi, M, N, K, TileShape, mCTA, nCTA, threadIdx)
# ═════════════════════════════════════════════════════════════════════════

func gemm_grid*[TA, TB, TC, ShC, StC, Epi](
    tma: static TiledMma;
    D: var (TensorView[TC, ShC, StC] or Tensor[TC, ShC, StC]);  # the thread's destination fragment view
    A: ptr UncheckedArray[TA]; ldA: static int;   # (M, K) col-major gmem, ldA the leading dim
    B: ptr UncheckedArray[TB]; ldB: static int;   # (N, K) col-major gmem, ldB the leading dim
    epi: Epi;                           # the fused epilogue op (concept), built by the caller
    M, N, K: static int;                # problem dims, compile-time in v1
    TileShape: static tuple[M: int, N: int, K: int];
    mCTA, nCTA, threadIdx: int) {.inline.} =   # CTA grid coords + thread id (blockIdx.x/y + threadIdx.x)
  ## One CTA tile of the problem GEMM: the (mCTA, nCTA) tile of
  ## C = f(A·B) over the full (M, N, K) problem.
  ##
  ## The grid layer owns the problem dims. gemm_grid divides the global
  ## A/B layouts into tile grids (flat_divide) and slices the CTA's tile
  ## by grid coordinate: the offset is resolved by the layout, the
  ## CUTLASS local_tile pattern. The epilogue op and its destination
  ## view (D) are built by the caller (the kernel wrapper): the op
  ## carries its own state (alpha/beta, C, bias), the grid never sees
  ## it.
  ##
  ## Args:
  ##   tma: the TiledMma (see gemm_tiled)
  ##   D: the thread's destination fragment view (partition_C of the C
  ##        tile), written by the epilogue. Same view the op was built
  ##        with
  ##   A: (M, K) col-major gmem. B: (N, K) col-major gmem
  ##   ldA, ldB: the leading dims. v1 asserts ldA = M, ldB = N (compact
  ##        problem buffers)
  ##   epi: the fused epilogue op (EpiAXPBY, EpiIdentity, EpiReLU,
  ##        EpiAddBias or a user op), built by the caller with its state
  ##   M, N, K: the problem dims, compile-time in v1 (partition offsets are
  ##        baked at compile time). K is the problem's K extent (the A/B
  ##        buffers' K)
  ##   TileShape: static (tileM, tileN, tileK), see gemm_tiled. The
  ##        problem K equals tileK in v1 (gemm_tiled has no k-tile loop)
  ##   mCTA, nCTA: the CTA grid coordinates, blockIdx.x/y. The tile
  ##        origin is m0 = mCTA·tileM, n0 = nCTA·tileN
  ##   threadIdx: the flat linear thread id in 0 ..< blockSize
  ##
  ## Preconditions:
  ##   - M mod tileM == 0 and N mod tileN == 0, the v1 divisibility
  ##     contract. Ragged tiles are a documented TODO
  ##   - K == tileK: the CTA tile spans the whole problem K in v1 (no
  ##     k-tile loop; the k-tile loop later slices K per tile)
  ##   - the CTA grid covers 0 ..< M div tileM by 0 ..< N div tileN, so
  ##     m0 + tileM <= M and n0 + tileN <= N (the launcher's contract)
  ##   - ldA = M, ldB = N. The epilogue's C leading dim is M
  ##   - K mod (thrK·atomK) == 0, enforced by gemm_tiled
  ##
  ## Postconditions:
  ##   - the (mCTA, nCTA) tile of the destination := the epilogue of the
  ##     accumulated tile
  ##   - A and B are unmodified
  const
    tileM = TileShape[0]
    tileN = TileShape[1]
    tileK = TileShape[2]
  static:
    doAssert M mod tileM == 0,
      "gemm_grid: M (" & $M & ") mod tileM (" & $tileM & ") != 0. v1 requires tile-aligned" &
      " problem dims, ragged tiles are a documented TODO"
    doAssert N mod tileN == 0,
      "gemm_grid: N (" & $N & ") mod tileN (" & $tileN & ") != 0. v1 requires tile-aligned" &
      " problem dims, ragged tiles are a documented TODO"
    doAssert tileK == K,
      "gemm_grid: TileShape.K (" & $tileK & ") != problem K (" & $K & "). v1 has no" &
      " k-tile loop — the CTA tile spans the whole problem K, so pass TileShape.K == K" &
      " (the k-tile loop later relaxes this)"
    doAssert ldA == M,
      "gemm_grid: ldA (" & $ldA & ") != M (" & $M & "). v1 requires compact (M, K) buffers"
    doAssert ldB == N,
      "gemm_grid: ldB (" & $ldB & ") != N (" & $N & "). v1 requires compact (N, K) buffers"

  # Global views -> tile grids (flat_divide keeps the operand strides),
  # then the CTA's tile by grid coordinate. The tile offset is the
  # layout's crd2idx, not a hand-written pointer offset. The k mode is
  # size 1 in v1 (tileK is the whole K, no k-tile loop), fixed to 0;
  # the k-tile loop later slices it per tile.
  # v1: tileK is the whole problem K (no k-tile loop), so the global
  # shape K equals TileShape.K. The k-tile loop later passes K here.
  let mA = make_view(A, (M, tileK), (1, ldA))
  let mB = make_view(B, (N, tileK), (1, ldB))
  let tAg = make_view(mA.data, flat_divide(mA.layout, (tileM, tileK)))
  let tBg = make_view(mB.data, flat_divide(mB.layout, (tileN, tileK)))
  let tA = tAg(_, _, mCTA, 0)
  let tB = tBg(_, _, nCTA, 0)
  tma.gemm_tiled(D, tA, tB, epi, TileShape, threadIdx)
