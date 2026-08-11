## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GEMM fragment operations, the GEBB microkernel, and the tiled GEMM.
##
## `gemm_atom` — the MMA (fragment-level) GEMM: one instruction on the
## thread's register fragments, in-place accumulate. The atom's `instr` +
## fragment counts + register types produce the inline-asm statement
## (NVIDIA mma.sync via kernel_gemm/nvidia_tensor_cores.nim; AMD MFMA and
## Intel AMX construct their instructions differently). No kc loop, no
## epilogue — the caller owns those.
## (the naive whole-tile outer-product reference oracle lives in the
## tests — `gemm_ref` in tests/gemm/gemm_test_lib.nim)
##
## `gemm_ukernel` — the GEBB microkernel: the loop over K on top of
##   `gemm_atom` (one atom instruction per k_block, accumulated in
##   cFrag). Atom-parametric — the same signature serves GPU tensor cores
##   and CPU FMA/AMX atoms.
##
## `gemm_tiled` — one tile of C = α·(A·B) + β·C: thread partitioning,
##   the k-tile loop over `gemm_ukernel`, and the fused axpby epilogue.
##
## Data flow — one CTA tile (BLK_M×BLK_N, K split into BLK_K k-tiles):
##
## ```
##   A (BLK_M, K) gmem          B (BLK_N, K) gmem
##        │ partition_A              │ partition_B
##        ▼                          ▼
##   tAv (V,RestM,RestK)      tBv (V,RestN,RestK)        tCv (V,RestM,RestN)
##        │                          │                     (partition_C of C)
##        └───────────┬──────────────┘
##                    ▼  per k-tile k_tile: view last mode restricted to
##              aFragTile     bFragTile   kBlocksPerTile — the fragment
##              (registers, make_fragment_A/B)  spans THIS k-tile only
##                    │           │
##                    └─────┬─────┘
##                          ▼
##              gemm_ukernel: cFrag += Σ_k aFragTile[k_block]·bFragTile[k_block]
##                          │   per k_block: copy slice → aSlice/bSlice
##                          │     └─► gemm_atom(mma, cFrag, aSlice, bSlice)
##                          ▼
##                    cFrag (VC registers, zero-cleared)
##                          │
##                          ▼  fused epilogue: C := α·cFrag + β·C (axpby)
##                    C (BLK_M, BLK_N) gmem
## ```
##
## Loop nesting:
##
## ```
## gemm_tiled:  for k_tile in 0 ..< K div BLK_K          # k-tile loop
##                gather aFragTile/bFragTile         # gmem → registers
##                gemm_ukernel: for k_block in 0 ..< kBlocksPerTile # k_block loop
##                  copy aFragTile[_, k_block] → aSlice  # fragment slice → operands
##                  copy bFragTile[_, k_block] → bSlice
##                  gemm_atom(mma, cFrag, aSlice, bSlice)  # one mma.sync
## ```
##
## Tile/block hierarchy (CUTLASS terms):
##
## ```
## problem (M, N, K)                                gmem          the whole GEMM
## └─ threadblock tile (BLK_M, BLK_N, BLK_K)        1 CTA         ← gemm_tiled's "tile"
##    ├─ k-tile (BLK_K)                             1 CTA         ← the k_tile loop
##    │   └─ k_block (atomK)                        1 warp        ← one gemm_atom
##    ├─ warp tile (BLK_M/nW, BLK_N/nW)             1 warp
##    │   └─ thread tile / fragment                 1 thread      ← rmem (aFragTile/cFrag)
##    └─ smem stage: one k-tile per pipeline stage
## ```
##
## Memory spaces (CuTe/CUTLASS terms):
##   gmem = global memory (the problem tensors)
##   smem = shared memory (per-CTA staging, one k-tile per pipeline stage)
##          — unused in v1, which reads gmem directly
##   rmem = register memory (the per-thread fragments aFragTile/bFragTile/cFrag)
##
## 1 threadblock tile = K/BLK_K k-tiles; 1 k-tile = kBlocksPerTile k_blocks
## (= CUTLASS K_BLOCK_MAX). "k-slice" is CUTLASS's split-K term — not used here.
##
## NVIDIA mma.sync assembly construction lives in
## kernel_gemm/nvidia_tensor_cores.nim.

import std/macros
import ./int_tuples
import ./layouts
import ./tensors
import ./atoms
import ./atoms_mma_partitioning
import ./layout_algebra
import ./kernel_fillwith_gpu
import ./kernel_axpby_gpu
import ./macros/static_for
import ./kernel_gemm/nvidia_tensor_cores

#  gemm_atom(mma, ...) — register-level MMA
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
  ##   dFrag: var register fragment tensor, the accumulator (also the asm
  ##        output — hardware in-place accumulate, D and C alias in the
  ##        asm)
  ##   aFrag, bFrag: register fragment tensors, the operands
  ##
  ## The language evaluates the tensor arguments once at the call
  ## boundary, so slices like `aFrag(_, _, k)` are passed ergonomically.
  ## gemm_mma (kernel_gemm/nvidia_tensor_cores.nim) builds the asm and
  ## hardcodes float32/uint32 (TF32) — see the TODO there.
  const
    aV = toIntVal(mma.valuesPerThread(opA))
    bV = toIntVal(mma.valuesPerThread(opB))
    dV = toIntVal(mma.valuesPerThread(opC))
  gemm_mma(mma.instr, aV, bV, dV, dFrag, aFrag, bFrag)

# ═════════════════════════════════════════════════════════════════════════
#  gemm_ukernel(mma, ...) — the GEBB microkernel (loop over K)
# ═════════════════════════════════════════════════════════════════════════

func gemm_ukernel*[TC, ShC, StC, TA, ShA, StA, TB, ShB, StB](
    mma: static MmaAtom;
    cFrag: var Tensor[TC, ShC, StC];
    aFrag: Tensor[TA, ShA, StA];
    bFrag: Tensor[TB, ShB, StB]) {.inline.} =
  ## GEBB microkernel: cFrag += Σ_k aFrag[k]·bFrag[k], one gemm_atom
  ## per k_block, accumulated in cFrag. The loop over K is the layer above
  ## the single-instruction gemm_atom — the K-loop dispatch of CuTe's
  ## sgemm_2.cu ukernel.
  ##
  ## Args:
  ##   mma: a compile-time MmaAtom (bkGPU_TensorCore / bkCPU_X86_AMX — has
  ##        `instr`), passed `static`: the atom is data that monomorphizes
  ##        the kernel (CuTe passes the atom as a type for the same reason)
  ##   cFrag: var register fragment tensor — the accumulator across all
  ##        k_blocks (the asm D operand, in-place accumulate)
  ##   aFrag: owning tensor, shape (VA, RestM, RestK), strides (1, VA, VA)
  ##        (make_fragment_A) — V flattened to the atom register order;
  ##        RestM·RestK = the k_blocks; gemm_atom reads data[k_block·VA+i]
  ##        with k_block the flat rest-mode index.
  ##   bFrag: owning tensor, same shape convention (VB, RestN, RestK)
  ##
  ## K = number of k_blocks (each of the atom's K depth), read from the
  ## tensor shape along with VA/VB. Each k_block is copied into a local
  ## register array before gemm_atom — the unrolled (staticFor) copy
  ## uses constant indices so the asm operands stay register-resident (a
  ## runtime k would spill aFrag[k][i] to local memory and break the "f"/
  ## "r" constraints). The data array is read physically (data[k·VA+i]),
  ## which matches the fragment layout's V-first enumeration.
  ## Atom-parametric: the same signature serves GPU tensor-core atoms and
  ## CPU FMA/AMX atoms (the atom decides the per-k_block instruction).
  const
    K = toIntVal(ShA.default[1]) * toIntVal(ShA.default[2])
    VA = toIntVal(ShA.default[0])
    VB = toIntVal(ShB.default[0])
    VC = toIntVal(ShC.default[0])
  static:
    doAssert toIntVal(cosize(aFrag.layout)) div VA === K,
      "gemm_ukernel: A rest-mode size (" & $(toIntVal(cosize(aFrag.layout)) div VA) &
        ") != k_block count (" & $K & ")"
    doAssert toIntVal(cosize(bFrag.layout)) div VB === K,
      "gemm_ukernel: B rest-mode size (" & $(toIntVal(cosize(bFrag.layout)) div VB) &
        ") != k_block count (" & $K & ")"
    doAssert VA === mma.valuesPerThread(opA),
      "gemm_ukernel: A fragment width (" & $VA & ") != atom valuesPerThread(opA)"
    doAssert VB === mma.valuesPerThread(opB),
      "gemm_ukernel: B fragment width (" & $VB & ") != atom valuesPerThread(opB)"
    doAssert VC === mma.valuesPerThread(opC),
      "gemm_ukernel: C fragment width (" & $VC & ") != atom valuesPerThread(opC)"
  staticFor k_block, 0, K:
    var aSlice: array[VA, TA]
    var bSlice: array[VB, TB]
    staticFor i, 0, VA:
      aSlice[i] = aFrag.data[k_block * VA + i]
    staticFor i, 0, VB:
      bSlice[i] = bFrag.data[k_block * VB + i]
    gemm_atom(mma, cFrag,
              make_view(addr aSlice[0], make_layout((VA,), (1,))),
              make_view(addr bSlice[0], make_layout((VB,), (1,))))

# ═════════════════════════════════════════════════════════════════════════
#  gemm_tiled(tma, threadIdx, alpha, A, B, beta, C) — the tiled GEMM
# ═════════════════════════════════════════════════════════════════════════

func gemm_tiled*[TA, TB, TC, T, ShA, StA, ShB, StB, ShC, StC](
    tma: static TiledMma;
    threadIdx: int;
    alpha: T;
    A: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA];
    B: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB];
    beta: T;
    C: var (TensorView[TC, ShC, StC] or Tensor[TC, ShC, StC]);
    BLK_K: static int) {.inline.} =
  ## One tile of C = α·(A·B) + β·C — order follows the formula.
  ##
  ## Args:
  ##   tma: the TiledMma — atom plus (ThrM, ThrN, ThrK) thread tiling
  ##   threadIdx: the flat linear thread id in 0 ..< blockSize (a
  ##        multi-dimensional block must be linearized by the caller)
  ##   alpha, beta: runtime scale factors (float32 in v1)
  ##   A: col-major (BLK_M, K) view, element type TA (tf32 uint32 in v1)
  ##   B: col-major (BLK_N, K) view, element type TB (tf32 uint32 in v1)
  ##   C: col-major (BLK_M, BLK_N) view, element type TC (float32 in v1),
  ##        read and written in place
  ##
  ## Computes the tile C = α·(A·B) + β·C with BLK_M = ThrM·atomM and
  ## BLK_N = ThrN·atomN. The K dimension is split into BLK_K-element
  ## k-tiles; each k-tile is gathered gmem → registers (aFragTile /
  ## bFragTile) and accumulated into a zero-cleared internal cFrag via
  ## gemm_ukernel. A fused epilogue applies α·cFrag + β·C. No smem, no
  ## TMA, no tile-origin logic — the caller bakes the origin into the
  ## view pointers.
  ##
  ## Preconditions:
  ##   - A/B/C are col-major views whose static shapes match the derived
  ##     tile: ShA == (BLK_M, K), ShB == (BLK_N, K), ShC == (BLK_M, BLK_N)
  ##   - K mod BLK_K == 0 and BLK_K mod (thrK·atomK) == 0
  ##   - ThrK == 1 — threads are never distributed along K in v1
  ##   - threadIdx < blockSize
  ##   - the backing buffers must address the tile — the ragged underlying
  ##     allocation is the caller's contract, not checked in v1
  ##   - C's initial contents are read iff beta != 0 — the caller must
  ##     initialize C for beta != 0; C is never read when beta == 0
  ##
  ## Postconditions:
  ##   - C := α·(A·B) + β·C_old elementwise, exact op order α·cFrag + β·C
  ##     (two multiplies then one add — no fma)
  ##   - A and B are unmodified
  ##
  ## Panic-if (expansion-time rejections):
  ##   - the A/B/C view shapes do not match the derived tile
  ##     (BLK_M = ThrM·atomM, BLK_N = ThrN·atomN); fix the views or the
  ##     TiledMma thread layout
  ##   - K mod BLK_K != 0 — k-tiles do not divide K; use a BLK_K that
  ##     divides K
  ##   - BLK_K mod (thrK·atomK) != 0 — the k-tile is not a multiple of
  ##     the thread k-depth; use a BLK_K multiple of thrK·atomK
  ##   - ThrK != 1 — v1 does not distribute threads along K
  ##   - view shape mismatch — pass (BLK_M, K), (BLK_N, K), (BLK_M, BLK_N)
  ##     col-major views
  const
    VA = toIntVal(tma.atom.valuesPerThread(opA))
    VB = toIntVal(tma.atom.valuesPerThread(opB))
    VC = toIntVal(tma.atom.valuesPerThread(opC))
    atomM = tma.atom.mnk.m
    atomN = tma.atom.mnk.n
    atomK = tma.atom.mnk.k
    thrM  = toIntVal(tma.threadLayout.shape[0])
    thrN  = toIntVal(tma.threadLayout.shape[1])
    thrK  = toIntVal(tma.threadLayout.shape[2])
    BLK_M = thrM * atomM
    BLK_N = thrN * atomN
    K = toIntVal(ShA.default[1])
    kBlocksPerTile = BLK_K div atomK

  static:
    doAssert BLK_K mod (thrK * atomK) == 0,
      "gemm_tiled: BLK_K (" & $BLK_K & ") mod (thrK·atomK) (" & $thrK & "·" & $atomK &
        ") != 0 — use a BLK_K multiple of thrK·atomK"
    doAssert K mod BLK_K == 0,
      "gemm_tiled: K (" & $K & ") mod BLK_K (" & $BLK_K &
        ") != 0 — use a BLK_K that divides K"
    doAssert thrK == 1,
      "gemm_tiled: ThrK (" & $thrK & ") != 1 — v1 does not distribute threads along K"
    doAssert ShA.default[0] === BLK_M,
      "gemm_tiled: A shape M (" & $ShA.default[0] & ") != BLK_M (" & $BLK_M &
        ") — pass a (BLK_M, K) view"
    doAssert ShA.default[1] === K,
      "gemm_tiled: A shape K (" & $ShA.default[1] & ") != K (" & $K &
        ") — pass a (BLK_M, K) view"
    doAssert ShB.default[0] === BLK_N,
      "gemm_tiled: B shape N (" & $ShB.default[0] & ") != BLK_N (" & $BLK_N &
        ") — pass a (BLK_N, K) view"
    doAssert ShB.default[1] === K,
      "gemm_tiled: B shape K (" & $ShB.default[1] & ") != K (" & $K &
        ") — pass a (BLK_N, K) view"
    doAssert ShC.default[0] === BLK_M,
      "gemm_tiled: C shape M (" & $ShC.default[0] & ") != BLK_M (" & $BLK_M &
        ") — pass a (BLK_M, BLK_N) view"
    doAssert ShC.default[1] === BLK_N,
      "gemm_tiled: C shape N (" & $ShC.default[1] & ") != BLK_N (" & $BLK_N &
        ") — pass a (BLK_M, BLK_N) view"
  let thr = tma.get_slice(threadIdx)
  # The thread's operand views (CuTe: thr_mma.partition_A/B/C):
  #   tAv = (V, RestM, RestK) — my A fragment in gmem, offset inside
  #   tBv = (V, RestN, RestK) — my B fragment in gmem
  #   tCv = (V, RestM, RestN) — my C in gmem (epilogue)
  let tAv = tma.partition_A(thr, A)
  let tBv = tma.partition_B(thr, B)
  var tCv = tma.partition_C(thr, C)

  var cFrag = make_tensor(TC, (VC,))
  cFrag.fillWith(TC(0))  # TC(0) not 0.0'f32 — the accumulator dtype derives from the atom's cType

  for k_tile in 0 ..< K div BLK_K:
    # Fragment gathering through the layout algebra (CuTe make_fragment_A
    # + copy(A(_,_,k), rA(_,_,k))): the fragment is shaped from the
    # partition view — V flattened to atom register order (stride-1), rest
    # modes compact by the view's order — so the data layout is the
    # hardware register enumeration, decoupled from the operand's strides.
    # The k_block coordinate (0, k_tile·kBlocksPerTile+k_block) indexes the (V, RestM,
    # RestK) view; the flat V index decomposes via the atom's (V0, V1)
    # layout. No bare data[] writes — the fragment's layout is real and
    # coordinate-accessed.
    #
    # The fragment spans THIS k-tile only: the view's last mode (RestK)
    # is restricted to kBlocksPerTile, keeping its stride so the fragment's
    # rest-mode order is unchanged. A full-K fragment (shaped from tAv
    # directly) would make gemm_ukernel — which accumulates every slice of
    # the fragment — read uninitialized registers beyond this k-tile's fill.
    # (CUTLASS sm80_mma_multistage: partition_fragment_A(sA(_,_,0)) — the
    # fragment is partitioned from the k-tile-sliced view, never the full K.)
    let aTileLayout = replaceMode(tAv.layout,
      make_layout(Int[kBlocksPerTile](), Int[toIntVal(mode(tAv.layout, tAv.layout.rank - 1).stride)]()),
      tAv.layout.rank - 1)
    let tAvTile = make_view(tAv.data, aTileLayout)
    var aFragTile = make_fragment_A(tma.atom, tAvTile)
    for k_block in 0 ..< kBlocksPerTile:
      for v in 0 ..< VA:
        # (v0, v1, 0, k_block) — the decomposed V coord + (RestM, RestK) coords
        # against the (V·, RestM, RestK) fragment and partition views
        aFragTile(v, 0, k_block) =
          tAv(concat(idx2crd(tma.atom.aLayout.shape[1], v), (0, k_tile * kBlocksPerTile + k_block)))
    let bTileLayout = replaceMode(tBv.layout,
      make_layout(Int[kBlocksPerTile](), Int[toIntVal(mode(tBv.layout, tBv.layout.rank - 1).stride)]()),
      tBv.layout.rank - 1)
    let tBvTile = make_view(tBv.data, bTileLayout)
    var bFragTile = make_fragment_B(tma.atom, tBvTile)
    for k_block in 0 ..< kBlocksPerTile:
      for v in 0 ..< VB:
        bFragTile(v, 0, k_block) =
          tBv(concat(idx2crd(tma.atom.bLayout.shape[1], v), (0, k_tile * kBlocksPerTile + k_block)))
    gemm_ukernel(tma.atom, cFrag, aFragTile, bFragTile)

  # Epilogue — CuTe gemm_device: axpby(alpha, tCrC, beta, tCgC). The
  # register fragment (identity view) and the thread's C view are zipped
  # by size; axpby's β=0 branch skips the C read (a NaN-prefilled C stays
  # untouched). Free-func call: axpby's parameter order is alpha, X, beta,
  # Y (CuTe mnemonic), so X is not the first arg — UFCS method syntax
  # cannot apply.
  axpby(alpha, cFrag, beta, tCv)
