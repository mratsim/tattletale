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
## `gemm_tiled`: one tile of C = α·(A·B) + β·C: thread partitioning,
##   fragment gathering, the k_block loop in `gemm_ukernel`, and the
##   fused epilogue (EpiAXPBY).
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
##                          ▼  fused epilogue (EpiAXPBY):
##                    D = α·AB + β·C_smem        (preflight stages C,
##                    C (tileM, tileN) gmem       β==0 skips the load)
## ```
##
## Loop nesting:
##
## ```
## gemm_tiled:  gather aFragTile/bFragTile         # gmem → registers (full K)
##                gemm_ukernel: for k_block in 0 ..< kBlocks  # k_block loop
##                  gemm_atom(mma, dFrag, aFrag(_,_,k), bFrag(_,_,k))  # one mma.sync
## ```
##
## Tile/block hierarchy:
##
## ```
## problem (M, N, K)                                  gmem       the whole GEMM
## └─ threadblock tile (tileM, tileN, tileK)          1 CTA      ← gemm_tiled's "tile"
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
import ./atoms
import ./atoms_mma_partitioning
import ./layout_algebra
import ./kernel_copy_gpu
import ./kernel_fillwith_gpu
import ./kernel_gemm_epilogues
import ./macros/static_for
import ./kernel_gemm/nvidia_tensor_cores

{.experimental: "callOperator".}

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
  ##        k·VA·RestM.
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
    kBlocks = toIntVal(ShA.default[1]) * toIntVal(ShA.default[2])
  static:
    doAssert toIntVal(ShA.default[0]) === VA,
      "gemm_ukernel: A fragment width (" & $toIntVal(ShA.default[0]) & ") != atom valuesPerThread(opA) (" & $VA & ")"
    doAssert toIntVal(ShB.default[0]) === VB,
      "gemm_ukernel: B fragment width (" & $toIntVal(ShB.default[0]) & ") != atom valuesPerThread(opB) (" & $VB & ")"
    doAssert toIntVal(ShA.default[2]) === kBlocks,
      "gemm_ukernel: A k dimension (" & $toIntVal(ShA.default[2]) & ") != k_block count (" & $kBlocks &
        "). The k dimension must be the last fragment mode"
    doAssert toIntVal(ShB.default[2]) === kBlocks,
      "gemm_ukernel: B k dimension (" & $toIntVal(ShB.default[2]) & ") != k_block count (" & $kBlocks &
        "). The k dimension must be the last fragment mode"
    doAssert toIntVal(cosize(dFrag.layout)) === toIntVal(mma.valuesPerThread(opC)),
      "gemm_ukernel: accumulator size (" & $(toIntVal(cosize(dFrag.layout))) &
        ") != atom valuesPerThread(opC) (" & $(toIntVal(mma.valuesPerThread(opC))) & ")"
  staticFor k_block, 0, kBlocks:
    gemm_atom(mma, dFrag, aFrag(_, _, k_block), bFrag(_, _, k_block))

# ═════════════════════════════════════════════════════════════════════════
#  gemm_tiled(tma, threadIdx, epi, A, B, TileShape): the tiled GEMM
# ═════════════════════════════════════════════════════════════════════════

func gemm_tiled*[TA, ShA, StA, TB, ShB, StB, TC, ShC, StC](
    tma: static TiledMma;
    threadIdx: int;
    epi: EpiAXPBY[TC, ShC, StC];
    A: TensorView[TA, ShA, StA] or Tensor[TA, ShA, StA];
    B: TensorView[TB, ShB, StB] or Tensor[TB, ShB, StB];
    TileShape: static tuple[M: int, N: int, K: int]) {.inline.} =
  ## One tile of C = α·(A·B) + β·C. Order follows the formula.
  ##
  ## Args:
  ##   tma: the TiledMma, atom plus (ThrM, ThrN, ThrK) thread tiling
  ##   threadIdx: the flat linear thread id in 0 ..< blockSize (a
  ##        multi-dimensional block must be linearized by the caller)
  ##   epi: the fused epilogue op (EpiAXPBY in v1), built by the caller
  ##        with the thread's C fragment view. It carries alpha/beta and
  ##        the C operand (the destination D)
  ##   A: col-major (tileM, tileK) view, element type TA (tf32 uint32 in v1)
  ##   B: col-major (tileN, tileK) view, element type TB (tf32 uint32 in v1)
  ##   TileShape: static (tileM, tileN, tileK), the tile dims. tileM and
  ##        tileN must be exactly the thread layout's coverage
  ##        (thrM·atomM, thrN·atomN). tileK is the views' K
  ##
  ## Computes the tile C = α·(A·B) + β·C. The operand fragments span the
  ## full tile K and are gathered gmem → registers once. gemm_ukernel
  ## loops the k_blocks into a zero-cleared accumulator (dFrag, the
  ## epilogue's AB). The fused epilogue stages its C operand into smem
  ## (preflight, skipped when beta == 0) and applies D = α·AB + β·C
  ## (apply). No TMA, no tile-origin logic. The caller bakes the origin
  ## into the view pointers.
  ##
  ## Preconditions:
  ##   - A/B are col-major views with shapes (tileM, tileK), (tileN, tileK)
  ##   - the epilogue op holds the thread's partition_C view of the tile
  ##   - tileK mod (thrK·atomK) == 0, ThrK == 1
  ##   - threadIdx < blockSize
  ##   - the backing buffers must address the tile. The ragged underlying
  ##     allocation is the caller's contract, not checked in v1
  ##   - the epilogue's C is read iff beta != 0. The caller must
  ##     initialize C for beta != 0. C is never read when beta == 0
  ##
  ## Postconditions:
  ##   - the tile's C := α·(A·B) + β·C_old elementwise, exact op order
  ##     α·AB + β·C (two multiplies then one add, no fma)
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
    doAssert toIntVal(ShC.default[0]) * toIntVal(ShC.default[1]) === tileM * tileN div blockSize,
      "gemm_tiled: epilogue C fragment size (" & $(toIntVal(ShC.default[0]) * toIntVal(ShC.default[1])) &
        ") != tile elements per thread (" & $(tileM * tileN div blockSize) & ")"

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

  # the operand fragments span the full tile K. The k_block loop lives
  # in gemm_ukernel
  var aFragTile = make_fragment_A(tma.atom, tAv)
  aFragTile.copyFrom(tAv)
  var bFragTile = make_fragment_B(tma.atom, tBv)
  bFragTile.copyFrom(tBv)
  gemm_ukernel(tma.atom, dFrag, aFragTile, bFragTile)

  # fused epilogue: preflight stages the op's C operand into smem
  # (skipped when beta == 0), apply writes D = α·AB + β·C_smem
  var o = epi
  o.preflight()
  o.apply(o.cView(), dFrag)
