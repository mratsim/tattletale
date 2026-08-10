## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## GEMM fragment operations and the GEBB microkernel.
##
## `gemm_fragment` — the fragment-level GEMM, three forms:
##   * the MMA (4-arg): one instruction on the thread's register fragments,
##     in-place accumulate; the atom's `instr` + fragment counts + register
##     types produce the inline-asm statement (NVIDIA mma.sync via
##     kernel_gemm/nvidia_tensor_cores.nim; AMD MFMA and Intel AMX construct
##     their instructions differently). No kc loop, no epilogue — the
##     caller owns those.
##   * the MMA (5-arg): same, with an explicit destination
##   * the reference (3-arg): the naive whole-tile outer product, the
##     correctness oracle the GPU kernels are tested against
##
## `gemm_ukernel` — the GEBB microkernel: the loop over K on top of
##   `gemm_fragment` (one atom instruction per k-slice, accumulated in
##   cFrag). Atom-parametric — the same signature serves GPU tensor cores
##   and CPU FMA/AMX atoms.
##
## `gemm_tiled` — one tile of C = α·(A·B) + β·C: thread partitioning,
##   the k-block loop over `gemm_ukernel`, and the fused axpby epilogue.
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

#  gemm_fragment(instr, ...) — register-level MMA
# ═════════════════════════════════════════════════════════════════════════

macro gemm_fragment*[VA: static int, VB: static int, VC: static int, TA, TB, TC](
    instr: static string;
    cFrag: var array[VC, TC];
    aFrag: array[VA, TA];
    bFrag: array[VB, TB]): untyped =
  ## Register-level MMA, in-place accumulate: cFrag += aFrag·bFrag.
  ##
  ## Args:
  ##   instr: the atom's mnemonic, read by field at the call site:
  ##     gemm_fragment(atom.instr, cFrag, aFrag, bFrag)
  ##   cFrag: var register array (the accumulator, also the asm output)
  ##   aFrag, bFrag: register arrays (the operands)
  ##
  ## The inline-asm statement is built here from `instr` + the fragment
  ## counts + register types; D and C alias in the asm (the output registers
  ## ARE the C operand — hardware in-place accumulate).
  let cElem = cFrag.getTypeInst()[2].repr
  let aElem = aFrag.getTypeInst()[2].repr
  let bElem = bFrag.getTypeInst()[2].repr
  let asmStr = buildNvidiaMmaAsm(instr, VA, VB, VC,
                                 cFrag.repr, aFrag.repr, bFrag.repr, cFrag.repr,
                                 cElem, aElem, bElem, cElem)
  result = newTree(nnkAsmStmt, newEmptyNode(), newLit(asmStr))

macro gemm_fragment*[VD: static int, VA: static int, VB: static int, VC: static int, TD, TA, TB, TC](
    instr: static string;
    dFrag: var array[VD, TD];
    aFrag: array[VA, TA];
    bFrag: array[VB, TB];
    cFrag: array[VC, TC]): untyped =
  ## Register-level MMA with explicit output: dFrag = aFrag·bFrag + cFrag.
  ##
  ## Args:
  ##   instr: the atom's mnemonic, read by field at the call site:
  ##     gemm_fragment(atom.instr, dFrag, aFrag, bFrag, cFrag)
  ##   dFrag: var register array — the destination (output first, per §1)
  ##   aFrag, bFrag: register arrays (the operands)
  ##   cFrag: register array — the accumulator input (read-only here)
  ##
  ## dFrag and cFrag may be the same array (passing the same name aliases
  ## them in the asm — equivalent to the 4-arg in-place form).
  if VD != VC:
    error("gemm_fragment: D and C fragment arrays must have the same length (" & $VD & " vs " & $VC & ")", dFrag)
  let dElem = dFrag.getTypeInst()[2].repr
  let aElem = aFrag.getTypeInst()[2].repr
  let bElem = bFrag.getTypeInst()[2].repr
  let cElem = cFrag.getTypeInst()[2].repr
  let asmStr = buildNvidiaMmaAsm(instr, VA, VB, VC,
                                 dFrag.repr, aFrag.repr, bFrag.repr, cFrag.repr,
                                 dElem, aElem, bElem, cElem)
  result = newTree(nnkAsmStmt, newEmptyNode(), newLit(asmStr))

# ═════════════════════════════════════════════════════════════════════════
#  gemm_ukernel(mma, ...) — the GEBB microkernel (loop over K)
# ═════════════════════════════════════════════════════════════════════════

func gemm_ukernel*[VC: static int, TA, TB, TC, ShA, StA, ShB, StB, LA, LB, LC](
    mma: static MmaAtom[LA, LB, LC];
    cFrag: var array[VC, TC];
    aFrag: Tensor[TA, ShA, StA];
    bFrag: Tensor[TB, ShB, StB]) {.inline.} =
  ## GEBB microkernel: cFrag += Σ_k aFrag[k]·bFrag[k], one gemm_fragment
  ## per k-slice, accumulated in cFrag. The loop over K is the layer above
  ## the single-instruction gemm_fragment (CuTe dispatch [5] analog).
  ##
  ## Args:
  ##   mma: a compile-time MmaAtom (bkGPU_TensorCore / bkCPU_X86_AMX — has
  ##        `instr`), passed `static`: the atom is data that monomorphizes
  ##        the kernel (CuTe passes the atom as a type for the same reason)
  ##   cFrag: var register array — the accumulator across all k-slices
  ##   aFrag: owning tensor, shape (K, VA), row-major strides (VA, 1) —
  ##        the K k-slices of A fragments (k as the outer mode, CuTe's
  ##        (V, K) register fragment); the make_tensor staging — no
  ##        raw-addr views at the call site
  ##   bFrag: owning tensor, shape (K, VB), row-major strides (VB, 1) —
  ##        the K k-slices of B fragments
  ##
  ## K = number of k-slices (each of the atom's K depth), read from the
  ## tensor shape along with VA/VB. Each k-slice is copied into a local
  ## register array before gemm_fragment — the unrolled (staticFor) copy
  ## uses constant indices so the asm operands stay register-resident (a
  ## runtime k would spill aFrag[k][i] to local memory and break the "f"/
  ## "r" constraints). The data array is read physically (data[k·VA+i]),
  ## which matches the default column-major fragment layout:
  ## make_tensor(T, (K, VA)) — copyFrom's layout-aware dst(i) is the
  ## identity there, so it fills data linearly in k-slice order.
  ## Atom-parametric: the same signature serves GPU tensor-core atoms and
  ## CPU FMA/AMX atoms (the atom decides the per-slice instruction).
  const
    K = toIntVal(ShA.default[0])
    VA = toIntVal(ShA.default[1])
    VB = toIntVal(ShB.default[1])
  static:
    doAssert toIntVal(ShB.default[0]) == K,
      "gemm_ukernel: B k-slice count (" & $toIntVal(ShB.default[0]) &
        ") != A k-slice count (" & $K & ")"
    doAssert VA == toIntVal(mma.valuesPerThread(opA)),
      "gemm_ukernel: A fragment width (" & $VA & ") != atom valuesPerThread(opA)"
    doAssert VB == toIntVal(mma.valuesPerThread(opB)),
      "gemm_ukernel: B fragment width (" & $VB & ") != atom valuesPerThread(opB)"
    doAssert VC == toIntVal(mma.valuesPerThread(opC)),
      "gemm_ukernel: C fragment width (" & $VC & ") != atom valuesPerThread(opC)"
  staticFor k, 0, K:
    var aSlice: array[VA, TA]
    var bSlice: array[VB, TB]
    staticFor i, 0, VA:
      aSlice[i] = aFrag.data[k * VA + i]
    staticFor i, 0, VB:
      bSlice[i] = bFrag.data[k * VB + i]
    gemm_fragment(mma.instr, cFrag, aSlice, bSlice)


#  gemm_fragment(C, A, B) — reference (whole-tile outer product)
# ═════════════════════════════════════════════════════════════════════════

template gemm_fragment*[T, ShA, StA, ShB, StB, ShC, StC](
    C: var (TensorView[T, ShC, StC] or Tensor[T, ShC, StC]),
    A: TensorView[T, ShA, StA] or Tensor[T, ShA, StA],
    B: TensorView[T, ShB, StB] or Tensor[T, ShB, StB]) =
  ## Reference fragment gemm: C[m,n] += A[m,k] * B[n,k] (outer product).
  ## Correctness oracle for the GPU kernels — not a performance kernel.
  const
    M = ShC.default[0]
    N = ShC.default[1]
    K = ShA.default[1]
  when typeof(ShA.default[0]) isnot typeof(M):
    {.error: "gemm_fragment: A mode 0 (M) != C mode 0".}
  when typeof(ShB.default[0]) isnot typeof(N):
    {.error: "gemm_fragment: B mode 0 (N) != C mode 1".}
  when typeof(ShA.default[1]) isnot typeof(K):
    {.error: "gemm_fragment: A mode 1 (K) != B mode 1".}
  for k in 0 ..< K:
    for m in 0 ..< M:
      for n in 0 ..< N:
        C[m, n] += A[m, k] * B[n, k]

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
  ## k-blocks; each block is staged gmem → registers (aFragBlock /
  ## bFragBlock) and accumulated into a zero-cleared internal cFrag via
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
  ##   - K mod BLK_K != 0 — k-blocks do not divide K; use a BLK_K that
  ##     divides K
  ##   - BLK_K mod (thrK·atomK) != 0 — the k-block is not a multiple of
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
    slicesPerBlock = BLK_K div atomK

  static:
    doAssert BLK_K mod (thrK * atomK) == 0,
      "gemm_tiled: BLK_K (" & $BLK_K & ") mod (thrK·atomK) (" & $thrK & "·" & $atomK &
        ") != 0 — use a BLK_K multiple of thrK·atomK"
    doAssert K mod BLK_K == 0,
      "gemm_tiled: K (" & $K & ") mod BLK_K (" & $BLK_K &
        ") != 0 — use a BLK_K that divides K"
    doAssert thrK == 1,
      "gemm_tiled: ThrK (" & $thrK & ") != 1 — v1 does not distribute threads along K"
    doAssert toIntVal(ShA.default[0]) == BLK_M,
      "gemm_tiled: A shape M (" & $toIntVal(ShA.default[0]) & ") != BLK_M (" & $BLK_M &
        ") — pass a (BLK_M, K) view"
    doAssert toIntVal(ShA.default[1]) == K,
      "gemm_tiled: A shape K (" & $toIntVal(ShA.default[1]) & ") != K (" & $K &
        ") — pass a (BLK_M, K) view"
    doAssert toIntVal(ShB.default[0]) == BLK_N,
      "gemm_tiled: B shape N (" & $toIntVal(ShB.default[0]) & ") != BLK_N (" & $BLK_N &
        ") — pass a (BLK_N, K) view"
    doAssert toIntVal(ShB.default[1]) == K,
      "gemm_tiled: B shape K (" & $toIntVal(ShB.default[1]) & ") != K (" & $K &
        ") — pass a (BLK_N, K) view"
    doAssert toIntVal(ShC.default[0]) == BLK_M,
      "gemm_tiled: C shape M (" & $toIntVal(ShC.default[0]) & ") != BLK_M (" & $BLK_M &
        ") — pass a (BLK_M, BLK_N) view"
    doAssert toIntVal(ShC.default[1]) == BLK_N,
      "gemm_tiled: C shape N (" & $toIntVal(ShC.default[1]) & ") != BLK_N (" & $BLK_N &
        ") — pass a (BLK_M, BLK_N) view"
  let thr = tma.get_slice(threadIdx)
  # The thread's operand views (CuTe: thr_mma.partition_A/B/C):
  #   tAv = (V, RestM, RestK) — my A fragment in gmem, offset inside
  #   tBv = (V, RestN, RestK) — my B fragment in gmem
  #   tCv = (V, RestM, RestN) — my C in gmem (epilogue)
  let tAv = tma.partition_A(thr, A)
  let tBv = tma.partition_B(thr, B)
  var tCv = tma.partition_C(thr, C)

  var cFrag: array[VC, TC]
  # crucible emits bare `float cFrag[4];` (no auto-zero) — explicit zeroing
  var cFragView = make_view(addr cFrag[0], make_layout((Int[VC](),)))
  fillWith(cFragView, 0.0'f32)

  for kb in 0 ..< K div BLK_K:
    var aFragBlock = make_tensor(TA, (slicesPerBlock, VA))
    for s in 0 ..< slicesPerBlock:
      for v in 0 ..< VA:
        # (v0, v1, 0, kb·slicesPerBlock+s) — the decomposed V coord +
        # (RestM, RestK) coords against the flat (V·, RestM, RestK) view
        aFragBlock.data[s * VA + v] = tAv(concat(idx2crd(tma.atom.aLayout.shape[1], v), (0, kb * slicesPerBlock + s)))
    var bFragBlock = make_tensor(TB, (slicesPerBlock, VB))
    for s in 0 ..< slicesPerBlock:
      for v in 0 ..< VB:
        bFragBlock.data[s * VB + v] = tBv(concat(idx2crd(tma.atom.bLayout.shape[1], v), (0, kb * slicesPerBlock + s)))
    gemm_ukernel(tma.atom, cFrag, aFragBlock, bFragBlock)

  # Epilogue — CuTe gemm_device: axpby(alpha, tCrC, beta, tCgC). The
  # register fragment (identity view) and the thread's C view are zipped
  # by size; axpby's β=0 branch skips the C read (a NaN-prefilled C stays
  # untouched).
  axpby(alpha, cFragView, beta, tCv)
