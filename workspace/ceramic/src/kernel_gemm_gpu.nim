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
## The general (tiled) gemm entry is not written yet.
##
## Macro support (atom/array extraction) lives in kernel_gemm/gemm_support.nim.

import std/macros
import ./int_tuples
import ./layouts
import ./tensors
import ./atoms
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

func gemm_ukernel*[K: static int, VA: static int, VB: static int, VC: static int,
               LA, LB, LC, TA, TB, TC](
    mma: static MmaAtom[LA, LB, LC];
    cFrag: var array[VC, TC];
    aFrag: array[K, array[VA, TA]];
    bFrag: array[K, array[VB, TB]]) {.inline.} =
  ## GEBB microkernel: cFrag += Σ_k aFrag[k]·bFrag[k], one gemm_fragment
  ## per k-slice, accumulated in cFrag. The loop over K is the layer above
  ## the single-instruction gemm_fragment (CuTe dispatch [5] analog).
  ##
  ## Args:
  ##   mma: a compile-time MmaAtom (bkGPU_TensorCore / bkCPU_X86_AMX — has
  ##        `instr`), passed `static`: the atom is data that monomorphizes
  ##        the kernel (CuTe passes the atom as a type for the same reason)
  ##   cFrag: var register array — the accumulator across all k-slices
  ##   aFrag: array[K, array[VA, TA]] — the K k-slices of A fragments
  ##   bFrag: array[K, array[VB, TB]] — the K k-slices of B fragments
  ##
  ## K = number of k-slices (each of the atom's K depth). The loop is
  ## unrolled with staticFor so every gemm_fragment sees constant fragment
  ## indices — the asm operands stay register-resident (a runtime k would
  ## spill aFrag[k][i] to local memory and break the "f"/"r" constraints).
  ## Atom-parametric: the same signature serves GPU tensor-core atoms and
  ## CPU FMA/AMX atoms (the atom decides the per-slice instruction).
  staticFor k, 0, K:
    gemm_fragment(mma.instr, cFrag, aFrag[k], bFrag[k])

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
