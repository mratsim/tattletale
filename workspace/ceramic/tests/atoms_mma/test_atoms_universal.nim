## Host-side tests for the universal scalar-FMA atom (UNIVERSAL_FMA_F32).
##
## The 1×1×1 degenerate atom exercises the tiled-GEMM machinery at its
## trivial limit: one thread, one value per operand, and the selector
## fallback for operand dtypes with no backend MMA. No GPU is needed:
## gemm_atom's bk_FMA branch is plain arithmetic, runnable on the host.
##
## References:
##   [CUTE] CUTLASS CuTe `cute/arch/mma.hpp` `UniversalFMA`: the
##          degenerate Shape_MNK (1,1,1) atom this record mirrors.
##   [MOYE] MoYe.jl `src/arch/mma/mma.jl` `UniversalFMA`: `fma!` as plain
##          `d .= a .* b .+ c`.

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm/atoms_nvidia
import workspace/ceramic/src/kernel_gemm/atoms_universal
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/crucible
import workspace/ceramic/tests/layouts_testutils

{.experimental: "callOperator".}

const atom = UNIVERSAL_FMA_F32
  ## The scalar-FMA atom: 1 thread, 1 value per operand, 1×1×1 tile.
const tma = TiledMma[typeof(atom), typeof(make_layout((1, 1, 1)))](
  atom: atom, threadLayout: make_layout((1, 1, 1)))

var dummyBuf = newSeq[float32](4)
let dummyPtr = cast[ptr UncheckedArray[float32]](addr dummyBuf[0])

# ═════════════════════════════════════════════════════════════════════════
#  1. Layout algebra — the trivial (T=1, V=1) limit
# ═════════════════════════════════════════════════════════════════════════

proc runLayoutAlgebraTests =
  block:  # one thread, one value per operand, for every operand
    check atom.threadCount(opA), 1, Int
    check atom.threadCount(opB), 1, Int
    check atom.threadCount(opC), 1, Int
    check atom.valuesPerThread(opA), 1, Int
    check atom.valuesPerThread(opB), 1, Int
    check atom.valuesPerThread(opC), 1, Int

  block:  # thrfrg_* on the (1,1) tile: a single fragment element
    let tileL = make_layout((1, 1))
    doAssert cosize(tma.thrfrg_A(tileL)) === 1, "trivial A fragment"
    doAssert cosize(tma.thrfrg_B(tileL)) === 1, "trivial B fragment"
    doAssert cosize(tma.thrfrg_C(tileL)) === 1, "trivial C fragment"

  block:  # partition_A/B/C: the single value maps to the (0,0) tile element
    #   The (1,1) tile's only element, at coord (0,0), holds a sentinel;
    #   the thread's single fragment value must read exactly it.
    dummyBuf[0] = 42.0'f32
    let thr = tma.get_slice(0)
    let tAv = tma.partition_A(thr, make_view(dummyPtr, make_layout((1, 1))))
    let tBv = tma.partition_B(thr, make_view(dummyPtr, make_layout((1, 1))))
    var tCv = tma.partition_C(thr, make_view(dummyPtr, make_layout((1, 1))))
    doAssert size(tAv.layout) === 1, "A partition size"
    doAssert size(tBv.layout) === 1, "B partition size"
    doAssert size(tCv.layout) === 1, "C partition size"
    doAssert tAv(0) == 42.0'f32, "A fragment reads the (0,0) tile element"
    doAssert tBv(0) == 42.0'f32, "B fragment reads the (0,0) tile element"
    doAssert tCv(0) == 42.0'f32, "C fragment reads the (0,0) tile element"

  echo "  1. Layout algebra (trivial limit): 3 blocks OK"

# ═════════════════════════════════════════════════════════════════════════
#  2. Numerics — gemm_atom on the host, plain arithmetic
# ═════════════════════════════════════════════════════════════════════════

proc checkFma(a, b, d0: float32): bool =
  ## One gemm_atom call: dFrag[0] = a·b + dFrag[0], with dFrag seeded
  ## first (accumulate semantics).
  var dFrag = make_tensor(float32, (1,))
  var aFrag = make_tensor(float32, (1,))
  var bFrag = make_tensor(float32, (1,))
  dFrag[0] = d0
  aFrag[0] = a
  bFrag[0] = b
  gemm_atom(UNIVERSAL_FMA_F32, dFrag, aFrag, bFrag)
  result = dFrag[0] == a * b + d0

proc runNumericsTests =
  block:  # exact plain-arithmetic results (f32-representable)
    doAssert checkFma(2.0'f32, 3.0'f32, 1.0'f32), "2·3+1 == 7"
    doAssert checkFma(-2.0'f32, 3.0'f32, 5.0'f32), "(-2)·3+5 == -1 (negative product)"
    doAssert checkFma(0.5'f32, 0.25'f32, -1.0'f32), "0.5·0.25-1 == -0.875"
    doAssert checkFma(0.0'f32, 1e30'f32, 0.0'f32), "0·1e30 == 0"

  block:  # accumulation across two gemm_atom calls
    var dFrag = make_tensor(float32, (1,))
    var aFrag = make_tensor(float32, (1,))
    var bFrag = make_tensor(float32, (1,))
    dFrag[0] = 1.0'f32
    aFrag[0] = 2.0'f32
    bFrag[0] = 3.0'f32
    gemm_atom(UNIVERSAL_FMA_F32, dFrag, aFrag, bFrag)
    aFrag[0] = -4.0'f32
    bFrag[0] = 0.5'f32
    gemm_atom(UNIVERSAL_FMA_F32, dFrag, aFrag, bFrag)
    doAssert dFrag[0] == 5.0'f32, "1 + 2·3 + (-4)·0.5 == 5"

  echo "  2. Numerics (host gemm_atom): 2 blocks OK"

# ═════════════════════════════════════════════════════════════════════════
#  3. Selector — the f32 fallback and the unchanged tf32 branch
# ═════════════════════════════════════════════════════════════════════════

proc runSelectorTests =
  block:  # plain f32 resolves to the universal atom
    doAssert typeof(atom_selector(float32, float32, float32)) is typeof(UNIVERSAL_FMA_F32),
      "f32 selector must resolve to the scalar-FMA atom"
    doAssert atom_selector(float32, float32, float32).name == "UNIVERSAL_FMA_F32",
      "f32 selector must name the scalar-FMA atom"

  block:  # the tf32 → SM80 branch is unchanged
    doAssert typeof(atom_selector(uint32, uint32, float32)) is
      typeof(SM80_16x8x8_F32TF32TF32F32_TN), "tf32 selector must resolve to the SM80 atom"
    doAssert atom_selector(uint32, uint32, float32).name == "SM80_16x8x8_F32TF32TF32F32_TN",
      "tf32 selector must name the SM80 atom"

  echo "  3. Selector: 2 blocks OK"

# ═════════════════════════════════════════════════════════════════════════
#  Test runner
# ═════════════════════════════════════════════════════════════════════════

proc runTests =
  runLayoutAlgebraTests()
  runNumericsTests()
  runSelectorTests()
  echo "\nALL TESTS PASSED"

when isMainModule:
  runTests()
