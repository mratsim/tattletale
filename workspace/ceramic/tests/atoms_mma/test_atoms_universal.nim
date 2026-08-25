## Host-side test: universal FMA atoms (UNIVERSAL_8x8x8_F32F32F32F32
## and UNIVERSAL_1x1x1_F32F32F32F32).
## No GPU: the 8×8×8 atom's layout algebra (T=32, V=2, the shared AC/B
## layouts), the 1×1×1 scalar gemm_atom's plain-arithmetic branch, and
## the dtype selector's scalar fallback.
##
## Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/test_atoms_universal.nim \
##     --nimcache:nimcache/tests/test_atoms_universal.nim \
##     workspace/ceramic/tests/atoms_mma/test_atoms_universal.nim

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/hardware/h_configgen
import workspace/ceramic/src/hardware/h_registry
import workspace/ceramic/src/hardware/h_properties
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/crucible
import workspace/ceramic/tests/layouts_testutils

{.experimental: "callOperator".}

const atom = UNIVERSAL_8x8x8_F32F32F32F32
  ## The universal 8×8×8 software-mma atom: 32 lanes, 2 values per
  ## operand, A and C on the AC layout, B on the B layout.
const tma = TiledMma[typeof(atom), typeof(make_layout((1, 1, 1)))](
  atom: atom, threadLayout: make_layout((1, 1, 1)))

var dummyBuf = newSeq[float32](64)
let dummyPtr = cast[ptr UncheckedArray[float32]](addr dummyBuf[0])

#  1. Layout algebra

proc runLayoutAlgebraTests =
  block:
    check atom.threadCount(opA), 32, Int
    check atom.threadCount(opB), 32, Int
    check atom.threadCount(opC), 32, Int
    check atom.valuesPerThread(opA), 2, Int
    check atom.valuesPerThread(opB), 2, Int
    check atom.valuesPerThread(opC), 2, Int
    # A and C share the AC fragment layout. B has its own layout.
    # The universal atoms carry their own layout consts, pinned here.
    static:
      doAssert atom.getLayoutA() === Universal8x8_AC_Layout,
        "universal A layout must be the AC layout"
      doAssert atom.getLayoutC() === Universal8x8_AC_Layout,
        "universal C layout must be the AC layout"
      doAssert atom.getLayoutB() === Universal8x8_B_Layout,
        "universal B layout must be the B layout"

  block:
    # One 8×8×8 atom per 32-lane threadgroup: the (1, 1, 1) tiling's
    # fragment of an 8×8 operand is the atom's whole 64-value fragment.
    let tileL = make_layout((8, 8))
    doAssert cosize(tma.thrfrg_A(tileL)) === 64, "A fragment cosize"
    doAssert cosize(tma.thrfrg_B(tileL)) === 64, "B fragment cosize"
    doAssert cosize(tma.thrfrg_C(tileL)) === 64, "C fragment cosize"

  block:
    dummyBuf[0] = 42.0'f32
    let thr = tma.get_slice(0)
    let tAv = tma.partition_A(thr, make_view(dummyPtr, make_layout((8, 8))))
    let tBv = tma.partition_B(thr, make_view(dummyPtr, make_layout((8, 8))))
    var tCv = tma.partition_C(thr, make_view(dummyPtr, make_layout((8, 8))))
    doAssert size(tAv.layout) === 2, "A partition size (V=2 per lane)"
    doAssert size(tBv.layout) === 2, "B partition size (V=2 per lane)"
    doAssert size(tCv.layout) === 2, "C partition size (V=2 per lane)"
    # Lane 0's A fragment reads the (0, 0) tile element.
    doAssert tAv(0) == 42.0'f32, "A fragment reads the (0,0) tile element"
    doAssert tBv(0) == 42.0'f32, "B fragment reads the (0,0) tile element"
    doAssert tCv(0) == 42.0'f32, "C fragment reads the (0,0) tile element"

  echo "  1. Layout algebra (8×8×8, T=32, V=2): 3 blocks OK"

#  2. Numerics (the scalar gemm_atom path)

proc checkFma(a, b, d0: float32): bool =
  ## One gemm_atom call on the scalar atom: dFrag[0] = a·b + dFrag[0].
  ## The scalar path's bk_FMA branch is one FMA. The tile layer's
  ## 8×8×8 universal atom runs its shuffle mma on-device instead.
  var dFrag = make_tensor(float32, (1,))
  var aFrag = make_tensor(float32, (1,))
  var bFrag = make_tensor(float32, (1,))
  dFrag[0] = d0
  aFrag[0] = a
  bFrag[0] = b
  gemm_atom(UNIVERSAL_1x1x1_F32F32F32F32, dFrag, aFrag, bFrag)
  result = dFrag[0] == a * b + d0

proc runNumericsTests =
  block:  # exact plain-arithmetic results (f32-representable)
    doAssert checkFma(-2.0'f32, 3.0'f32, 5.0'f32), "(-2)·3+5 == -1 (negative product)"
    doAssert checkFma(0.5'f32, 0.25'f32, -1.0'f32), "0.5·0.25-1 == -0.875"
    doAssert checkFma(0.0'f32, 1e30'f32, 0.0'f32), "0·1e30 == 0"

  block:
    var dFrag = make_tensor(float32, (1,))
    var aFrag = make_tensor(float32, (1,))
    var bFrag = make_tensor(float32, (1,))
    dFrag[0] = 1.0'f32
    aFrag[0] = 2.0'f32
    bFrag[0] = 3.0'f32
    gemm_atom(UNIVERSAL_1x1x1_F32F32F32F32, dFrag, aFrag, bFrag)
    aFrag[0] = -4.0'f32
    bFrag[0] = 0.5'f32
    gemm_atom(UNIVERSAL_1x1x1_F32F32F32F32, dFrag, aFrag, bFrag)
    doAssert dFrag[0] == 5.0'f32, "1 + 2·3 + (-4)·0.5 == 5"

  echo "  2. Numerics (scalar gemm_atom): 2 blocks OK"

#  3. Selector

proc runSelectorTests =
  block:
    doAssert atom_selector(float32, float32, float32) == UNIVERSAL_1x1x1_F32F32F32F32,
      "f32 selector must resolve to the 1×1×1 universal atom"
    doAssert $atom_selector(float32, float32, float32) == "UNIVERSAL_1x1x1_F32F32F32F32",
      "f32 selector must stringify to the 1×1×1 universal atom's name"

  block:
    doAssert atom_selector(uint32, uint32, float32) == SM80_16x8x8_F32TF32TF32F32_TN,
      "tf32 selector must resolve to the SM80 atom"
    doAssert $atom_selector(uint32, uint32, float32) == "SM80_16x8x8_F32TF32TF32F32_TN",
      "tf32 selector must stringify to the SM80 atom's name"

  echo "  3. Selector: 2 blocks OK"

proc runTests =
  runLayoutAlgebraTests()
  runNumericsTests()
  runSelectorTests()
  echo "\nALL TESTS PASSED"

when isMainModule:
  runTests()
