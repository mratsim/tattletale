## Test: the Epilogue concept + the four shipped epilogues.
##
## Validates the PoC from GEMM-ARCHITECTURE §4 Level 3 (Epilogue concept):
##   * concept conformance — each shipped op satisfies Epilogue
##     (compile-time, via `is Epilogue`)
##   * generic dispatch — `applyEpilogue(op: Epilogue, ...)` binds each op
##     and calls its `apply` (Nim concepts V2, works on 2.2.10)
##   * the math — D = α·AB + β·C, identity, bias column broadcast, ReLU
##   * dispatch hoisting — EpiAXPBY β=0 never reads C (NaN-prefilled C
##     proves it), α=1 skips the multiply
##   * layout robustness — row-major D/AB/C, strided C (row padding),
##     mixed layouts (each tensor indexed through its own layout)
##
## Loads (getLoads / LoadKind) and capability flags are the NEXT step —
## this file pins the math-only PoC.

import std/math
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/kernel_gemm_epilogues

{.experimental: "callOperator".}

template test(label: string; body: untyped) =
  block:
    body
  echo "  [OK] ", label

# The generic dispatcher — PoC validation that the concept binds and
# dispatches. A template, with `auto` view params: concept-typed view
# params bind to ONE fixed instantiation (Nim V2 "first acceptable
# candidate" — a rank-1 bias C would not fit a rank-2 binding), but
# `auto` views + a concept-constrained op re-check conformance per call
# site with the actual shapes. gemm_tiled will call op.apply directly,
# statically, the same way.
template applyEpilogue(op: Epilogue; D, AB, C: auto): untyped =
  op.apply(D, AB, C)

proc runEpilogueTests =

  # ── concept conformance (compile-time) ──
  static:
    doAssert EpiAXPBY[float32] is Epilogue, "EpiAXPBY must satisfy Epilogue"
    doAssert EpiIdentity is Epilogue, "EpiIdentity must satisfy Epilogue"
    doAssert EpiAddBias is Epilogue, "EpiAddBias must satisfy Epilogue"
    doAssert EpiReLU is Epilogue, "EpiReLU must satisfy Epilogue"

  test "all four epilogues conform to the Epilogue concept (static)":
    discard

  # ── EpiAXPBY ──
  test "EpiAXPBY D = α·AB + β·C (generic concept dispatch)":
    const M = 2; const N = 3
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = float32(10 + i)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = EpiAXPBY[float32](alpha: 2.0'f32, beta: 3.0'f32)
    applyEpilogue(op, D, AB, C)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j] + 3.0'f32 * C[i, j]

  test "EpiAXPBY method-call syntax (op.apply) matches generic dispatch":
    const M = 2; const N = 3
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    var bufD2 = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = float32(10 + i)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    var D2 = make_view(bufD2 +% 0, make_layout((M, N), (1, M)))
    let op = EpiAXPBY[float32](alpha: 0.5'f32, beta: 2.0'f32)
    op.apply(D, AB, C)
    applyEpilogue(op, D2, AB, C)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == D2[i, j]

  test "EpiAXPBY β=0 never reads C (NaN-prefilled C stays untouched)":
    const M = 2; const N = 3
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = NaN
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = EpiAXPBY[float32](alpha: 2.0'f32, beta: 0.0'f32)
    applyEpilogue(op, D, AB, C)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j]
        doAssert not D[i, j].isNaN

  test "EpiAXPBY α=1 skips the multiply (β=2 path)":
    const M = 2; const N = 3
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = float32(10 + i)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = EpiAXPBY[float32](alpha: 1.0'f32, beta: 2.0'f32)
    applyEpilogue(op, D, AB, C)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j] + 2.0'f32 * C[i, j]

  # ── EpiIdentity ──
  test "EpiIdentity D = AB, C ignored (NaN-prefilled C)":
    const M = 2; const N = 3
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = NaN
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = EpiIdentity()
    applyEpilogue(op, D, AB, C)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j]
        doAssert not D[i, j].isNaN

  # ── EpiAddBias ──
  test "EpiAddBias D = AB + bias column broadcast over rows":
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i)
    for j in 0 ..< N: bufC[j] = float32(100 + j)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 0, make_layout((N,), (1,)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = EpiAddBias()
    applyEpilogue(op, D, AB, C)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j] + C[j]

  # ── EpiReLU ──
  test "EpiReLU D = max(0, AB), C ignored (NaN-prefilled C)":
    const M = 2; const N = 3
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(int(i) - 2)   # -2, -1, 0, 1, 2, 3
    for i in 0 ..< M*N: bufC[i] = NaN
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = EpiReLU()
    applyEpilogue(op, D, AB, C)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == max(AB[i, j], 0.0'f32)
        doAssert not D[i, j].isNaN

  # ── layout robustness: row-major / strided / mixed layouts ──
  # The apply procs index D, AB, C each through their own layout (zip by
  # shape, like axpby) — these pin that the epilogues are not col-major
  # only. The GPU gemm path feeds the accumulator fragment (atom layout)
  # and the gmem C tile (arbitrary stride) into the same apply.

  test "EpiAXPBY row-major D/AB/C (α=2 β=3)":
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = float32(100 + i)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (N, 1)))   # row-major
    let C = make_view(bufC +% 0, make_layout((M, N), (N, 1)))
    var D = make_view(bufD +% 0, make_layout((M, N), (N, 1)))
    let op = EpiAXPBY[float32](alpha: 2.0'f32, beta: 3.0'f32)
    applyEpilogue(op, D, AB, C)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j] + 3.0'f32 * C[i, j]

  test "EpiAXPBY strided C (row padding), col-major D/AB (α=2 β=3)":
    const M = 3; const N = 4; const LDN = 6   # C rows padded to LDN
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * LDN)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*LDN: bufC[i] = float32(i)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 0, make_layout((M, N), (LDN, 1)))   # strided rows
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = EpiAXPBY[float32](alpha: 2.0'f32, beta: 3.0'f32)
    applyEpilogue(op, D, AB, C)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j] + 3.0'f32 * C[i, j]

  test "EpiAXPBY mixed layouts (D row-major, AB col-major, C strided)":
    const M = 3; const N = 4; const LDC = 5
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * LDC)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*LDC: bufC[i] = float32(i)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))       # col-major
    let C = make_view(bufC +% 0, make_layout((M, N), (LDC, 1)))       # strided rows
    var D = make_view(bufD +% 0, make_layout((M, N), (N, 1)))         # row-major
    let op = EpiAXPBY[float32](alpha: 2.0'f32, beta: 3.0'f32)
    applyEpilogue(op, D, AB, C)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j] + 3.0'f32 * C[i, j]

  test "EpiAddBias row-major D/AB with bias column":
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i)
    for j in 0 ..< N: bufC[j] = float32(100 + j)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (N, 1)))   # row-major
    let C = make_view(bufC +% 0, make_layout((N,), (1,)))
    var D = make_view(bufD +% 0, make_layout((M, N), (N, 1)))
    let op = EpiAddBias()
    applyEpilogue(op, D, AB, C)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j] + C[j]

proc main() =
  runEpilogueTests()

when isMainModule:
  main()
