## Test: the Epilogue concept + the four shipped epilogues.
##
## Validates the Epilogue concept:
##   * concept conformance: each shipped op satisfies Epilogue
##     (compile-time, via `is Epilogue`)
##   * generic dispatch: `applyEpilogue(op: Epilogue, ...)` binds each op
##     and calls its `apply` (Nim concepts V2, works on 2.2.10)
##   * the math: D = α·AB + β·C, identity, bias column broadcast, ReLU
##   * dispatch hoisting: EpiAXPBY β=0 never reads C (NaN-prefilled C
##     proves it), α=1 skips the multiply
##   * uniform 2-arg `apply`: inputs are op state (EpiAXPBY's C_gmem,
##     EpiAddBias's bias) — the concept takes D and AB only
##   * layout robustness: row-major D/AB/C, strided C (row padding),
##     mixed layouts, sliced tensors (in / out / in+out), inverted
##     (negative) strides, the sm80 m16n8k8 C-fragment layout, rank-3
##     and nested+broadcast layouts. All operands share the shape type
##     Sh (the compiler enforces equal shapes). `apply` iterates `size(D)`
##     and indexes each operand through its own layout
##
## `preflight` stages the op's gmem operands into smem buffers (EpiAddBias's
## bias today; EpiAXPBY's C is read per-thread from gmem in `apply` — direct
## register→gmem, cp.async/TMA staging pending). Async staging and capability
## flags are future work. This file pins the math.

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

# The generic dispatcher: PoC validation that the concept binds and
# dispatches. A template, with `auto` view params. Concept-typed view
# params bind to ONE fixed instantiation (Nim V2 "first acceptable
# candidate"). `auto` views + a concept-constrained op re-check
# conformance per call site with the actual shapes. gemm_tiled will call
# `op.preflight()` then `op.apply(D, AB)`, statically, the same way: the
# staging template injects the smem buffer, `apply` consumes it through
# the op's fields (bias_smem today; EpiAXPBY reads C via C_gmem directly).
template applyEpilogue(op: Epilogue; D, AB: auto): untyped =
  block:
    var o = op
    o.preflight()
    o.apply(D, AB)

proc runEpilogueTests =

  # ── concept conformance (compile-time) ──
  const ConfShp = (Int[2](), Int[3]())
  const ConfStp = (Int[1](), Int[2]())
  static:
    doAssert EpiAXPBY[float32, ConfShp, ConfStp] is Epilogue,
      "EpiAXPBY must satisfy Epilogue"
    doAssert EpiIdentity is Epilogue, "EpiIdentity must satisfy Epilogue"
    doAssert EpiAddBias[float32, ConfShp, ConfStp] is Epilogue,
      "EpiAddBias must satisfy Epilogue"
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
    let op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    applyEpilogue(op, D, AB)
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
    var op = initEpiAXPBY(0.5'f32, 2.0'f32, C)
    op.preflight()
    op.apply(D, AB)
    applyEpilogue(op, D2, AB)
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
    let op = initEpiAXPBY(2.0'f32, 0.0'f32, C)
    applyEpilogue(op, D, AB)
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
    let op = initEpiAXPBY(1.0'f32, 2.0'f32, C)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j] + 2.0'f32 * C[i, j]

  # ── EpiIdentity ──
  test "EpiIdentity D = AB":
    const M = 2; const N = 3
    var bufAB = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = EpiIdentity()
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j]

  # ── EpiAddBias ──
  test "EpiAddBias D = AB + bias column broadcast over rows":
    ## The bias is op state: a same-size stride-0 broadcast view of the
    ## column.
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N)
    var bufBias = newSeq[float32](N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i)
    for j in 0 ..< N: bufBias[j] = float32(100 + j)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    # TODO: remove the pointer-offset arithmetic — the bias view must
    # come from the (N,) buffer's own layout, not `+%` offsets
    let bias = make_view(bufBias +% 0, (M, N), (0, 1))   # stride-0 rows
    let op = initEpiAddBias(bias)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j] + bufBias[j]

  # ── EpiReLU ──
  test "EpiReLU D = max(0, AB)":
    const M = 2; const N = 3
    var bufAB = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(int(i) - 2)   # -2, -1, 0, 1, 2, 3
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = EpiReLU()
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == max(AB[i, j], 0.0'f32)

  # ── layout robustness I: row-major / strided / mixed layouts ──
  # The `apply` procs index D and AB each through their own layout
  # size). These pin that the epilogues are not col-major only. The GPU
  # gemm path feeds the accumulator fragment (atom layout) and the gmem
  # C tile (arbitrary stride, as op state) into the same `apply`.

  test "EpiAXPBY row-major D/AB with row-major C (α=2 β=3)":
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = float32(100 + i)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (N, 1)))   # row-major
    let C = make_view(bufC +% 0, make_layout((M, N), (N, 1)))
    var D = make_view(bufD +% 0, make_layout((M, N), (N, 1)))
    let op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    applyEpilogue(op, D, AB)
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
    let op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    applyEpilogue(op, D, AB)
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
    let op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j] + 3.0'f32 * C[i, j]

  test "EpiAddBias row-major D/AB with bias column":
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N)
    var bufBias = newSeq[float32](N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i)
    for j in 0 ..< N: bufBias[j] = float32(100 + j)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (N, 1)))   # row-major
    var D = make_view(bufD +% 0, make_layout((M, N), (N, 1)))
    # TODO: remove the pointer-offset arithmetic — the bias view must
    # come from the (N,) buffer's own layout, not `+%` offsets
    let bias = make_view(bufBias +% 0, (M, N), (0, 1))   # stride-0 rows
    let op = initEpiAddBias(bias)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j] + bufBias[j]

  test "EpiAddBias strided bias (stride-2 column, holes never read)":
    ## The bias's own layout carries the stride: a (N,) column stored at
    ## even offsets. The NaN holes prove the epilogue addresses through
    ## the view, not a packed scan.
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N)
    var bufBias = newSeq[float32](2 * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N:
      bufAB[i] = float32(i)
    for j in 0 ..< N:
      bufBias[2 * j] = float32(100 + j)
      bufBias[2 * j + 1] = NaN
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    # TODO: remove the pointer-offset arithmetic — the bias view must
    # come from the (N,) buffer's own layout, not `+%` offsets
    let bias = make_view(bufBias +% 0, (M, N), (0, 2))   # stride-0 rows, stride-2 columns
    let op = initEpiAddBias(bias)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j] + bufBias[2 * j]
        doAssert not D[i, j].isNaN, "stride-2 holes must never be read"

  test "EpiAddBias reversed bias (negative stride, natural order restored)":
    ## The column stored in reverse order (bufBias[N-1-j] = 100 + j) is
    ## viewed with stride -1: element (i, j) reads bufBias[N-1-j].
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N)
    var bufBias = newSeq[float32](N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N:
      bufAB[i] = float32(i)
    for j in 0 ..< N:
      bufBias[j] = float32(100 + (N - 1 - j))
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    # TODO: remove the pointer-offset arithmetic — the bias view must
    # come from the (N,) buffer's own layout, not `+%` offsets
    let bias = make_view(bufBias +% (N - 1), (M, N), (0, -1))   # reversed column
    let op = initEpiAddBias(bias)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j] + bufBias[N - 1 - j]

  test "EpiAddBias bias sliced from a larger buffer (in+out)":
    ## The bias view may start mid-buffer: the data pointer carries the
    ## offset, the layout the stride.
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N)
    var bufBig = newSeq[float32](2 + N + 2)   # [slack | bias | slack]
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N:
      bufAB[i] = float32(i)
    for j in 0 ..< N:
      bufBig[2 + j] = float32(100 + j)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    # TODO: remove the pointer-offset arithmetic — the bias view must
    # come from the (N,) buffer's own layout, not `+%` offsets
    let bias = make_view(bufBig +% 2, (M, N), (0, 1))
    let op = initEpiAddBias(bias)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j] + bufBig[2 + j]

  test "EpiAddBias fragment-shaped bias (m16n8k8 C-fragment layout, column-cyclic)":
    ## The GPU path: the bias view mirrors the accumulator fragment shape
    ## (rows {0,8} × cols {0,1}), with the ROW mode stride-0 — the
    ## broadcast over the fragment's column-cyclic order.
    const M = 2; const N = 2
    var bufTile = newSeq[float32](16 * 8)   # the full 16×8 accumulator tile
    var bufBias = newSeq[float32](2)        # the 2 columns
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< 16*8:
      bufTile[i] = float32(i + 1)
    for j in 0 ..< 2:
      bufBias[j] = float32(100 + j)
    let AB = make_view(bufTile +% 0, (2, 2), (8, 1))          # fragment: rows {0,8} × cols {0,1}
    var D = make_view(bufD +% 0, (M, N), (1, M))
    # TODO: remove the pointer-offset arithmetic — the bias view must
    # come from the (N,) buffer's own layout, not `+%` offsets
    let bias = make_view(bufBias +% 0, (2, 2), (0, 1))        # rows broadcast, cols stride 1
    let op = initEpiAddBias(bias)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j] + bufBias[j]
    doAssert D[0, 0] == bufTile[0] + bufBias[0]   # fragment (0,0) = tile[0], col 0
    doAssert D[1, 0] == bufTile[8] + bufBias[0]   # fragment row 1 = tile[8], col 0
    doAssert D[0, 1] == bufTile[1] + bufBias[1]   # fragment (0,1) = tile[1], col 1
    doAssert D[1, 1] == bufTile[9] + bufBias[1]   # fragment (1,1) = tile[9], col 1

  # ── layout robustness II: slices, inverted strides, real atom shapes ──
  # C and D live in gmem and their strides are NOT ours to choose. They
  # may arrive reshaped from an implicit convolution, a sliced tensor, a
  # permutation, etc. AB comes from the MMA accumulator fragment, whose
  # layout is the atom's register map (never assumed col-major).
  # The `apply` iterates `size(D)` and indexes each operand through its own
  # layout, so rank-3 fragments, nested layouts and broadcast (stride-0)
  # views all work. The AMX nested layout is exercised below.

  test "EpiAXPBY sliced tensors — in (view at buffer start, slack after)":
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N + 8)   # [data | slack]
    var bufC = newSeq[float32](M * N + 8)
    var bufD = newSeq[float32](M * N + 8)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = float32(100 + i)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j] + 3.0'f32 * C[i, j]
    doAssert bufD[M*N] == 0.0'f32, "slack after the slice must stay untouched"

  test "EpiAXPBY sliced tensors — out (view ends at buffer end)":
    const M = 3; const N = 4
    var bufAB = newSeq[float32](8 + M * N)   # [slack | data]
    var bufC = newSeq[float32](8 + M * N)
    var bufD = newSeq[float32](8 + M * N)
    for i in 0 ..< M*N: bufAB[8 + i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[8 + i] = float32(100 + i)
    let AB = make_view(bufAB +% 8, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 8, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 8, make_layout((M, N), (1, M)))
    let op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j] + 3.0'f32 * C[i, j]
    doAssert bufD[7] == 0.0'f32, "slack before the slice must stay untouched"

  test "EpiAXPBY sliced tensors — in+out (middle slice of a larger buffer)":
    const M = 3; const N = 4
    var bufAB = newSeq[float32](4 + M * N + 4)   # [slack | data | slack]
    var bufC = newSeq[float32](4 + M * N + 4)
    var bufD = newSeq[float32](4 + M * N + 4)
    for i in 0 ..< M*N: bufAB[4 + i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[4 + i] = float32(100 + i)
    let AB = make_view(bufAB +% 4, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 4, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 4, make_layout((M, N), (1, M)))
    let op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j] + 3.0'f32 * C[i, j]
    doAssert bufD[0] == 0.0'f32 and bufD[4 + M*N] == 0.0'f32,
      "slack on both sides must stay untouched"

  test "EpiAXPBY fully inverted strides (D, AB, C all (-1, -M))":
    ## Negative strides: logical (0,0) sits at the buffer END; element
    ## (i, j) lives at buf[(M-1) - i + (N-1-j)*M]. The epilogue must
    ## address through the layout, not assume a forward scan.
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = float32(100 + i)
    let AB = make_view(bufAB +% (M*N - 1), make_layout((M, N), (-1, -M)))
    let C = make_view(bufC +% (M*N - 1), make_layout((M, N), (-1, -M)))
    var D = make_view(bufD +% (M*N - 1), make_layout((M, N), (-1, -M)))
    let op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j] + 3.0'f32 * C[i, j]
    # concrete buffer positions: logical (M-1, N-1) → buf[0]; (0, 0) → buf[MN-1]
    doAssert bufD[0] == 2.0'f32 * bufAB[0] + 3.0'f32 * bufC[0]
    doAssert bufD[M*N - 1] == 2.0'f32 * bufAB[M*N - 1] + 3.0'f32 * bufC[M*N - 1]

  test "EpiAXPBY mixed inverted strides (AB rows inverted, C cols inverted, D normal)":
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = float32(100 + i)
    # AB rows inverted: (i, j) at buf[(M-1) - i + M*j], pointer at (M-1)
    let AB = make_view(bufAB +% (M - 1), make_layout((M, N), (-1, M)))
    # C cols inverted: (i, j) at buf[i - M*j + M*(N-1)], pointer at M*(N-1)
    let C = make_view(bufC +% (M * (N - 1)), make_layout((M, N), (1, -M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j] + 3.0'f32 * C[i, j]
    # AB rows inverted: (i,j) at buf[(M-1) - i + M*j]
    #   (0,0)=buf[M-1], (M-1,N-1)=buf[M*(N-1)]
    # C cols inverted:  (i,j) at buf[i + M*(N-1-j)]
    #   (0,0)=buf[M*(N-1)], (M-1,N-1)=buf[M-1]
    doAssert D[0, 0] == 2.0'f32 * bufAB[M - 1] + 3.0'f32 * bufC[M * (N - 1)]
    doAssert D[M - 1, N - 1] == 2.0'f32 * bufAB[M * (N - 1)] + 3.0'f32 * bufC[M - 1]

  test "EpiAXPBY AB with the m16n8k8 C-fragment layout (shape (2,2), strides (8,1))":
    ## The per-thread C fragment of the sm80 m16n8k8 tensor core
    ## (atoms_nvidia.nim SM80_16x8_Row = (T32,V4) → (M16,N8)): 4 elements at
    ## rows {0, 8} × cols {0, 1} of the 16×8 accumulator tile. The rows
    ## are non-contiguous, with row pitch 8. AB must be addressed through
    ## this layout, never assumed col-major.
    const M = 2; const N = 2
    var bufTile = newSeq[float32](16 * 8)   # the full 16×8 accumulator tile
    var bufC = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< 16*8:
      bufTile[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = float32(10 + i)
    let AB = make_view(bufTile +% 0, make_layout((2, 2), (8, 1)))
    let C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j] + 3.0'f32 * C[i, j]
    doAssert D[0, 0] == 2.0'f32 * bufTile[0] + 3.0'f32 * C[0, 0]   # fragment (0,0) = tile[0]
    doAssert D[1, 0] == 2.0'f32 * bufTile[8] + 3.0'f32 * C[1, 0]   # fragment row 1 = tile[8]
    doAssert D[0, 1] == 2.0'f32 * bufTile[1] + 3.0'f32 * C[0, 1]   # fragment (0,1) = tile[1]
    doAssert D[1, 1] == 2.0'f32 * bufTile[9] + 3.0'f32 * C[1, 1]   # fragment (1,1) = tile[9]

  test "EpiReLU with the m16n8k8 C-fragment AB layout":
    ## Same fragment layout, activation op: proves the AB read through a
    ## non-trivial atom layout is layout-correct, not just for EpiAXPBY.
    const M = 2; const N = 2
    var bufTile = newSeq[float32](16 * 8)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< 16*8: bufTile[i] = float32(int(i) - 16)   # mix of signs
    let AB = make_view(bufTile +% 0, make_layout((2, 2), (8, 1)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let op = EpiReLU()
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == max(AB[i, j], 0.0'f32)

  test "EpiIdentity row-major D/AB":
    ## Round out op × row-major coverage.
    const M = 3; const N = 4
    var bufAB = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (N, 1)))
    var D = make_view(bufD +% 0, make_layout((M, N), (N, 1)))
    let op = EpiIdentity()
    applyEpilogue(op, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == AB[i, j]

  test "EpiAXPBY rank-3 operands (V, RestM, RestN) = (4, 2, 2), distinct strides":
    ## All operands share the shape type Sh. Each is indexed through its
    ## own stride pattern, here rank-3 with V stride-1 on AB (register
    ## order, the atom map) and different strides on C and D.
    const Shp = (4, 2, 2)
    var bufAB = newSeq[float32](32)
    var bufC = newSeq[float32](16)
    var bufD = newSeq[float32](32)
    for i in 0 ..< 32: bufAB[i] = float32(i + 1)
    for i in 0 ..< 16: bufC[i] = float32(100 + i)
    let AB = make_view(bufAB +% 0, make_layout(Shp, (1, 8, 16)))
    let C = make_view(bufC +% 0, make_layout(Shp, (1, 4, 8)))
    var D = make_view(bufD +% 0, make_layout(Shp, (1, 4, 16)))
    let op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    applyEpilogue(op, D, AB)
    for i in 0 ..< 16:
      doAssert D(i) == 2.0'f32 * AB(i) + 3.0'f32 * C(i)
    doAssert D(0) == 2.0'f32 * bufAB[0] + 3.0'f32 * bufC[0]
    # flat 4 on (4,2,2):(1,8,16) decomposes (0,1,0) -> offset 8 (V=0, RestM=1)
    doAssert AB(4) == bufAB[8]

  test "EpiIdentity AMX nested broadcast layout (1, (16, 16)):(0, (1, 16))":
    ## The AMX 16x16x32 accumulator layout (atoms_amx.py): nested 16x16
    ## with a leading stride-0 (broadcast) mode. All operands share the
    ## shape; the nested+broadcast layout is exercised on each.
    const M = 16; const N = 16
    var bufAB = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N:
      bufAB[i] = float32(i)
    let AB = make_view(bufAB +% 0, make_layout((1, (16, 16)), (0, (1, 16))))
    var D = make_view(bufD +% 0, make_layout((1, (16, 16)), (0, (1, 16))))
    let op = EpiIdentity()
    applyEpilogue(op, D, AB)
    for i in 0 ..< M*N:
      doAssert D(i) == AB(i)

  test "preflight is callable on every op":
    ## `preflight` is a structural contract (not a concept member).
    ## It injects the staging buffer and copies the op's operands into
    ## it. Here: call it on each op, then `apply` still works.
    const M = 2; const N = 3
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * N)
    var bufBias = newSeq[float32](N)
    var bufD = newSeq[float32](M * N)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = float32(10 + i)
    for j in 0 ..< N:
      bufBias[j] = float32(100 + j)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    let bias = make_view(bufBias +% 0, (M, N), (0, 1))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    var opAX = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    var opId = EpiIdentity()
    var opBias = initEpiAddBias(bias)
    var opReLU = EpiReLU()
    opAX.preflight()
    opId.preflight()
    opBias.preflight()
    opReLU.preflight()
    applyEpilogue(opAX, D, AB)
    for i in 0 ..< M:
      for j in 0 ..< N:
        doAssert D[i, j] == 2.0'f32 * AB[i, j] + 3.0'f32 * C[i, j]

  test "AB with a different shape must not compile (shared Sh)":
    ## D and AB share the shape type Sh in the signature. The compiler
    ## enforces the same shape; a mismatched operand is a type error, no
    ## runtime or static assert needed.
    const M = 2; const N = 2
    var bufAB = newSeq[float32](M * N)
    var bufC = newSeq[float32](M * N)
    var bufD = newSeq[float32](M * N)
    var bufSmall = newSeq[float32](1)
    for i in 0 ..< M*N: bufAB[i] = float32(i + 1)
    for i in 0 ..< M*N: bufC[i] = float32(100 + i)
    let AB = make_view(bufAB +% 0, make_layout((M, N), (1, M)))
    let C = make_view(bufC +% 0, make_layout((M, N), (1, M)))
    var D = make_view(bufD +% 0, make_layout((M, N), (1, M)))
    let small = make_view(bufSmall +% 0, make_layout((1, 1), (1, 1)))
    let op = initEpiAXPBY(2.0'f32, 3.0'f32, C)
    doAssert not compiles(applyEpilogue(op, D, small)),
      "AB with a different shape must be a type error"

proc main() =
  runEpilogueTests()

when isMainModule:
  main()
