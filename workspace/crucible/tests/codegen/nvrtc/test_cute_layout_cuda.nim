## CuTe Layout + GEMM tile — NVRTC (CUDA) backend
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/codegen/nvrtc/test_cute_layout_cuda.nim
##
## CuTe Layout: S0,S1 = shape (rows,cols), D0,D1 = stride (row,col).
##   apply(r, c) = r * D0 + c * D1
##   Struct names include static values (gtStatic fix): e.g. Layout_i4_i1_i2_i1
import std/strformat
import workspace/crucible/src/codegen/nvrtc

# ── CuTe-style types ────────────────────────────────────────────────────────

type
  Layout[S0, S1, D0, D1: static int] = object
    ## Shape (S0,S1), Stride (D0,D1). at(r,c) = data[r*D0 + c*D1]
    data: array[(S0-1)*D0 + (S1-1)*D1 + 1, uint32]

  Tile[M, N: static int] = object
    ## Dense row-major tile for GEMM.
    data: array[M * N, uint32]

# ── Device functions ────────────────────────────────────────────────────────

proc at[S0, S1, D0, D1: static int](l: Layout[S0, S1, D0, D1];
                                       r, c: uint32): uint32 {.device.} =
  l.data[r * uint32(D0) + c * uint32(D1)]

proc tileAt[M, N: static int](t: Tile[M, N]; r, c: uint32): uint32 {.device.} =
  t.data[r * uint32(N) + c]

# ── Single kernel: all tests share one compilation ──────────────────────────

const kernelCode = cuda:
  proc cuteKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # ── 1D strided: Layout[4,1,2,1] → data size 7, stride 2 ──
    let l1 = Layout[4, 1, 2, 1](
      data: [10'u32, 0'u32, 20'u32, 0'u32, 30'u32, 0'u32, 40'u32])
    output[0] = at(l1, 0'u32, 0'u32)  # idx = 0*2+0*1 = 0  → 10
    output[1] = at(l1, 1'u32, 0'u32)  # idx = 1*2+0*1 = 2  → 20
    output[2] = at(l1, 3'u32, 0'u32)  # idx = 3*2+0*1 = 6  → 40

    # ── 2D row-major: Layout[2,3,3,1] → data size 6, stride (3,1) ──
    let l2 = Layout[2, 3, 3, 1](
      data: [1'u32, 2'u32, 3'u32, 4'u32, 5'u32, 6'u32])
    output[3] = at(l2, 0'u32, 0'u32)  # idx = 0*3+0*1 = 0 → 1
    output[4] = at(l2, 1'u32, 2'u32)  # idx = 1*3+2*1 = 5 → 6

    # ── Dense tile ──
    let t = Tile[2, 3](data: [10'u32, 20'u32, 30'u32,
                              40'u32, 50'u32, 60'u32])
    output[5] = tileAt(t, 0'u32, 1'u32)  # idx = 0*3+1 = 1 → 20
    output[6] = tileAt(t, 1'u32, 2'u32)  # idx = 1*3+2 = 5 → 60

    # ── GEMM (C[2×3] = A[2×2] × B[2×3]) ──
    let a00 = 1'u32; let a01 = 2'u32
    let a10 = 3'u32; let a11 = 4'u32
    let gemmB = Tile[2, 3](data: [5'u32, 6'u32, 7'u32, 8'u32, 9'u32, 10'u32])
    output[8] = a00 * tileAt(gemmB, 0'u32, 0'u32) + a01 * tileAt(gemmB, 1'u32, 0'u32)
    output[9] = a10 * tileAt(gemmB, 0'u32, 2'u32) + a11 * tileAt(gemmB, 1'u32, 2'u32)

# ── Execute ─────────────────────────────────────────────────────────────────

var buf: array[10, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.numBlocks = 1
nv.threadsPerBlock = 1
nv.execute("cuteKernel", buf, ())

doAssert buf[0] == 10, &"1D[0]: {buf[0]}"
doAssert buf[1] == 20, &"1D[1]: {buf[1]}"
doAssert buf[2] == 40, &"1D[3]: {buf[2]}"
doAssert buf[3] == 1,  &"2D[0,0]: {buf[3]}"
doAssert buf[4] == 6,  &"2D[1,2]: {buf[4]}"
doAssert buf[5] == 20, &"tile[0,1]: {buf[5]}"
doAssert buf[6] == 60, &"tile[1,2]: {buf[6]}"
doAssert buf[8] == 21, &"gemm[0,0]: {buf[8]}"
doAssert buf[9] == 61, &"gemm[1,2]: {buf[9]}"
echo "  OK — CuTe Layout + GEMM (CUDA)"
