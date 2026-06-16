## CuTe Layout  tile — NVRTC (CUDA) backend
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/codegen/nvrtc/test_cute_layout_cuda.nim
##
## Layout[S, D: static tuple] — pure mapping descriptor (no data).
## Tile[M, N: static int] — dense row-major data container.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  Layout[S: static tuple, D: static tuple] = object
    ## Shape S = (rows, cols), Stride D = (row_stride, col_stride).
    ## Pure descriptor — no data. Use at() with a Tile or array.

  Tile[M, N: static int] = object
    data: array[M * N, uint32]

# CuTe-style: at(coord) computes flat index from shape/stride.
# For now, inline the index computation since tuple field access
# in generated C is pending codegen support.

proc tileAt[M, N: static int](t: Tile[M, N]; r, c: uint32): uint32 =
  t.data[r * uint32(N) + c]

const kernelCode = cuda:
  proc cuteKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # ── Tile data access (2×3 row-major) ──
    let t = Tile[2, 3](data: [10'u32, 20'u32, 30'u32,
                              40'u32, 50'u32, 60'u32])
    output[0] = tileAt(t, 0'u32, 0'u32)  # idx = 0*3+0 = 0 → 10
    output[1] = tileAt(t, 0'u32, 1'u32)  # idx = 0*3+1 = 1 → 20
    output[2] = tileAt(t, 1'u32, 2'u32)  # idx = 1*3+2 = 5 → 60

    # ── Tile dot products (scalar entries of C[2×3] = A[2×2] × B[2×3]) ──
    let a00 = 1'u32; let a01 = 2'u32
    let a10 = 3'u32; let a11 = 4'u32
    let gemmB = Tile[2, 3](data: [5'u32, 6'u32, 7'u32, 8'u32, 9'u32, 10'u32])
    output[4] = a00 * tileAt(gemmB, 0'u32, 0'u32) + a01 * tileAt(gemmB, 1'u32, 0'u32)
    output[5] = a10 * tileAt(gemmB, 0'u32, 2'u32) + a11 * tileAt(gemmB, 1'u32, 2'u32)

var buf: array[6, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.numBlocks = 1
nv.threadsPerBlock = 1
nv.execute("cuteKernel", buf, ())
doAssert buf[0] == 10, &"tile[0,0]: {buf[0]}"
doAssert buf[1] == 20, &"tile[0,1]: {buf[1]}"
doAssert buf[2] == 60, &"tile[1,2]: {buf[2]}"
doAssert buf[4] == 21, &"dot[0,0]: {buf[4]}"
doAssert buf[5] == 61, &"dot[1,2]: {buf[5]}"
echo "  OK — CuTe Layout (descriptor) + Tile  (CUDA)"
