## CuTe Layout + Tile dot products — NVRTC (CUDA) backend
## Run with: nim c -r workspace/crucible/tests/codegen/nvrtc/test_cute_layout_cuda.nim
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  Layout[S: static tuple, D: static tuple] = object
    discard
  Tile[M, N: static int] = object
    data: array[M * N, uint32]

proc tileAt[M, N: static int](t: Tile[M, N]; r, c: uint32): uint32 =
  t.data[r * uint32(N) + c]

const kernelCode = cuda:
  proc cuteKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let t = Tile[2, 3](data: [10'u32, 20'u32, 30'u32, 40'u32, 50'u32, 60'u32])
    output[0] = tileAt(t, 0'u32, 0'u32)
    output[1] = tileAt(t, 0'u32, 1'u32)
    output[2] = tileAt(t, 1'u32, 2'u32)
    let a00 = 1'u32; let a01 = 2'u32; let a10 = 3'u32; let a11 = 4'u32
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
echo "  OK — CuTe Layout + Tile  (CUDA)"
