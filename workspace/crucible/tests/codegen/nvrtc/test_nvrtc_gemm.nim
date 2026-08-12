## CuTe GEMM single-tile — NVRTC (CUDA) backend
## A(2×3) * B(3×2) = C(2×2)
## Run with:
##   nim c -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_gemm.nim
import std/strformat
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

type
  Layout[S: static tuple, D: static tuple] = object
    discard
  Tile[M, N: static int] = object
    data: array[M * N, uint32]

proc tileAt[M, N: static int](t: Tile[M, N]; r, c: uint32): uint32 =
  t.data[r * uint32(N) + c]

const kernelCode = cuda:
  proc gemmKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let A = Tile[2, 3](data: [1'u32, 2, 3, 4, 5, 6])
    let B = Tile[3, 2](data: [7'u32, 8, 9, 10, 11, 12])
    var Ct: Tile[2, 2]
    for i in 0 ..< 2:
      for j in 0 ..< 2:
        var sum: uint32 = 0
        for k in 0 ..< 3:
          sum += tileAt(A, uint32(i), uint32(k)) * tileAt(B, uint32(k), uint32(j))
        Ct.data[i * 2 + j] = sum
    C[0] = Ct.data[0]
    C[1] = Ct.data[1]
    C[2] = Ct.data[2]
    C[3] = Ct.data[3]

var buf: array[4, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"
engine.run<<(1, 1)>>("gemmKernel", buf, ())
let expected = [58'u32, 64, 139, 154]
for i in 0 ..< 4:
  doAssert buf[i] == expected[i], &"C[{i}]: {buf[i]} != {expected[i]}"
echo "  OK — CuTe GEMM (CUDA)"
