## OpenCL: CuTe GEMM single tile, A(2×3) * B(3×2) = C(2×2), tile type
## and tileAt device fn inside the `opencl:` block.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/test_opencl_gemm.nim

import std/strformat
import workspace/crucible

type
  Layout[S: static tuple, D: static tuple] = object
    discard
  Tile[M, N: static int] = object
    data: array[M * N, uint32]

proc tileAt[M, N: static int](t: Tile[M, N]; r, c: uint32): uint32 =
  t.data[r * uint32(N) + c]

const kernelCode = opencl:
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

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernelCode

  block:
    var engine = bkOpenCL.init()
    engine.ingest(kernelCode)
    var res: array[4, uint32]
    let oDev = engine.allocDevice(4 * sizeof(uint32))
    engine.run("gemmKernel", oDev, ())
    engine.copyToHostFromDevice(addr res[0], oDev, 4 * sizeof(uint32))
    engine.freeDevice(oDev)
    let expected = [58'u32, 64, 139, 154]
    for i in 0 ..< 4:
      doAssert res[i] == expected[i], &"C[{i}]: {res[i]} != {expected[i]}"
    echo "  OK — CuTe GEMM (OpenCL)"

when isMainModule:
  runTest()
