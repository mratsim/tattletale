## Metal: CuTe GEMM single tile, A(2×3) * B(3×2) = C(2×2).
## The Tile type, the tileAt device fn, and the kernel all live
## inside the `metal:` block and mirror the printer-compile gate's gemm shape.
## The kernel takes no input buffers. A and B are tile constants
## in the body, and the result is written to the output buffer,
## then asserted byte-exact on the host after `engine.run()`.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_gemm.nim

import std/unittest
import workspace/crucible

const kernelCode = metal:
  type Tile[M, N: static int] = object
    data: array[M * N, uint32]
  proc tileAt[M, N: static int](t: Tile[M, N]; r, c: uint32): uint32 =
    t.data[r * uint32(N) + c]
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

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernelCode

  suite "Metal - CuTe GEMM":
    test "A(2×3) * B(3×2) = C(2×2) on the device":
      var engine = bkMetal.init()
      engine.ingest(kernelCode)
      var res: array[4, uint32]
      engine.run("gemmKernel", res, ())
      let expected = [58'u32, 64, 139, 154]
      for i in 0 ..< 4:
        check res[i] == expected[i]

when isMainModule:
  runTest()
