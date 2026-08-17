## Metal: CuTe Layout + Tile dot products. The Tile type and tileAt device fn
## live inside the `metal:` block. The kernel reads tile elements through tileAt
## and computes two GEMM-style dot products with local coefficients.
## Slots 0..2 cover tileAt reads, and slots 4..5 cover the dot products.
## The kernel never writes slot 3. The engine uploads the output's current bytes
## before launch, so it reads back 0, pinning the in-place β·C upload path.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_cute_layout.nim

import std/unittest
import workspace/crucible

const kernelMsl = metal:
  type Tile[M, N: static int] = object
    data: array[M * N, uint32]
  proc tileAt[M, N: static int](t: Tile[M, N], r, c: uint32): uint32 =
    t.data[r * uint32(N) + c]
  proc cuteKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let t = Tile[2, 3](data: [10'u32, 20'u32, 30'u32, 40'u32, 50'u32, 60'u32])
    output[0] = tileAt(t, 0'u32, 0'u32)
    output[1] = tileAt(t, 0'u32, 1'u32)
    output[2] = tileAt(t, 1'u32, 2'u32)
    let a00 = 1'u32
    let a01 = 2'u32
    let a10 = 3'u32
    let a11 = 4'u32
    let gemmB = Tile[2, 3](data: [5'u32, 6'u32, 7'u32, 8'u32, 9'u32, 10'u32])
    output[4] = a00 * tileAt(gemmB, 0'u32, 0'u32) + a01 * tileAt(gemmB, 1'u32, 0'u32)
    output[5] = a10 * tileAt(gemmB, 0'u32, 2'u32) + a11 * tileAt(gemmB, 1'u32, 2'u32)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernelMsl

  suite "Metal - CuTe Layout + Tile":
    test "tileAt reads and dot products on the device":
      var engine = bkMetal.init()
      engine.ingest(kernelMsl)
      var res: array[6, uint32]
      engine.run("cuteKernel", res, ())
      check res[0] == 10
      check res[1] == 20
      check res[2] == 60
      check res[3] == 0
      check res[4] == 21
      check res[5] == 61

when isMainModule:
  runTest()
