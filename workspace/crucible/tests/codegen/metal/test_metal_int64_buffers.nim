## Metal: int64/uint64 buffer arithmetic.
## `int64Kernel` adds two `int64` elements and doubles a `uint64` element with
## a 10^12 constant. Runs through `engine.run()` and asserts the output
## byte-exact — proving 8-byte buffer elements marshal at their Nim width.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_int64_buffers.nim

import std/unittest
import workspace/crucible

const int64Msl = metal:
  proc int64Kernel(output: ptr UncheckedArray[uint64];
                   a: ptr UncheckedArray[int64];
                   b: ptr UncheckedArray[uint64]) {.global.} =
    let x = a[0] + a[1]
    let y = b[0] * 2'u64 + 1000000000000'u64
    output[0] = uint64(x)
    output[1] = y

proc runTest() =
  suite "Metal - int64/uint64 buffers":
    test "int64 add + uint64 scale round-trip":
      var engine = bkMetal.init()
      engine.ingest(int64Msl)
      var a: array[2, int64] = [-5'i64, 12'i64]
      var b: array[2, uint64] = [3'u64, 0'u64]
      var res: array[2, uint64]
      engine.run("int64Kernel", res, (a, b))
      check res[0] == 7'u64
      check res[1] == 1000000000006'u64

when isMainModule:
  runTest()
