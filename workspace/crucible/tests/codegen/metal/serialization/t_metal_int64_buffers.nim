## Metal: int64/uint64 buffer arithmetic. `int64Kernel` adds two `int64`
## elements and doubles a `uint64` element with a 10^12 constant, proving
## 8-byte buffer elements marshal at their Nim width.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_int64_buffers.nim

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
      let aDev = engine.allocDevice(a.len * sizeof(int64))
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(int64))
      let bDev = engine.allocDevice(b.len * sizeof(uint64))
      engine.copyToDeviceFromHost(bDev, addr b[0], b.len * sizeof(uint64))
      let oDev = engine.allocDevice(2 * sizeof(uint64))
      engine.launchKernel("int64Kernel", oDev, (aDev, bDev))
      var res: array[2, uint64]
      engine.copyToHostFromDevice(addr res[0], oDev, 2 * sizeof(uint64))
      engine.freeDevice(aDev)
      engine.freeDevice(bDev)
      engine.freeDevice(oDev)
      check res[0] == 7'u64
      check res[1] == 1000000000006'u64

when isMainModule:
  runTest()
