## Metal: integer `and`/`or`/`xor` compute the bitwise result on the device.
##
## Test that the logical lowering of integer `and`/`or` rewrites to the bit
## form and reads back the same values Nim computes.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_int_bitwise_ops.nim

import std/unittest
import workspace/crucible

const bitOpsMsl = metal:
  proc bitOps(o, x: ptr UncheckedArray[int32]) {.global.} =
    let v = x[0]
    o[0] = v and 0x0F'i32
    o[1] = v or 0x30'i32
    o[2] = v xor 0xFF'i32

proc runTest() =
  suite "Metal - integer and/or/xor bitwise results":

    test "and/or/xor over an int32 operand match the bitwise values":
      var engine = newEngine(runMetal)
      engine.ingest(bitOpsMsl)
      let xDev = engine.allocDevice(int32, 1)
      let oDev = engine.allocDevice(int32, 3)
      var x = [0xA5'i32]
      engine.copyToDeviceFromHost(xDev, addr x[0], 1)
      engine.launchKernel("bitOps", oDev, (xDev,))
      var res: array[3, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 3)
      check res[0] == 0x05'i32  # 0xA5 and 0x0F
      check res[1] == 0xB5'i32  # 0xA5 or 0x30
      check res[2] == 0x5A'i32  # 0xA5 xor 0xFF

when isMainModule:
  runTest()
