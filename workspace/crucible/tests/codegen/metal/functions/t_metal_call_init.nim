## Metal: call-valued variable and let initializers.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_call_init.nim

import workspace/crucible

import std/unittest

const code = metal:
  proc mkv(v: uint32): uint32 {.device.} =
    result = v * 3'u32

  proc initKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var x = mkv(7'u32)
    let y = mkv(5'u32)
    var z = mkv(x)
    output[0] = x + y + z

proc runInit(): array[1, uint32] =
  ## One launch per call: device buffers serve a single launch so handles never need a copy.
  var engine = newEngine(runMetal)
  engine.ingest(code)
  let oDev = engine.allocDevice(uint32, 1)
  engine.launchKernel("initKernel", oDev, ())
  engine.copyToHostFromDevice(addr result[0], oDev, 1)

proc runTest() =
  suite "Metal - call-valued initializers":
    test "var and let initialized from a device function call":
      let r = runInit()
      check r[0] == 99'u32

when isMainModule:
  runTest()
