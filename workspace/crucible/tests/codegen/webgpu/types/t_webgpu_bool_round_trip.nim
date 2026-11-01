## WebGPU: a bool derived from a uint32 storage word round-trips.
##
## Test that a load from a uint32 storage buffer coerces to a bool.
## `not` and the branch selection keep working after the coercion.
## Each branch writes its selected value into the output buffer.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/types/t_webgpu_bool_round_trip.nim

import workspace/crucible

const boolWgsl = webgpu:
  proc boolRoundTrip(output, flags: ptr UncheckedArray[uint32]) {.global.} =
    var b = flags[0] != 0'u32
    var c = not b
    if b:
      output[0] = 111'u32
    else:
      output[0] = 222'u32
    if c:
      output[1] = 1'u32
    else:
      output[1] = 0'u32

proc launchAndRead(flag: uint32): array[2, uint32] =
  # DeviceMem has no =dup: one fresh engine per launch keeps every binding
  # single-read, and engine destruction at proc return releases the buffers.
  var engine = newEngine(runWebGPU)
  engine.ingest(boolWgsl)
  let oDev = engine.allocDevice(uint32, 2)
  let fDev = engine.allocDevice(uint32, 1)
  var flagBuf: array[1, uint32] = [flag]
  engine.copyToDeviceFromHost(fDev, addr flagBuf[0], 1)
  engine.launchKernel("boolRoundTrip", oDev, (fDev,))
  engine.copyToHostFromDevice(addr result[0], oDev, 2)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  block:  # a set flag: b true, c false
    let res = launchAndRead(1'u32)
    doAssert res[0] == 111'u32, "set flag branch: got " & $res[0] & ", want 111"
    doAssert res[1] == 0'u32, "negated flag branch: got " & $res[1] & ", want 0"

  block:  # a clear flag: b false, c true
    let res = launchAndRead(0'u32)
    doAssert res[0] == 222'u32, "clear flag branch: got " & $res[0] & ", want 222"
    doAssert res[1] == 1'u32, "negated flag branch: got " & $res[1] & ", want 1"

  echo "[OK] bool round-trips through the uint32 storage stripping, both flag polarities"

when isMainModule:
  runTest()
