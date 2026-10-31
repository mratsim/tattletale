## Metal: coordinate builtins thread down to device functions as parameters.
##
## Test that a `.device.` proc reads thread coordinates through appended
## parameters and that forwarded call-site values produce byte-exact
## `index + 100` readback from the launched kernels.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_device_fn_builtin_threading.nim

import std/unittest
import workspace/crucible

proc bump(p: ptr UncheckedArray[uint32]) {.device.} =
  let t = thread_index_in_threadgroup
  p[t] = p[t] + 100'u32

proc mid(p: ptr UncheckedArray[uint32]) {.device.} =
  let x = thread_position_in_threadgroup.x
  p[x] = uint32(x)
  bump(p)

const chainMsl = metal:
  proc chainKernel(p: ptr UncheckedArray[uint32]) {.global.} =
    mid(p)

  proc directKernel(o: ptr UncheckedArray[uint32]) {.global.} =
    let t = thread_index_in_threadgroup
    o[t] = uint32(t)
    bump(o)

proc runTest() =
  suite "Metal - device fn builtin threading":
    test "the two-deep chain forwards the coordinates, readback is index + 100":
      var engine = newEngine(runMetal)
      engine.ingest(chainMsl)
      let pDev = engine.allocDevice(uint32, 64)
      let oDev = engine.allocDevice(uint32, 64)
      var sentinels: array[64, uint32]
      for i in 0 ..< 64:
        sentinels[i] = 0xEEEE_0000'u32 + uint32(i)
      engine.copyToDeviceFromHost(pDev, addr sentinels[0], 64)
      engine.copyToDeviceFromHost(oDev, addr sentinels[0], 64)
      engine.launchKernel<<(grid: (1, 1), blk: (64, 1))>>("chainKernel", pDev, ())
      engine.launchKernel<<(grid: (1, 1), blk: (64, 1))>>("directKernel", oDev, ())
      var res: array[64, uint32]
      engine.copyToHostFromDevice(addr res[0], pDev, 64)
      for i in 0 ..< 64:
        check res[i] == uint32(i) + 100'u32
      engine.copyToHostFromDevice(addr res[0], oDev, 64)
      for i in 0 ..< 64:
        check res[i] == uint32(i) + 100'u32

when isMainModule:
  runTest()
