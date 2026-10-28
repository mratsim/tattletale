## WebGPU: global binding flavors from the write analysis.
## Test that a pointer parameter the kernel writes binds read_write while
## a pointer parameter it only reads stays read-only.
## Test that a bool buffer round-trips through the i32 storage stripping.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/types/t_webgpu_global_bindings.nim

import std/unittest
import workspace/crucible

const bindingWgsl = webgpu:
  proc writtenAndRead(output, consts: ptr UncheckedArray[uint32]) {.global.} =
    # output is written, consts is only read: the write analysis must
    # give them different binding flavors
    for i in 0 ..< 4:
      output[i] = consts[i] * 3'u32

const boolWgsl = webgpu:
  proc boolRoundTrip(output, flags: ptr UncheckedArray[uint32]) {.global.} =
    # bool storage strips to i32, loads wrap back through conversions
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

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "WebGPU - global binding flavors":

    test "the written parameter writes, the read-only parameter reads":
      var engine = newEngine(runWebGPU)
      engine.ingest(bindingWgsl)
      let oDev = engine.allocDevice(uint32, 4)
      let cDev = engine.allocDevice(uint32, 4)
      var consts: array[4, uint32] = [7'u32, 8'u32, 9'u32, 10'u32]
      engine.copyToDeviceFromHost(cDev, addr consts[0], 4)
      engine.launchKernel("writtenAndRead", oDev, (cDev,))
      var res: array[4, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      for i in 0 ..< 4:
        check res[i] == consts[i] * 3'u32

    test "a bool round-trips through the storage stripping":
      var engine = newEngine(runWebGPU)
      engine.ingest(boolWgsl)
      let oDev = engine.allocDevice(uint32, 2)
      let fDev = engine.allocDevice(uint32, 1)
      var flag: array[1, uint32] = [1'u32]
      engine.copyToDeviceFromHost(fDev, addr flag[0], 1)
      engine.launchKernel("boolRoundTrip", oDev, (fDev,))
      var res: array[2, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 111'u32
      check res[1] == 0'u32

when isMainModule:
  runTest()
