## Metal: one DeviceMem buffer relaunched with readback between launches.
##
## Test that repeated launches with readback in between reuse one DeviceMem
## buffer and that a temporary buffer argument crosses to the kernel.
## One launch binds a buffer into both its output and an argument slot.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_relaunch_same_buffer.nim

import std/unittest
import workspace/crucible

const scaleMsl = metal:
  proc scaleBy(output: ptr UncheckedArray[uint32], input: ptr UncheckedArray[uint32], factor: uint32) {.global.} =
    output[0] = input[0] * factor

proc initBuffer(engine: MetalEngine, v: uint32): auto =
  result = engine.allocDevice(uint32, 1)
  var host: array[1, uint32] = [v]
  engine.copyToDeviceFromHost(result, addr host[0], 1)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - same buffer relaunched with readback between launches":

    test "one buffer launched twice, readback between launches":
      var engine = newEngine(runMetal)
      engine.ingest(scaleMsl)
      let inDev = engine.allocDevice(uint32, 1)
      let outDev = engine.allocDevice(uint32, 1)
      var host: array[1, uint32] = [3'u32]
      engine.copyToDeviceFromHost(inDev, addr host[0], 1)
      var res: array[1, uint32]

      engine.launchKernel("scaleBy", outDev, (inDev, 2'u32))
      engine.copyToHostFromDevice(addr res[0], outDev, 1)
      check res[0] == 6'u32

      engine.launchKernel("scaleBy", outDev, (inDev, 5'u32))
      engine.copyToHostFromDevice(addr res[0], outDev, 1)
      check res[0] == 15'u32

    test "a temporary buffer argument crosses to the kernel":
      var engine = newEngine(runMetal)
      engine.ingest(scaleMsl)
      let outDev = engine.allocDevice(uint32, 1)
      var res: array[1, uint32]

      engine.launchKernel("scaleBy", outDev, (engine.initBuffer(7), 4'u32))
      engine.copyToHostFromDevice(addr res[0], outDev, 1)
      check res[0] == 28'u32

    test "the output aliases an argument":
      var engine = newEngine(runMetal)
      engine.ingest(scaleMsl)
      let dev = engine.allocDevice(uint32, 1)
      var host: array[1, uint32] = [3'u32]
      engine.copyToDeviceFromHost(dev, addr host[0], 1)
      var res: array[1, uint32]

      engine.launchKernel("scaleBy", dev, (dev, 2'u32))
      engine.copyToHostFromDevice(addr res[0], dev, 1)
      check res[0] == 6'u32

when isMainModule:
  runTest()
