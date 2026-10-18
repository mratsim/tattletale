## Vulkan end-to-end: an array of buffer references in the push constants,
## one reference per block buffer (`array[4, ptr UncheckedArray[uint32]]`).
##
## Run:
##   cd tattletale && \
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/device/t_vulkan_array_of_ptrs.nim

import std/strformat
import workspace/crucible

const kernelCode = vulkan:
  proc k(sink: ptr UncheckedArray[uint32], ps: array[4, ptr UncheckedArray[uint32]]) {.global.} =
    ps[0][1] = ps[2][3] + 1
    sink[0] = ps[0][1]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  block:
    var engine = bkVulkan.init()
    engine.ingest(kernelCode)
    let sDev = engine.allocDevice(uint32, 1)
    var devs = [engine.allocDevice(uint32, 4), engine.allocDevice(uint32, 4),
                engine.allocDevice(uint32, 4), engine.allocDevice(uint32, 4)]
    var seed: array[4, uint32] = [7'u32, 8'u32, 9'u32, 10'u32]
    for i in 0 ..< 4:
      engine.copyToDeviceFromHost(devs[i], addr seed[0], seed.len)
    engine.launchKernel("k", sDev, (devs[0], devs[1], devs[2], devs[3]))
    var got: uint32
    engine.copyToHostFromDevice(addr got, sDev, 1)
    for i in 0 ..< 4:
      engine.freeDevice(devs[i])
    engine.freeDevice(sDev)
    doAssert got == 11'u32, &"ps[0][1] = ps[2][3] + 1: {got} != 11"
    echo "  OK — array of buffer references in the push constants"

when isMainModule:
  runTest()
