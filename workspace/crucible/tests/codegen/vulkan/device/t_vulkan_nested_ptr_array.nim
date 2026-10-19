## Vulkan nested pointer collection
##
## Test that an array of references whose pointee appears in no other
## pointer surface still gets its reference block and int8 extensions declared
##
## Run:
##   cd tattletale && \
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/device/t_vulkan_nested_ptr_array.nim

import std/strformat
import workspace/crucible

const kernelCode = vulkan:
  proc k(sink: ptr UncheckedArray[uint32], ps: array[4, ptr UncheckedArray[uint64]]) {.global.} =
    ps[0][1] = ps[2][3]
    sink[0] = uint32(ps[0][1])

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  block:
    var engine = bkVulkan.init()
    engine.ingest(kernelCode)
    let sDev = engine.allocDevice(uint32, 1)
    var devs = [engine.allocDevice(uint64, 4), engine.allocDevice(uint64, 4),
                engine.allocDevice(uint64, 4), engine.allocDevice(uint64, 4)]
    var seed: array[4, uint64] = [7'u64, 8'u64, 9'u64, 10'u64]
    for i in 0 ..< 4:
      engine.copyToDeviceFromHost(devs[i], addr seed[0], seed.len)
    engine.launchKernel("k", sDev, (devs[0], devs[1], devs[2], devs[3]))
    var got: uint32
    engine.copyToHostFromDevice(addr got, sDev, 1)
    for i in 0 ..< 4:
    doAssert got == 10'u32, &"ps[0][1] = ps[2][3]: {got} != 10"
    echo "  OK — reference block declared from a nested array-of-refs pointee"

when isMainModule:
  runTest()
