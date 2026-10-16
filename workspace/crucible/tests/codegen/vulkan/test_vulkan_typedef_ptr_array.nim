## Vulkan end-to-end: struct typedefs with array fields.
##
## Run:
##   cd tattletale && \
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_typedef_ptr_array.nim

import std/strformat
import workspace/crucible

type
  Band = object
    lo, hi: uint32

  Bands = object
    edges: array[2, uint32]
    band: Band
    m: uint32

proc takeBands(w: Bands): uint32 {.device.} = w.edges[1] + w.band.hi + w.m

const kernelCode = vulkan:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = takeBands(Bands(edges: [10'u32, 20'u32],
                               band: Band(lo: 1'u32, hi: 2'u32), m: 5'u32))

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  block:
    var engine = bkVulkan.init()
    engine.ingest(kernelCode)
    let oDev = engine.allocDevice(sizeof(uint32))
    engine.launchKernel("k", oDev, ())
    var res: array[1, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
    engine.freeDevice(oDev)
    doAssert res[0] == 27, &"edges[1] + band.hi + m: {res[0]} != 27"
    echo "  OK — struct with array field through a device function (Vulkan)"

when isMainModule:
  runTest()
