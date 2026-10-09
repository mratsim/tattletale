## Vulkan end-to-end: struct typedefs with array fields.
##
## - the pointer scan keeps pointer-free typedefs alive
## - a pointer-free struct with an array field compiles under glslangValidator
##   and computes on the device
## - a pointer-bearing struct behind an array field has no valid emission:
##   as a kernel param the explosion pass fails compilation:
##   manual_explode_array_ptr_field.nim covers that case
##
## Run:
##   nim c -r --hints:off --warnings:off workspace/crucible/tests/codegen/vulkan/test_vulkan_typedef_ptr_array.nim (from tattletale)

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
    var res: array[1, uint32]
    engine.run("k", res, ())
    doAssert res[0] == 27, &"edges[1] + band.hi + m: {res[0]} != 27"
    echo "  OK — struct with array field through a device function (Vulkan)"

when isMainModule:
  runTest()
