## Vulkan end-to-end: a runtime array of fixed 32-byte blocks through one
## buffer reference (`ptr UncheckedArray[array[32, byte]]`), the sha256 shape.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/t_vulkan_ptr_array_of_arrays.nim

import std/strformat
import workspace/crucible

const kernelCode = vulkan:
  proc k(outP: ptr UncheckedArray[array[32, byte]]) {.global.} =
    # outP[i][j] = uint8(i + j), the element type is a fixed 32-byte block
    for i in 0 ..< 2:
      for j in 0 ..< 32:
        outP[i][j] = byte(i * 32 + j)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  block:
    var engine = newEngine(runVulkan)
    engine.ingest(kernelCode)
    let oDev = engine.allocDevice(array[32, byte], 2)
    engine.launchKernel("k", oDev, ())
    var res: array[64, byte]
    engine.copyToHostFromDevice(cast[ptr array[32, byte]](addr res[0]), oDev, 2)
    for i in 0 ..< 64:
      doAssert res[i] == byte(i), &"digest byte {i}: {res[i]} != {i}"
    echo "  OK — runtime array of byte[32] blocks (sha256 shape) through a Vulkan ref block"

when isMainModule:
  runTest()
