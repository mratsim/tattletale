## Vulkan int8 extension from a byte view
##
## Test that a cast to a byte pointer pulls in the int8 extensions even
## though no parameter or return type carries uint8, the byte view writes
## the top byte of the word
##
## Run:
##   cd tattletale && \
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/device/t_vulkan_cast_byte_view.nim

import std/strformat
import workspace/crucible

const kernelCode = vulkan:
  proc k(outP: ptr UncheckedArray[uint32]) {.global.} =
    let bytes = cast[ptr UncheckedArray[uint8]](outP)
    bytes[3] = uint8(outP[0] + 1)
    outP[0] = outP[0] + 1

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  block:
    var engine = bkVulkan.init()
    engine.ingest(kernelCode)
    let oDev = engine.allocDevice(sizeof(uint32))
    var zero: uint32 = 0
    engine.copyToDeviceFromHost(oDev, addr zero, sizeof(uint32))
    engine.launchKernel("k", oDev, ())
    var got: uint32
    engine.copyToHostFromDevice(addr got, oDev, sizeof(uint32))
    engine.freeDevice(oDev)
    doAssert got == 16777217'u32, &"byte view wrote the top byte: {got} != 16777217"
    echo "  OK — byte view through a cast pulls in the int8 extensions"

when isMainModule:
  runTest()
