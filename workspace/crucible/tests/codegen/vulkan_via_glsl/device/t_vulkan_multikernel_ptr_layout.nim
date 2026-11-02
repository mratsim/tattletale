## Vulkan: multi-kernel pointer-only sources with differing param counts.
##
## Two kernels with differing pointer counts share the module-wide
## push-constant block and serializes its own addresses from offset 0.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/t_vulkan_multikernel_ptr_layout.nim

import std/strformat
import workspace/crucible

const code = glsl:
  proc k1(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 1'u32
  proc k2(output, extra: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = extra[0]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo "=== Vulkan: multi-kernel pointer-only push-constant sharing ===\n"

  var engine = newEngine(runVulkan)
  engine.ingest(code)
  let oDev = engine.allocDevice(uint32, 1)
  let eDev = engine.allocDevice(uint32, 1)

  block:  # k1 writes its literal, one buffer address at offset 0
    var seed: array[1, uint32] = [0xDEADBEEF'u32]
    engine.copyToDeviceFromHost(oDev, addr seed[0], 1)
    engine.launchKernel("k1", oDev, ())
    var got: array[1, uint32]
    engine.copyToHostFromDevice(addr got[0], oDev, 1)
    doAssert got[0] == 1'u32, &"k1: got {got[0]}, expected 1"
    echo "  OK — k1 writes 1"

  block:  # k2 reads extra[0] through the second block member, offset 8
    var seedOut: array[1, uint32] = [0xDEADBEEF'u32]
    var seedExtra: array[1, uint32] = [77'u32]
    engine.copyToDeviceFromHost(oDev, addr seedOut[0], 1)
    engine.copyToDeviceFromHost(eDev, addr seedExtra[0], 1)
    engine.launchKernel("k2", oDev, (eDev,))
    var got: array[1, uint32]
    engine.copyToHostFromDevice(addr got[0], oDev, 1)
    doAssert got[0] == 77'u32, &"k2: got {got[0]}, expected 77"
    echo "  OK — k2 copies extra[0] = ", got[0]

when isMainModule:
  runTest()
