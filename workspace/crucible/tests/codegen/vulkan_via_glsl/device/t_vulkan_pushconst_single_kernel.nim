## Vulkan: single-kernel push-constant scalars.
##
## Test that single-kernel by-value scalars pack into the KernelParams
## push-constant block and the launch reads back the computed results.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/t_vulkan_pushconst_single_kernel.nim

import std/strformat
import workspace/crucible

const codeSingle = glsl:
  proc kernelB(output: ptr UncheckedArray[uint32], a: uint32, b: uint32) {.global.} =
    output[0] = a + b

const codeWithVal = glsl:
  proc kernelWithVal(output: ptr UncheckedArray[uint32], val: uint32) {.global.} =
    output[0] = val

proc runTest() =
  echo "=== Vulkan: single-kernel push-constant scalars ===\n"

  block:  # kernelB(3, 4) gives 7
    var engine = newEngine(runVulkan)
    engine.ingest(codeSingle)
    var gpuOut: array[1, uint32] = [0xDEADBEEF'u32]
    let oDev = engine.allocDevice(uint32, 1)
    engine.copyToDeviceFromHost(oDev, gpuOut.toOpenArray(0, 1 - 1))
    engine.launchKernel("kernelB", oDev, (3'u32, 4'u32,))
    engine.copyToHostFromDevice(gpuOut.toOpenArray(0, 1 - 1), oDev)
    doAssert gpuOut[0] == 7, &"kernelB(3, 4): got {gpuOut[0]}, expected 7"
    echo "  OK — kernelB(3, 4) = ", gpuOut[0]

  block:  # single by-value scalar
    var engine = newEngine(runVulkan)
    engine.ingest(codeWithVal)
    var gpuOut: array[1, uint32] = [0xDEADBEEF'u32]
    let oDev = engine.allocDevice(uint32, 1)
    engine.copyToDeviceFromHost(oDev, gpuOut.toOpenArray(0, 1 - 1))
    engine.launchKernel("kernelWithVal", oDev, (42'u32,))
    engine.copyToHostFromDevice(gpuOut.toOpenArray(0, 1 - 1), oDev)
    doAssert gpuOut[0] == 42, &"kernelWithVal(42): got {gpuOut[0]}, expected 42"
    echo "  OK — kernelWithVal(42) = ", gpuOut[0]

when isMainModule:
  runTest()
