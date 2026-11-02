## Vulkan: multi-kernel push-constant sharing in the positive form.
##
## Test that two kernels in one `glsl:` source with the same ordered params
## share one push-constant block, and that each launch reads its arguments
## at the correct offsets.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/t_vulkan_multikernel_pushconst_poc.nim

import std/strformat
import workspace/crucible

# Identical ordered params: the module-wide push-constant block keeps
# one offset per member, and both kernels read the same offsets.
const codeMulti = glsl:
  proc kernelMul(output: ptr UncheckedArray[uint32], a: uint32, b: uint32) {.global.} =
    output[0] = a * b
  proc kernelAdd(output: ptr UncheckedArray[uint32], a: uint32, b: uint32) {.global.} =
    output[0] = a + b

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo "=== Vulkan: multi-kernel push-constant sharing (same layout) ===\n"

  var engine = newEngine(runVulkan)
  engine.ingest(codeMulti)
  let oDev = engine.allocDevice(uint32, 1)
  var gpuOut: array[1, uint32] = [0xDEADBEEF'u32]

  engine.launchKernel("kernelMul", oDev, (3'u32, 4'u32,))
  engine.copyToHostFromDevice(addr gpuOut[0], oDev, 1)
  doAssert gpuOut[0] == 12'u32,
    &"kernelMul(3, 4) through the shared push-constant block: got {gpuOut[0]}, expected 12"

  engine.launchKernel("kernelAdd", oDev, (3'u32, 4'u32,))
  engine.copyToHostFromDevice(addr gpuOut[0], oDev, 1)
  doAssert gpuOut[0] == 7'u32,
    &"kernelAdd(3, 4) through the shared push-constant block: got {gpuOut[0]}, expected 7"

  echo &"  OK — kernelMul(3, 4) = 12, kernelAdd(3, 4) = 7, one shared push-constant block"

when isMainModule:
  runTest()
