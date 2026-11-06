##
## Vulkan: entry-point parse with a void-returning device helper.
## Entry-point resolution must return the first kernel's name, never a device-helper name.
## Each kernel opens with its own `layout(local_size_x ...) in;` preamble.
## Entry point resolution anchors on that preamble, not on the first bare `void `.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/functions/t_vulkan_entrypoint_helper.nim

import std/[strformat, strutils]
import workspace/crucible

# A void-returning device function gets forward-declared
# (`void helper(uint x);`) before the kernel's preamble, exactly the shape
# that breaks a naive first-`void ` entry-point scan.
const helperVk = glsl:
  proc helper(x: uint32) {.device.} =
    let y = x + 1

  proc entryKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    helper(41'u32)
    output[0] = 42'u32

proc runTest() =
  echo "=== Vulkan: entry point with void-returning device helper ===\n"

  echo helperVk
  echo ""

  var engine = newEngine(runVulkan)
  engine.ingest(helperVk)

  # A wrong entry point fails ingest (glslang `-e helper`) or silently
  # recompiles per launch.
  var gpuOut: array[1, uint32] = [0xDEADBEEF'u32]
  let oDev = engine.allocDevice(uint32, 1)
  engine.copyToDeviceFromHost(oDev, addr gpuOut[0], 1)
  engine.launchKernel("entryKernel", oDev, ())
  engine.copyToHostFromDevice(addr gpuOut[0], oDev, 1)
  echo &"  entryKernel: got {gpuOut[0]}, expected 42"
  doAssert gpuOut[0] == 42, &"entryKernel: got {gpuOut[0]}, expected 42"

  echo "  OK: entry point parsed as the kernel name, fast path fires"

when isMainModule:
  runTest()
