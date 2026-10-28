## Vulkan (GLSL): `{.unroll.}` marks a device loop for full unrolling
##
## Test that a kernel with an `{.unroll.}`-marked loop compiles through
## glslang via `[[unroll]]` (GL_EXT_control_flow_attributes) and the loop
## computes the right result
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/control_flow/t_vulkan_unroll_loop.nim

import std/strutils
import workspace/crucible

const unrollGlsl = glsl:
  proc unrollKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 4:
      {.unroll.}
      output[i] = uint32(i)

proc runTest() =
  var engine = newEngine(runVulkan)
  engine.ingest(unrollGlsl)
  doAssert "[[unroll]]" in engine.getArtifact(), "unroll pragma missing in the generated code"
  let oDev = engine.allocDevice(uint32, 4)
  engine.launchKernel("unrollKernel", oDev, ())
  var res: array[4, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, 4)
  for i in 0 ..< 4:
    doAssert res[i] == uint32(i), "output[" & $i & "] = " & $res[i]
  echo "  OK - t_vulkan_unroll_loop"

when isMainModule:
  runTest()
