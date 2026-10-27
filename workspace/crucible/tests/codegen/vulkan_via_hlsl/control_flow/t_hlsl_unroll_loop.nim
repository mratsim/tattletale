## HLSL: `{.unroll.}` marks a device loop for full unrolling
##
## Test that a kernel with an `{.unroll.}`-marked loop compiles through
## the DXC route with the `[unroll]` attribute in the emitted source
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/control_flow/t_hlsl_unroll_loop.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const unrollHlsl = hlsl:
  proc unrollKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 4:
      {.unroll.}
      output[i] = uint32(i)

proc runTest() =
  let bindings: seq[BindingExpectation] = @[("output", 0, false)]
  discard compileCheck("unroll_loop", unrollHlsl, bindings, expect = "[unroll]")
  echo "  OK - t_hlsl_unroll_loop"

when isMainModule:
  runTest()
