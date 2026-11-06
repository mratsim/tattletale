## HLSL: a recursive device function refuses at compile time.
##
## Test that a `{.device.}` function calling itself dies in the HLSL call-graph
## pass before any shader is emitted. The refusal names the function:
##
##   Error: recursive device function `f` would not terminate, recursion is rejected
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/functions/manual_hlsl_recursive_device_fn_refused.nim

import workspace/crucible

const recursiveHlsl = hlsl:
  proc f(x: uint32): uint32 {.device.} =
    f(x)

  proc kern(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = f(1'u32)

proc runTest() =
  var engine = newEngine(runVulkanHlsl)
  engine.ingest(recursiveHlsl)
  echo "UNREACHABLE — the recursive device function must be refused at compile time"

when isMainModule:
  runTest()
