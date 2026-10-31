## Vulkan: a var-returning-call compound-assign lvalue refuses on GLSL.
##
## Test that `slotAt(x, k) += 5` over a `var int32` device function
## refuses on GLSL: the lvalue's pointer temp has no GLSL function storage.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/control_flow/manual_vulkan_varcall_lvalue_refused.nim

import workspace/crucible

const code = glsl:
  proc slotAt(x: ptr UncheckedArray[int32], k: int32): var int32 {.device.} =
    return x[k]

  proc k(output: ptr UncheckedArray[int32]) {.global.} =
    slotAt(output, 0) += 5

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runVulkan)
  engine.ingest(code)
  echo "UNREACHABLE: the pointer temp must refuse in glslDissolvePointerLocals"

when isMainModule:
  runTest()
