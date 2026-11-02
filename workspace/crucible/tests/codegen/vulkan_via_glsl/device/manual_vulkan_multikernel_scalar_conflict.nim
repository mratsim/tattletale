## Vulkan: a multi-kernel source with a by-value scalar in the shared
## push-constant block is refused at engine ingest. A scalar member's
## offset must be launch-invariant, one kernel's scalar sits between
## another kernel's block members.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/manual_vulkan_multikernel_scalar_conflict.nim

import workspace/crucible

const code = glsl:
  proc kernelA(output: ptr UncheckedArray[uint32], val: uint32) {.global.} =
    output[0] = val
  proc kernelB(output: ptr UncheckedArray[uint32], a: uint32, b: uint32) {.global.} =
    output[0] = a + b

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runVulkan)
  engine.ingest(code)
  echo "UNREACHABLE — differing multi-kernel push-constant layouts must be rejected at codegen"

when isMainModule:
  runTest()
