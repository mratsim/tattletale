## Vulkan void pointer arguments
##
## Test that a kernel argument of type ptr void fails to compile: there is
## nothing to point at, view the bytes through ptr UncheckedArray[uint8]
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/manual_vulkan_void_pointee_rejected.nim

import workspace/crucible

const code = vulkan:
  proc k(output: ptr void) {.global.} =
    discard

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = bkVulkan.newEngine()
  engine.ingest(code)
  echo "UNREACHABLE — void pointee refs must be rejected at compile time"

when isMainModule:
  runTest()
