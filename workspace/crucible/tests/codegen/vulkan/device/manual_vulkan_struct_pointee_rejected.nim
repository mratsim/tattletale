## Vulkan pointer-to-object arguments
##
## Test that a kernel argument pointing at an object fails to compile,
## pointer arguments expose plain values: pass the object's fields separately
##
## Run:
##   cd tattletale
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/device/manual_vulkan_struct_pointee_rejected.nim

import workspace/crucible

type Pair = object
  a: int32
  b: int32

const code = vulkan:
  proc k(output: ptr Pair) {.global.} =
    discard

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = bkVulkan.init()
  engine.ingest(code)
  echo "UNREACHABLE — object pointee refs must be rejected at compile time"

when isMainModule:
  runTest()
