## GLSL: a pointer local rebound inside a branch refuses at compile time.
##
## Test that a pointer local rebound inside an `if` refuses to compile
## with the pointer-local dissolve refusal.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/control_flow/manual_vulkan_ptr_local_rebind.nim

import workspace/crucible

const code = glsl:
  proc k(dst: ptr UncheckedArray[uint32], flag: uint32) {.global.} =
    var x: array[2, uint32] = [11'u32, 12'u32]
    var y: array[2, uint32] = [22'u32, 23'u32]
    var p = x.addr
    if flag == 1'u32:
      p = y.addr
    dst[0] = p[0]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runVulkan)
  engine.ingest(code)
  echo "UNREACHABLE — a rebinding of a pointer local must refuse at compile time"

when isMainModule:
  runTest()
