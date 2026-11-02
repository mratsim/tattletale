## Vulkan: even identically-ordered multi-kernel scalar params refuse.
##
## Test that a multi-kernel source whose kernels declare the same ordered
## by-value scalar params still refuses at engine ingest: the restored
## Vulkan route allows one kernel per source when by-value scalars pack
## through the shared push-constant block, only buffer references share.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/manual_vulkan_multikernel_scalar_refused.nim

import workspace/crucible

const code = glsl:
  proc kernelMul(output: ptr UncheckedArray[uint32], a: uint32, b: uint32) {.global.} =
    output[0] = a * b
  proc kernelAdd(output: ptr UncheckedArray[uint32], a: uint32, b: uint32) {.global.} =
    output[0] = a + b

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runVulkan)
  engine.ingest(code)
  echo "UNREACHABLE — multi-kernel sources with by-value scalar params must be refused at ingest"

when isMainModule:
  runTest()
