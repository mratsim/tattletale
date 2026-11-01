## Vulkan: a multi-kernel source whose kernels declare different scalar
## push-constant params is rejected at codegen. The push block is shared
## by every kernel of the source, a differing ordered layout would read
## args at wrong offsets.
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
