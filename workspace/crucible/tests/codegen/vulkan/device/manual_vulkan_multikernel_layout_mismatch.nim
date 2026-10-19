## Vulkan: a multi-kernel source whose kernels declare different push-constant
## params is rejected at codegen. The push block is shared by every kernel.
## Differing ordered layouts would read args at wrong offsets.
##
## Run:
##   cd tattletale && \
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/device/manual_vulkan_multikernel_layout_mismatch.nim

import workspace/crucible

const code = vulkan:
  proc k1(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 1'u32
  proc k2(output, extra: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = extra[0]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = bkVulkan.newEngine()
  engine.ingest(code)
  echo "UNREACHABLE — differing multi-kernel push-constant layouts must be rejected at codegen"

when isMainModule:
  runTest()
