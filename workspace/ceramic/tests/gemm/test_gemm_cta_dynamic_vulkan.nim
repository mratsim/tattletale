## Unified GEMM entry: Vulkan compile-gate.
##
## The unified manual_gemm_cta_dynamic entry compiles its portable
## coordinates + synchronization slice for Vulkan (the tensor-core atom is
## NVIDIA-gated and out of scope for this sprint). This test ingests the generated GLSL
## and pins the native spellings, validating the compilation on this machine:
## the flat global id lowers to `gl_GlobalInvocationID.x` and the barrier to `barrier()`.
## The baked `{.workgroup: (128, 1, 1).}` becomes `layout(local_size_x = 128, ...)`.
## This gate pins the slice compile. The shared-memory + barrier pattern
## is executed by test_vulkan_cross_vocabulary_shared.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/gemm --nimcache:nimcache/tests/gemm \
##     workspace/ceramic/tests/gemm/test_gemm_cta_dynamic_vulkan.nim

import std/strutils
import workspace/ceramic/tests/gemm/manual_gemm_cta_dynamic
import workspace/crucible

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # The unified dispatch lowers to gl_GlobalInvocationID.x and barrier().
  doAssert "gl_GlobalInvocationID.x" in kernelCodeVulkan,
    "Vulkan global-id lowering missing:\n" & kernelCodeVulkan
  doAssert "barrier()" in kernelCodeVulkan,
    "Vulkan barrier lowering missing:\n" & kernelCodeVulkan
  # The baked (128, 1, 1) workgroup size.
  doAssert "layout(local_size_x = 128, local_size_y = 1, local_size_z = 1)" in kernelCodeVulkan,
    "Vulkan baked workgroup size missing:\n" & kernelCodeVulkan

  var engine = bkVulkan.init()
  engine.ingest(kernelCodeVulkan)
  echo "  OK — unified GEMM Vulkan compile-gate (ingest + source inspect)"

when isMainModule:
  runTest()
