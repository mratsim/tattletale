## Unified GEMM entry: WebGPU compile-gate.
##
## The unified manual_gemm_cta_dynamic entry compiles its portable
## coordinates + synchronization slice for WebGPU (the tensor-core atom is
## NVIDIA-gated and out of scope for this sprint). This test ingests the generated WGSL
## and pins the native spellings, validating the compilation on this machine:
## the flat global id lowers to the injected `@builtin(global_invocation_id)`
## param and the barrier to `workgroupBarrier()`.
## The baked `{.workgroup: (128, 1, 1).}` becomes `@workgroup_size(128, 1, 1)`.
## This gate pins the slice compile. WebGPU has no shared-memory emission
## (the workgroup address space is out of scope). The memory pattern is not exercised here.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/gemm --nimcache:nimcache/tests/gemm \
##     workspace/ceramic/tests/gemm/test_gemm_cta_dynamic_webgpu.nim

import std/strutils
import workspace/ceramic/tests/gemm/manual_gemm_cta_dynamic
import workspace/crucible

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # The unified dispatch lowers to the injected global_invocation_id param.
  doAssert "@builtin(global_invocation_id)" in kernelCodeWebGPU,
    "WebGPU global-id injection missing:\n" & kernelCodeWebGPU
  doAssert "workgroupBarrier()" in kernelCodeWebGPU,
    "WebGPU barrier lowering missing:\n" & kernelCodeWebGPU
  # The baked (128, 1, 1) workgroup size.
  doAssert "@workgroup_size(128, 1, 1)" in kernelCodeWebGPU,
    "WebGPU baked workgroup size missing:\n" & kernelCodeWebGPU

  var engine = bkWGSL.init()
  engine.ingest(kernelCodeWebGPU)
  echo "  OK — unified GEMM WebGPU compile-gate (ingest + source inspect)"

when isMainModule:
  runTest()
