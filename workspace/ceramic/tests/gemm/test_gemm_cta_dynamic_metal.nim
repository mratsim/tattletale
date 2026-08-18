## Unified GEMM entry: Metal compile-gate.
##
## The unified manual_gemm_cta_dynamic entry compiles its portable
## coordinates + synchronization slice for Metal (the tensor-core atom is
## NVIDIA-gated and out of scope for this sprint). This test ingests the generated MSL
## and pins the native spellings, validating the compilation on this machine:
## the canonical names are the MSL attribute names
## (`thread_position_in_grid [[thread_position_in_grid]]`) and the barrier
## lowers to `threadgroup_barrier(mem_flags::mem_threadgroup)`. The 128
## work-items are well under Metal's 1024-thread dispatch limit.
## This gate pins the slice compile. The shared-memory + barrier pattern
## is executed by test_metal_cross_vocabulary.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/gemm --nimcache:nimcache/tests/gemm \
##     workspace/ceramic/tests/gemm/test_gemm_cta_dynamic_metal.nim

import std/strutils
import workspace/ceramic/tests/gemm/manual_gemm_cta_dynamic
import workspace/crucible

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # The unified dispatch uses the native MSL attribute name.
  doAssert "thread_position_in_grid [[thread_position_in_grid]]" in kernelCodeMetal,
    "Metal thread-position attribute missing:\n" & kernelCodeMetal
  doAssert "threadgroup_barrier(mem_flags::mem_threadgroup)" in kernelCodeMetal,
    "Metal barrier lowering missing:\n" & kernelCodeMetal

  var engine = bkMetal.init()
  engine.ingest(kernelCodeMetal)
  echo "  OK — unified GEMM Metal compile-gate (ingest + source inspect)"

when isMainModule:
  runTest()
