## GLSL: the feature scan reaches a call nested under `discard`.
##
## Test that a subgroup call inside a discarded expression still selects the subgroup extension.
## The scan must walk the discarded expression, otherwise the kernel references `subgroupAdd` without its `#extension` lines.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/compiletime/t_glsl_discard_call_scan.nim

import std/[strutils, unittest]
import workspace/crucible

const discardScanGlsl = glsl:
  proc discardScanKernel(outBuf: ptr UncheckedArray[float32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let v = float32(lane)
    discard 0.0'f32 + reduceAdd(v)
    outBuf[lane] = v

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "GLSL - feature scan under discard":

    test "a subgroup call under discard selects the subgroup extension":
      check "#extension GL_KHR_shader_subgroup_basic : enable" in discardScanGlsl

when isMainModule:
  runTest()
  echo "t_glsl_discard_call_scan: all checks passed"
