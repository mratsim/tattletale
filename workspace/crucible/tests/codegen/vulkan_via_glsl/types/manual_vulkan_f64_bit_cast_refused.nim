## GLSL: a bit-reinterpret cast over a 64-bit float refuses at compile time.
##
## Test that a `cast[uint64]` over a float64 value in a glsl: kernel dies in the GLSL printer before any shader source is emitted.
## The refusal names the pass:
##
##   Error: cirv2 printer glsl: at kern: bit cast of a 64-bit float has no GLSL emission, only the 32-bit floatBits builtins
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/types/manual_vulkan_f64_bit_cast_refused.nim

import workspace/crucible

const f64BitCastGlsl = glsl:
  proc kern(output: ptr UncheckedArray[uint64], input: ptr UncheckedArray[float64]) {.global, workgroup: (4, 1, 1).} =
    output[0] = cast[uint64](input[0])

proc runTest() =
  var engine = newEngine(runVulkan)
  engine.ingest(f64BitCastGlsl)
  echo "UNREACHABLE — the 64-bit float bit cast must be refused at compile time"

when isMainModule:
  runTest()
