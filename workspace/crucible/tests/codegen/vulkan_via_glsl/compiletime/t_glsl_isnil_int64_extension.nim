## GLSL: an isNil check is the module's only 64-bit use.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/compiletime/t_glsl_isnil_int64_extension.nim

import std/strutils
import workspace/crucible

const isnilGlsl = glsl:
  proc isnilKernel(outBuf: ptr UncheckedArray[float32]) {.global.} =
    if outBuf.isNil():
      return
    outBuf[0] = 1.0'f32

const plainGlsl = glsl:
  proc plainKernel(outBuf: ptr UncheckedArray[float32]) {.global.} =
    outBuf[0] = 1.0'f32

proc runTest() =
  const int64Extension =
    "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable"
  doAssert int64Extension in isnilGlsl,
    "isNil prints a uint64_t comparison, the int64 extension line is missing:\n" & isnilGlsl
  doAssert int64Extension notin plainGlsl,
    "a kernel with no isNil and no 64-bit type must not request the int64 extension:\n" & plainGlsl

  var engine = newEngine(runVulkan)
  engine.ingest(isnilGlsl)
  let oDev = engine.allocDevice(float32, 1)
  engine.launchKernel("isnilKernel", oDev, ())
  var got: array[1, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 1.0'f32,
    "the isnil kernel ran: got " & $got[0] & ", want 1.0"
  echo "  OK - t_glsl_isnil_int64_extension"

when isMainModule:
  runTest()
