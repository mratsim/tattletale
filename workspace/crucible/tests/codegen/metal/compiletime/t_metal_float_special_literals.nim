## Metal: folded float32 ±Inf and NaN constants emit valid MSL tokens
##
## Test that float32 special-value constants (±Inf, NaN) emit the MSL
## INFINITY/-INFINITY/NAN tokens, compile on the device, and carry the true
## IEEE-754 bit patterns
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/compiletime/t_metal_float_special_literals.nim

import std/[math, strutils, unittest]
import workspace/crucible

const kernelSpecials = metal:
  proc specials(O: ptr UncheckedArray[float32]) {.global.} =
    O[0] = float32(-Inf)
    O[1] = float32(Inf)
    O[2] = float32(NaN)

const kernelFoldedDiv = metal:
  proc foldedDiv(O: ptr UncheckedArray[float32]) {.global.} =
    const negInf {.genSym.} = -1.0'f32/0.0'f32
    const posInf {.genSym.} = 1.0'f32/0.0'f32
    O[0] = negInf
    O[1] = posInf

proc bits32(x: float32): uint32 =
  var y = x
  result = cast[ptr uint32](addr y)[]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - float special-value literals":
    test "±Inf and NaN emit INFINITY/-INFINITY/NAN, never inff/nanf":
      check "-INFINITY" in kernelSpecials
      check "INFINITY" in kernelSpecials
      check "NAN" in kernelSpecials
      check "inff" notin kernelSpecials
      check "nanf" notin kernelSpecials
      check "inff" notin kernelFoldedDiv
    test "device values carry the true ±Inf bit patterns":
      var engine = newEngine(runMetal)
      engine.ingest(kernelSpecials)
      let oDev = engine.allocDevice(float32, 3)
      engine.launchKernel("specials", oDev, ())
      var res: array[3, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 3)
      check bits32(res[0]) == 0xFF800000'u32
      check bits32(res[1]) == 0x7F800000'u32
      check classify(res[2]) == fcNan
    test "folded ±1/0 constants compile and reach the device as ±Inf":
      var engine = newEngine(runMetal)
      engine.ingest(kernelFoldedDiv)
      let oDev = engine.allocDevice(float32, 2)
      engine.launchKernel("foldedDiv", oDev, ())
      var res: array[2, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check bits32(res[0]) == 0xFF800000'u32
      check bits32(res[1]) == 0x7F800000'u32

when isMainModule:
  runTest()
