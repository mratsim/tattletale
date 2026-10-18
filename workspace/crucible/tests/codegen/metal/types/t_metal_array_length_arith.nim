## Arithmetic array lengths: `array[A * B, T]` resolves its length.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_array_length_arith.nim

import std/unittest
import workspace/crucible

proc sumCorners*(xs: array[2 * 2, float32]): float32 =
  ## Pulled into device code: its signature carries the arithmetic-length
  ## array type, so resolving the signature resolves `array[2 * 2, float32]`.
  xs[0] + xs[3]

const kernelMsl = metal:
  proc repro(outp: ptr UncheckedArray[float32]) {.global.} =
    var xs: array[2 * 2, float32]
    xs[0] = 1.0'f32
    xs[3] = 2.0'f32
    outp[0] = sumCorners(xs)

proc runTest() =
  suite "Metal - arithmetic array lengths":
    test "array[2 * 2, T] resolves its length in a pulled-in proc signature":
      var engine = bkMetal.init()
      engine.ingest(kernelMsl)
      let oDev = engine.allocDevice(float32, 1)
      engine.launchKernel("repro", oDev, ())
      var res: array[1, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      engine.freeDevice(oDev)
      check res[0] == 3.0'f32

when isMainModule:
  runTest()
