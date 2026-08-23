## Arithmetic array lengths: `array[A * B, T]` resolves its length.
##
## Anti-regression for `resolveArrayLength` (the `evalConstInt` rewrite):
## the length node of a statically sized array is not always a `0..N` range —
## `getTypeInst` keeps product expressions like `2 * 2` unfolded, and the
## old resolver doAsserted on any length whose range start was not 0.
## The pulled-in-proc signature typing reaches the array type, so the
## arithmetic length must evaluate.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_array_length_arith.nim

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
      var res: array[1, float32]
      engine.run("repro", res, ())
      check res[0] == 3.0'f32

when isMainModule:
  runTest()
