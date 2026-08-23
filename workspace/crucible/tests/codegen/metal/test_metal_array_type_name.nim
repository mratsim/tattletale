## Array types inside pulled-in proc signatures name without recursing.
##
## Anti-regression for the `assignTypeName` array case (`Array_N_T`):
## naming a type that contains an array (here a tuple field) recursed
## through the `array` symbol — whose type inst is the array itself — until
## the VM call-depth limit. Array types must name from their length and
## element type instead.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_array_type_name.nim

import std/unittest
import workspace/crucible

proc probeTuple*(p: tuple[a: array[5, int32], b: int32]): int32 =
  ## Pulled into device code: naming its signature names the anonymous
  ## tuple, whose `a` field is the array type.
  p.a[0] + p.b

const kernelMsl = metal:
  proc repro(outp: ptr UncheckedArray[int32]) {.global.} =
    var p: tuple[a: array[5, int32], b: int32]
    p.a[0] = 7
    p.b = 1
    outp[0] = probeTuple(p)

proc runTest() =
  suite "Metal - array types in proc signatures":
    test "a tuple-of-array param names without recursion":
      var engine = bkMetal.init()
      engine.ingest(kernelMsl)
      var res: array[1, int32]
      engine.run("repro", res, ())
      check res[0] == 8'i32

when isMainModule:
  runTest()
