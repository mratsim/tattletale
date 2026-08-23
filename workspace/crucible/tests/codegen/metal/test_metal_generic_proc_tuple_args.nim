## Tuples: ensure anonymous tuples in generics generate the expected type name
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_generic_proc_tuple_args.nim

import std/unittest
import workspace/crucible

type Int*[V: static int] = object

type Box*[T; Sh] = object
  data: array[2, T]

proc scaleBox*[V: static int](dst: var Box[float32, (Int[V], Int[2])],
                              src: Box[float32, (Int[V], Int[2])],
                              s: float32) {.inline.} =
  dst.data[0] = src.data[0] * s
  dst.data[1] = src.data[1] * s

const kernelMsl = metal:
  proc repro(outp: ptr UncheckedArray[float32]) {.global.} =
    var d: Box[float32, (Int[1], Int[2])]
    let a = Box[float32, (Int[1], Int[2])](data: [2.0'f32, 3.0'f32])
    d.scaleBox(a, 3.0'f32)
    outp[0] = d.data[0]
    outp[1] = d.data[1]

proc runTest() =
  suite "Metal - generic proc with tuple-of-static-int type args":
    test "scaleBox lowers with one type name for the tuple args":
      var engine = bkMetal.init()
      engine.ingest(kernelMsl)
      var res: array[2, float32]
      engine.run("repro", res, ())
      check res[0] == 6.0'f32
      check res[1] == 9.0'f32

when isMainModule:
  runTest()
