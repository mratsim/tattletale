## Metal: construction-then-field projection lowering.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_constr_projection.nim

import workspace/crucible

import std/unittest

type P2 = object
  a, b: int32

const code = metal:
  proc projKernel(output: ptr UncheckedArray[int32]) {.global.} =
    let i = thread_position_in_grid.x
    output[0] = P2(a: int32(i), b: 7).b
    output[1] = (int32(i), 7'i32)[1]
    output[2] = P2(a: 1, b: int32(i)).a * 2
    output[3] = P2(a: int32(i), b: 3).b + P2(a: 5, b: int32(i)).a

proc runProj(i: int32): array[4, int32] =
  ## One launch per call: device buffers serve a single launch so handles never need a copy.
  var engine = newEngine(runMetal)
  engine.ingest(code)
  let oDev = engine.allocDevice(int32, 4)
  engine.launchKernel("projKernel", oDev, ())
  engine.copyToHostFromDevice(result.toOpenArray(0, 4 - 1), oDev)

proc runTest() =
  suite "Metal - construction-then-field projection":
    test "projecting a field of a fresh constructor value":
      let r = runProj(3'i32)
      check r[0] == 7
      check r[1] == 7
      check r[2] == 2
      check r[3] == 8
      let r9 = runProj(9'i32)
      check r9[0] == 7
      check r9[1] == 7
      check r9[2] == 2
      check r9[3] == 8

when isMainModule:
  runTest()
