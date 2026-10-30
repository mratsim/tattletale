## Metal: zero-arg object constructors compile and zero-initialize.
##
## Test that a constructor with every field omitted (`Pair()`) compiles
## and each field reads back as zero, on the device.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_zero_arg_constructor.nim

import std/unittest
import workspace/crucible

type
  Pair = object
    x: int32
    y: int32
  Triple = object
    a: int32
    b: float32
    c: uint32

const zeroArgMsl = metal:
  proc zeroArgKernel(o: ptr UncheckedArray[int32]) {.global.} =
    let p = Pair()
    let t = Triple()
    o[0] = p.x
    o[1] = p.y
    o[2] = int32(t.b)
    o[3] = int32(t.c)

const mixedMsl = metal:
  proc mixedKernel(o: ptr UncheckedArray[int32]) {.global.} =
    # one field written, every omitted field zero-fills
    let p = Pair(x: 7)
    o[0] = p.x
    o[1] = p.y

proc runTest() =
  suite "Metal - zero-arg object constructor":
    test "Pair() and Triple() compile and every field reads back zero":
      var engine = newEngine(runMetal)
      engine.ingest(zeroArgMsl)
      let oDev = engine.allocDevice(int32, 4)
      engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("zeroArgKernel", oDev, ())
      var res: array[4, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res[0] == 0
      check res[1] == 0
      check res[2] == 0
      check res[3] == 0

    test "Pair(x: 7) zero-fills the omitted y field":
      var engine = newEngine(runMetal)
      engine.ingest(mixedMsl)
      let oDev = engine.allocDevice(int32, 2)
      engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("mixedKernel", oDev, ())
      var res: array[2, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 7
      check res[1] == 0

when isMainModule:
  runTest()
