## Metal: structs with dummy/empty fields. The object constructor emits
## a valid MSL initializer, never bare braces, for a single dummy struct
## const and for a tuple of two dummy struct consts.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_dummy_init.nim

import std/[strutils, unittest]
import workspace/crucible

type
  FixMe*[V: static int] = object

const kernelCode = metal:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

const kernelCode2 = metal:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (FixMe[1](), FixMe[8]())
    C[0] = 1'u32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - dummy-field initializers":
    test "single dummy struct const":
      var engine = newEngine(runMetal)
      engine.ingest(kernelCode)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("dummyKernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 1

    test "tuple of dummy structs const":
      var engine = newEngine(runMetal)
      engine.ingest(kernelCode2)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("dummyKernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 1

when isMainModule:
  runTest()
