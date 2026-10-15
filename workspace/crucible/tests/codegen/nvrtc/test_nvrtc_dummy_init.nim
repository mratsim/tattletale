## crucible CUDA codegen for structs with dummy/empty fields
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/test_nvrtc_dummy_init.nim
import std/[unittest, strformat]
import workspace/crucible

type
  FixMe*[V: static int] = object

const kernelCode = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

const kernelCode2 = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (FixMe[1](), FixMe[8]())
    C[0] = 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "CUDA - dummy-field initializers":
    test "single dummy struct const":
      var buf: array[1, uint32]
      var engine = bkCuda.init()
      engine.ingest(kernelCode)
      let oDev = engine.allocDevice(sizeof(buf))
      engine.run<<(1, 1)>>("kernel", oDev, ())
      engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
      engine.freeDevice(oDev)
      check buf[0] == 1

    test "tuple of dummy structs const":
      var buf: array[1, uint32]
      var engine = bkCuda.init()
      engine.ingest(kernelCode2)
      let oDev = engine.allocDevice(sizeof(buf))
      engine.run<<(1, 1)>>("kernel", oDev, ())
      engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
      engine.freeDevice(oDev)
      check buf[0] == 1

when isMainModule:
  runTest()
