## OpenCL: structs with no fields (padded with `char _`) must emit a valid
## object-constructor initializer, not empty braces.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/device/t_opencl_dummy_init.nim

import std/[unittest, strformat]
import workspace/crucible

type
  FixMe*[V: static int] = object

const kernelCode = opencl:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

const kernelCode2 = opencl:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (FixMe[1](), FixMe[8]())
    C[0] = 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "OpenCL - dummy-field initializers":
    test "single dummy struct const":
      var engine = bkOpenCL.init()
      engine.ingest(kernelCode)
      echo "===="
      echo kernelCode
      echo "===="
      var buf: array[1, uint32]
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("dummyKernel", oDev, ())
      engine.copyToHostFromDevice(addr buf[0], oDev, 1)
      engine.freeDevice(oDev)
      check buf[0] == 1

    test "tuple of dummy structs const":
      var engine = bkOpenCL.init()
      engine.ingest(kernelCode2)
      echo "===="
      echo kernelCode2
      echo "===="
      var buf: array[1, uint32]
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("dummyKernel", oDev, ())
      engine.copyToHostFromDevice(addr buf[0], oDev, 1)
      engine.freeDevice(oDev)
      check buf[0] == 1

when isMainModule:
  runTest()
