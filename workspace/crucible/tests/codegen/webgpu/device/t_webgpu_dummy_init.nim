## WebGPU: WGSL codegen for structs with dummy/empty fields, the object constructor emits a valid initializer.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/device/t_webgpu_dummy_init.nim

import std/[unittest, strformat]
import workspace/crucible

type
  FixMe*[V: static int] = object

const kernelCode = webgpu:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

const kernelCode2 = webgpu:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (FixMe[1](), FixMe[8]())
    C[0] = 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "WebGPU - dummy-field initializers":
    test "single dummy struct const":
      var engine = bkWGSL.init()
      engine.ingest(kernelCode)
      echo kernelCode
      var res: array[1, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel("dummyKernel", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      check res[0] == 1


    test "tuple of dummy structs const":
      var engine = bkWGSL.init()
      engine.ingest(kernelCode2)
      echo kernelCode2
      var res: array[1, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel("dummyKernel", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      check res[0] == 1

when isMainModule:
  runTest()
