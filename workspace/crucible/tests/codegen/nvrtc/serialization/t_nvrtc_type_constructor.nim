## NVRTC: nnkCall inside nnkTypeDef (constructor type definitions).
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/serialization/t_nvrtc_type_constructor.nim
import std/strformat
import workspace/crucible

type
  Point = object
    x, y: uint32

const kernelCode = cuda:
  proc typeConstructorKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let p = Point(x: 10'u32, y: 20'u32)
    output[0] = p.x
    output[1] = p.y

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[2, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(sizeof(buf))
  engine.launchKernel("typeConstructorKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(oDev)
  doAssert buf[0] == 10
  doAssert buf[1] == 20
  echo "  OK (test_nvrtc_type_constructor)"

when isMainModule:
  runTest()
