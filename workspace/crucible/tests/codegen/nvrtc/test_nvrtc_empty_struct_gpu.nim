## NVRTC: empty struct gpuTypeDef (char _ fallback)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/test_nvrtc_empty_struct_gpu.nim
import std/strformat
import workspace/crucible

type
  EmptyObj* = object
    ## No fields — exercises the char _; fallback

const kernelCode = cuda:
  proc emptyStructKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 42'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[1, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(sizeof(buf))
  engine.launchKernel("emptyStructKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(oDev)
  doAssert buf[0] == 42
  echo "  OK (test_nvrtc_empty_struct_gpu)"

when isMainModule:
  runTest()
