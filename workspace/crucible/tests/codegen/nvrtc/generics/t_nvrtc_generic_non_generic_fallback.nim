## NVRTC: generic to non-generic fallback in initGpuGenericInst for a generic
## object with a static-int-sized array.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/nvrtc/generics/t_nvrtc_generic_non_generic_fallback.nim
import std/strformat
import workspace/crucible

type
  Sized[N: static int] = object
    data: array[N, uint32]

const kernelCode = cuda:
  proc sizedKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let x = Sized[4](data: [1'u32, 2'u32, 3'u32, 4'u32])
    output[0] = x.data[0]
    output[1] = x.data[3]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[2, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"

  let oDev = engine.allocDevice(sizeof(buf))
  engine.launchKernel("sizedKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(oDev)
  doAssert buf[0] == 1, &"Sized[4].data[0]: got {buf[0]}, expected 1"
  doAssert buf[1] == 4, &"Sized[4].data[3]: got {buf[1]}, expected 4"

  echo "  OK (test_nvrtc_generic_non_generic_fallback)"

when isMainModule:
  runTest()
