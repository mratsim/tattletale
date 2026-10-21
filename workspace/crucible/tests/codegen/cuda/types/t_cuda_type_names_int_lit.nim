## NVRTC: getTypeName handles nnkIntLit and nnkUIntLit nodes.
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/cuda/types/t_cuda_type_names_int_lit.nim
import std/strformat
import workspace/crucible

type
  Sized[N: static int] = object
    len: uint32

const kernelCode = cuda:
  proc intLitTypeNameKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let x = Sized[128]()
    output[0] = 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[1, uint32]
  var engine = bkCuda.newEngine()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel("intLitTypeNameKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  doAssert buf[0] == 1
  echo "  OK (test_cuda_type_names_int_lit)"

when isMainModule:
  runTest()
