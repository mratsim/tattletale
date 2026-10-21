## CuTe: complex macro-generated code hitting except path (B25)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/cuda/app/cute/t_cuda_cute_macro_complex.nim
import std/strformat
import workspace/crucible

const kernelCode = cuda:
  type
    Tensor2 = object
      data: array[2, uint32]
    Tensor3 = object
      data: array[3, uint32]

  proc complexKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let a = Tensor2(data: [1'u32, 2'u32])
    output[0] = a.data[0]
    output[1] = a.data[1]

    let b = Tensor3(data: [42'u32, 43'u32, 44'u32])
    output[2] = b.data[0]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[3, uint32]
  var engine = bkCuda.newEngine()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel("complexKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  doAssert buf[0] == 1,   &"a[0]: {buf[0]}"
  doAssert buf[1] == 2,   &"a[1]: {buf[1]}"
  doAssert buf[2] == 42,  &"b[0]: {buf[2]}"
  echo "  OK — macro complex path (B25)"

when isMainModule:
  runTest()
