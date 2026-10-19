## NVRTC: ntyStatic branch in nimToGpuType.
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/types/t_nvrtc_type_proc_and_static.nim
import std/strformat
import workspace/crucible

type
  ## ntyStatic: array sized by static int
  Buffer[N: static int] = object
    data: array[N, uint32]

const kernelCode = cuda:
  proc typeBranchesKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # ntyStatic: Buffer[N] with N as a static parameter
    let b = Buffer[4](data: [1'u32, 2'u32, 3'u32, 4'u32])
    output[0] = b.data[0]
    output[1] = b.data[3]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[2, uint32]
  var engine = bkCuda.newEngine()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel("typeBranchesKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  doAssert buf[0] == 1, &"ntyStatic buf[0]: got {buf[0]}"
  doAssert buf[1] == 4, &"ntyStatic buf[1]: got {buf[1]}"
  echo "  OK (test_nvrtc_type_proc_and_static)"

when isMainModule:
  runTest()
