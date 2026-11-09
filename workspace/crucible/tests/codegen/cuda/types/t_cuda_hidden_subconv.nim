## NVRTC: hidden subtype conversions (nnkHiddenSubConv).
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/cuda/types/t_cuda_hidden_subconv.nim
import std/strformat
import workspace/crucible

const kernelCode = cuda:
  proc hiddenSubConvKernel(output: ptr UncheckedArray[int64]) {.global.} =
    let a: uint32 = 42
    output[0] = int64(a)   # may produce HiddenSubConv

proc runTest() =
  var buf: array[1, int64]
  var engine = newEngine(runCuda)
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(int64, buf.len)
  engine.launchKernel("hiddenSubConvKernel", oDev, ())
  engine.copyToHostFromDevice(buf, oDev)
  doAssert buf[0] == 42, &"hiddenSubConv: got {buf[0]}"
  echo "  OK (test_cuda_hidden_subconv)"

when isMainModule:
  runTest()
