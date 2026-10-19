## NVRTC: hidden subtype conversions (nnkHiddenSubConv).
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/types/t_nvrtc_hidden_subconv.nim
import std/strformat
import workspace/crucible

const kernelCode = cuda:
  proc hiddenSubConvKernel(output: ptr UncheckedArray[int64]) {.global.} =
    let a: uint32 = 42
    output[0] = int64(a)   # may produce HiddenSubConv

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[1, int64]
  var engine = bkCuda.newEngine()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(int64, buf.len)
  engine.launchKernel("hiddenSubConvKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  doAssert buf[0] == 42, &"hiddenSubConv: got {buf[0]}"
  echo "  OK (test_nvrtc_hidden_subconv)"

when isMainModule:
  runTest()
