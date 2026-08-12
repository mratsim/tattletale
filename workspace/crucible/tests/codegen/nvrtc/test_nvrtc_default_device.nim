## NVRTC: -default-device NVRTC flag (global vars default to __device__)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_default_device.nim
##
## Coverage: nvrtc.nim:99
import std/strformat
import workspace/crucible

const kernelCode = cuda:
  proc defaultDeviceKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 42'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[1, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  engine.run("defaultDeviceKernel", buf, ())
  doAssert buf[0] == 42
  echo "  OK (test_nvrtc_default_device)"

when isMainModule:
  runTest()
