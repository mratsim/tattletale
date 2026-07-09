## NVRTC: -default-device NVRTC flag (global vars default to __device__)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_default_device.nim
##
## Coverage: nvrtc.nim:99
import std/strformat
import workspace/crucible/src/codegen/nvrtc

const kernelCode = cuda:
  proc defaultDeviceKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 42'u32

var buf: array[1, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("defaultDeviceKernel", buf, ())
doAssert buf[0] == 42
echo "  OK (test_nvrtc_default_device)"
