## NVRTC: ptr array + device func chaining via codegen pipeline
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_ptr_unchecked.nim
##   Note: `cuda:` macro always generates CUDA now; `-d:cuda` only needed for NVRTC runtime
import workspace/crucible

const kernelCode = cuda:
  proc fillArray(p: ptr UncheckedArray[uint32]; n: uint32) {.device.} =
    for i in 0 ..< n:
      p[i] = i + 10'u32
  proc addArrays(c, a, b: ptr UncheckedArray[uint32]; n: uint32) {.device.} =
    for i in 0 ..< n:
      c[i] = a[i] + b[i]
  proc fillKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    fillArray(output, 8)
  proc addKernel(c, a, b: ptr UncheckedArray[uint32]) {.global.} =
    addArrays(c, a, b, 8)

var engine = bkCuda.init()

engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"

var outBuf: array[8, uint32]
engine.run<<(1, 1)>>("fillKernel", outBuf, ())
echo "  fill: ", outBuf
for i in 0..7: doAssert outBuf[i] == uint32(i + 10)

var a, b: array[8, uint32]
for i in 0..7: a[i] = uint32(i); b[i] = uint32(i * 10)
var c: array[8, uint32]
engine.run<<(1, 1)>>("addKernel", c, (a, b))
echo "  add:  ", c
for i in 0..7: doAssert c[i] == uint32(i + i*10)
echo "  OK (test_nvrtc_ptr_unchecked)"
