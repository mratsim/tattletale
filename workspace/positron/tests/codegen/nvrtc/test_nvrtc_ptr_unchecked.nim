## NVRTC: ptr array + device func chaining via codegen pipeline
## Run with: nim cpp -d:cuda -r workspace/positron/tests/nvrtc/test_nvrtc_ptr_unchecked.nim
##   Note: `cuda:` macro always generates CUDA now; `-d:cuda` only needed for NVRTC runtime
import workspace/positron/src/codegen/nvrtc

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

var nv = initNvrtc(kernelCode)
nv.numBlocks = 1
nv.threadsPerBlock = 1
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"

var outBuf: array[8, uint32]
nv.execute("fillKernel", outBuf, ())
echo "  fill: ", outBuf
for i in 0..7: doAssert outBuf[i] == uint32(i + 10)

var a, b: array[8, uint32]
for i in 0..7: a[i] = uint32(i); b[i] = uint32(i * 10)
var c: array[8, uint32]
nv.execute("addKernel", c, (a, b))
echo "  add:  ", c
for i in 0..7: doAssert c[i] == uint32(i + i*10)
echo "  OK"
