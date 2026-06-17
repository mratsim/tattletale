## NVRTC: built-in functions test — verifies the gpuVoid path doesn't trigger doAssert
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_builtins.nim
import std/strformat
import workspace/crucible/src/codegen/nvrtc

const kernelCode = cuda:
  proc builtinKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = uint32(int32(5) * int32(3))  # 15
    output[1] = uint32(int32(10) / int32(2)) # 5

var buf: array[2, uint32]
var nv = initNvrtc(kernelCode)
nv.numBlocks = 1
nv.threadsPerBlock = 1
nv.compile()
nv.getPtx()
nv.execute("builtinKernel", buf, ())
doAssert buf[0] == 15, &"mul: {buf[0]}"
doAssert buf[1] == 5, &"div: {buf[1]}"
echo "  OK — builtins"
