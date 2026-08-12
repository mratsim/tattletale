## NVRTC: cross-platform GPU builtins (min, max, abs)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_builtins.nim
##
## min, max, abs are available on all 4 backends (CUDA, OpenCL, GLSL, WGSL).
## Registered as known builtins in addProcToGenericInsts to avoid parsing
## Nim's system module body (which has if-expressions the codegen can't handle).
import std/strformat
import workspace/crucible

const kernelCode = cuda:
  proc builtinKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = min(10'u32, 20'u32)
    output[1] = max(30'u32, 7'u32)
    output[2] = abs(int32(-5)).uint32

var buf: array[3, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
engine.run<<(1, 1)>>("builtinKernel", buf, ())
doAssert buf[0] == 10, &"min: {buf[0]}"
doAssert buf[1] == 30, &"max: {buf[1]}"
doAssert buf[2] == 5, &"abs: {buf[2]}"
echo "  OK — builtins"
