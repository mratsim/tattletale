## NVRTC: hidden subtype conversions (nnkHiddenSubConv)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_hidden_subcoengine.nim
##
## Coverage: nim_to_gpu.nim:1454-1456
import std/strformat
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

const kernelCode = cuda:
  proc hiddenSubConvKernel(output: ptr UncheckedArray[int64]) {.global.} =
    let a: uint32 = 42
    output[0] = int64(a)   # may produce HiddenSubConv

var buf: array[1, int64]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"
engine.run("hiddenSubConvKernel", buf, ())
doAssert buf[0] == 42, &"hiddenSubConv: got {buf[0]}"
echo "  OK (test_nvrtc_hidden_subconv)"
