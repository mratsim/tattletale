## NVRTC: empty struct gpuTypeDef (char _; fallback)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_empty_struct_gpu.nim
##
## Coverage: cuda_lang.nim:410-412
import std/strformat
import workspace/crucible

type
  EmptyObj* = object
    ## No fields — exercises the char _; fallback

const kernelCode = cuda:
  proc emptyStructKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 42'u32

var buf: array[1, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"
engine.run("emptyStructKernel", buf, ())
doAssert buf[0] == 42
echo "  OK (test_nvrtc_empty_struct_gpu)"
