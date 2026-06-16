## NVRTC: empty struct gpuTypeDef (char _; fallback)
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/nvrtc/test_nvrtc_empty_struct_gpu.nim
##
## Coverage: cuda_lang.nim:410-412
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  EmptyObj* = object
    ## No fields — exercises the char _; fallback

const kernelCode = cuda:
  proc emptyStructKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 42'u32

var buf: array[1, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("emptyStructKernel", buf, ())
doAssert buf[0] == 42
echo "  OK"
