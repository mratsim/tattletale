## NVRTC: var pragma handlers (collectAttributes - inject/gensym)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_var_pragmas.nim
##
## Coverage: nim_to_gpu.nim:611-612
import std/strformat
import workspace/crucible/src/codegen/nvrtc

const kernelCode = cuda:
  proc varPragmaKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 42'u32

var buf: array[1, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("varPragmaKernel", buf, ())
doAssert buf[0] == 42
echo "  OK"
