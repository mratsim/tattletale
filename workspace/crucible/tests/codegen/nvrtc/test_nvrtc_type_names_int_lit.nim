## NVRTC: getTypeName with nnkIntLit/nnkUIntLit
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/nvrtc/test_nvrtc_type_names_int_lit.nim
##
## Coverage: nim_to_gpu.nim:332-335
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  Sized[N: static int] = object
    len: uint32

const kernelCode = cuda:
  proc intLitTypeNameKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let x = Sized[128]()
    output[0] = 1'u32

var buf: array[1, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("intLitTypeNameKernel", buf, ())
doAssert buf[0] == 1
echo "  OK"
