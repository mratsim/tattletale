## NVRTC: getTypeName with nnkIntLit/nnkUIntLit
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_type_names_int_lit.nim
##
## Coverage: nim_to_gpu.nim:332-335
import std/strformat
import workspace/crucible

type
  Sized[N: static int] = object
    len: uint32

const kernelCode = cuda:
  proc intLitTypeNameKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let x = Sized[128]()
    output[0] = 1'u32

var buf: array[1, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"
engine.run("intLitTypeNameKernel", buf, ())
doAssert buf[0] == 1
echo "  OK (test_nvrtc_type_names_int_lit)"
