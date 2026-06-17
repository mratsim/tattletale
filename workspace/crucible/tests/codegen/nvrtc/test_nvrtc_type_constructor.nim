## NVRTC: nnkCall in nnkTypeDef (constructor type definitions)
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/nvrtc/test_nvrtc_type_constructor.nim
##
## Coverage: nim_to_gpu.nim:1356-1358
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  Point = object
    x, y: uint32

const kernelCode = cuda:
  proc typeConstructorKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let p = Point(x: 10'u32, y: 20'u32)
    output[0] = p.x
    output[1] = p.y

var buf: array[2, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("typeConstructorKernel", buf, ())
doAssert buf[0] == 10
doAssert buf[1] == 20
echo "  OK"
