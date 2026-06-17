## CuTe: complex macro-generated code hitting except path (B25)
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/nvrtc/test_cute_macro_complex.nim
##
## Tests multiple generic instantiations with varying params
## to exercise the generic resolution machinery.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

const kernelCode = cuda:
  type
    Tensor2 = object
      data: array[2, uint32]
    Tensor3 = object
      data: array[3, uint32]

  proc complexKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let a = Tensor2(data: [1'u32, 2'u32])
    output[0] = a.data[0]
    output[1] = a.data[1]

    let b = Tensor3(data: [42'u32, 43'u32, 44'u32])
    output[2] = b.data[0]

var buf: array[3, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("complexKernel", buf, ())
doAssert buf[0] == 1,   &"a[0]: {buf[0]}"
doAssert buf[1] == 2,   &"a[1]: {buf[1]}"
doAssert buf[2] == 42,  &"b[0]: {buf[2]}"
echo "  OK — macro complex path (B25)"
