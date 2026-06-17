## NVRTC: var T (byref) param via codegen pipeline
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_var_param.nim
##   Note: `cuda:` macro always generates CUDA now; `-d:cuda` only needed for NVRTC runtime
import workspace/crucible/src/codegen/nvrtc

const kernelCode = cuda:
  type Pair = object
    x: uint32
    y: uint32
  proc setPair(p: var Pair; vx, vy: uint32) {.device.} =
    p.x = vx
    p.y = vy
  proc swap(a, b: var uint32) {.device.} =
    let tmp = a
    a = b
    b = tmp
  proc varParamKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var p: Pair
    setPair(p, 10'u32, 20'u32)
    output[0] = p.x
    output[1] = p.y
    var a, b: uint32 = 1
    b = 2
    swap(a, b)
    output[2] = a
    output[3] = b

var buf: array[4, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("varParamKernel", buf, ())
echo "  p.x=", buf[0], " p.y=", buf[1], " a=", buf[2], " b=", buf[3]
doAssert buf[0] == 10
doAssert buf[1] == 20
doAssert buf[2] == 2
doAssert buf[3] == 1
echo "  OK"
