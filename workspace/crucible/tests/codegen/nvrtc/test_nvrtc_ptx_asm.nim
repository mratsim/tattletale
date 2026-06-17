## NVRTC: PTX inline asm via codegen pipeline
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_ptx_asm.nim
##   Note: `cuda:` macro always generates CUDA now; `-d:cuda` only needed for NVRTC runtime
import workspace/crucible/src/codegen/nvrtc

const kernelCode = cuda:
  proc add_co(a, b: uint32): uint32 {.device, forceinline.} =
    var res: uint32
    asm "\"add.cc.u32 %0, %1, %2;\" : \"=r\"(res) : \"r\"(a), \"r\"(b)"
    return res
  proc add_ci(a, b: uint32): uint32 {.device, forceinline.} =
    var res: uint32
    asm "\"addc.u32 %0, %1, %2;\" : \"=r\"(res) : \"r\"(a), \"r\"(b)"
    return res
  proc add_cio(a, b: uint32): uint32 {.device, forceinline.} =
    var res: uint32
    asm "\"addc.cc.u32 %0, %1, %2;\" : \"=r\"(res) : \"r\"(a), \"r\"(b)"
    return res
  proc ptxAsmKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = add_co(3'u32, 5'u32)
    output[1] = add_ci(10'u32, 20'u32)

var buf: array[2, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("ptxAsmKernel", buf, ())
echo "  [0]=", buf[0], " [1]=", buf[1]
doAssert buf[0] == 8
doAssert buf[1] == 30
echo "  OK"
