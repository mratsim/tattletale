## NVRTC: thread/block identifiers via codegen pipeline
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_thread_id.nim
##   Note: `cuda:` macro always generates CUDA now; `-d:cuda` only needed for NVRTC runtime
import workspace/crucible/src/codegen/nvrtc

const kernelCode = cuda:
  proc threadIdKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let tid = blockIdx.x * blockDim.x + threadIdx.x
    output[0] = uint32(tid)
    output[1] = uint32(blockIdx.x)
    output[2] = uint32(threadIdx.x)
    output[3] = uint32(blockDim.x)

var nv = initNvrtc(kernelCode)
nv.numBlocks = 1
nv.threadsPerBlock = 1
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
var buf: array[4, uint32]
nv.execute("threadIdKernel", buf, ())
echo "  tid=", buf[0], " bid=", buf[1], " tix=", buf[2], " bdx=", buf[3]
doAssert buf[0] == 0
doAssert buf[1] == 0
doAssert buf[2] == 0
doAssert buf[3] == 1
echo "  OK"
