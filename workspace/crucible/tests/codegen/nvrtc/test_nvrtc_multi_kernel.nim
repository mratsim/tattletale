## NVRTC: multiple kernels via codegen pipeline
## Run with:
##   nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_multi_kernel.nim
##   Note: `cuda:` macro always generates CUDA now; `-d:cuda` only needed for NVRTC runtime
import workspace/crucible

const kernelCode = cuda:
  proc initArray(p: ptr UncheckedArray[uint32]; val: uint32; n: uint32) {.device.} =
    for i in 0 ..< n:
      p[i] = val
  proc setConstantKernel(output: ptr UncheckedArray[uint32]; val: uint32) {.global.} =
    initArray(output, val, 8)
  proc addKernel(c, a, b: ptr UncheckedArray[uint32]) {.global.} =
    let tid = blockIdx.x * blockDim.x + threadIdx.x
    if tid < 8:
      c[tid] = a[tid] + b[tid]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var engine = bkCuda.init()

  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"

  var buf: array[8, uint32]
  engine.run<<(1, 8)>>("setConstantKernel", buf, (42'u32))
  echo "  setConstant(42): ", buf
  for i in 0..7: doAssert buf[i] == 42

  var a, b: array[8, uint32]
  for i in 0..7: a[i] = uint32(i); b[i] = uint32(i * 10)
  engine.run<<(1, 8)>>("addKernel", buf, (a, b))
  echo "  add: ", buf
  for i in 0..7: doAssert buf[i] == uint32(i + i*10)
  echo "  OK (test_nvrtc_multi_kernel)"

when isMainModule:
  runTest()
