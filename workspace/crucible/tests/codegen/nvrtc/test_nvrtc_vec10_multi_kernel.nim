## Multi-kernel Vec10 add/mul — NVRTC (CUDA) backend
## Run with:
##   nim c -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_vec10_multi_kernel.nim
import workspace/crucible

const kernelCode = cuda:
  proc vec10_add(
    output: ptr UncheckedArray[uint32];
    a: ptr UncheckedArray[uint32];
    b: ptr UncheckedArray[uint32]) {.global.} =
    let tid = blockIdx.x * blockDim.x + threadIdx.x
    if tid < 10:
      output[tid] = a[tid] + b[tid]
  proc vec10_mul(
    output: ptr UncheckedArray[uint32];
    a: ptr UncheckedArray[uint32];
    b: ptr UncheckedArray[uint32]) {.global.} =
    let tid = blockIdx.x * blockDim.x + threadIdx.x
    if tid < 10:
      output[tid] = a[tid] * b[tid]

var engine = bkCuda.init()

engine.ingest(kernelCode)

var a: array[10, uint32] = [1'u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]
var b: array[10, uint32] = [10'u32, 20, 30, 40, 50, 60, 70, 80, 90, 100]

var buf: array[10, uint32]
engine.run<<(1, 10)>>("vec10_add", buf, (a, b))
echo "  vec10_add: ", buf
for i in 0 ..< 10:
  doAssert buf[i] == a[i] + b[i]

engine.run<<(1, 10)>>("vec10_mul", buf, (a, b))
echo "  vec10_mul: ", buf
for i in 0 ..< 10:
  doAssert buf[i] == a[i] * b[i]

echo "  OK — Multi-kernel vec10 (CUDA)"
