## NVRTC: bool/uint32 kernel args via codegen pipeline
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_bool_arg.nim
##   Note: `cuda:` macro always generates CUDA now; `-d:cuda` only needed for NVRTC runtime
import workspace/crucible

# NOTE: res gets prepended → output MUST be first kernel param
const kernelCode = cuda:
  proc condAdd(c, a, b: ptr UncheckedArray[uint32]; useAdd: bool) {.global.} =
    let tid = blockIdx.x * blockDim.x + threadIdx.x
    if tid < 8:
      if useAdd:
        c[tid] = a[tid] + b[tid]
      else:
        c[tid] = a[tid] - b[tid]
  proc scale(output, input: ptr UncheckedArray[uint32]; factor: int32; subtract: bool) {.global.} =
    let tid = blockIdx.x * blockDim.x + threadIdx.x
    if tid < 8:
      if subtract:
        output[tid] = uint32(int32(input[tid]) * factor - 1)
      else:
        output[tid] = uint32(int32(input[tid]) * factor)

var engine = bkCuda.init()

engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"

var a, b, r: array[8, uint32]
for i in 0..7: a[i] = uint32(i + 1); b[i] = uint32((i + 1) * 2)

# res=r → kernel gets (c=r, a, b, useAdd=true)
engine.run("condAdd", r, (a, b, true))
echo "  condAdd(true): ", r
for i in 0..7: doAssert r[i] == a[i] + b[i]

engine.run("condAdd", r, (a, b, false))
echo "  condAdd(false): ", r
for i in 0..7: doAssert r[i] == a[i] - b[i]

engine.run("scale", r, (a, 3'i32, false))
echo "  scale(3, false): ", r
for i in 0..7: doAssert r[i] == uint32(int32(a[i]) * 3)

engine.run("scale", r, (a, 3'i32, true))
echo "  scale(3, true): ", r
for i in 0..7: doAssert r[i] == uint32(int32(a[i]) * 3 - 1)

echo "  OK (test_nvrtc_bool_arg)"
