## Multi-kernel Vec10 add/mul, NVRTC (CUDA) backend.
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/device/t_nvrtc_vec10_multi_kernel.nim
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

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var engine = bkCuda.init()

  engine.ingest(kernelCode)

  var a: array[10, uint32] = [1'u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  var b: array[10, uint32] = [10'u32, 20, 30, 40, 50, 60, 70, 80, 90, 100]

  var buf: array[10, uint32]
  let aDev = engine.allocDevice(sizeof(a))
  engine.copyToDeviceFromHost(aDev, addr a[0], sizeof(a))
  let bDev = engine.allocDevice(sizeof(b))
  engine.copyToDeviceFromHost(bDev, addr b[0], sizeof(b))
  let oDev = engine.allocDevice(sizeof(buf))
  engine.launchKernel<<(1, 10)>>("vec10_add", oDev, (aDev, bDev))
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  echo "  vec10_add: ", buf
  for i in 0 ..< 10:
    doAssert buf[i] == a[i] + b[i]

  engine.launchKernel<<(1, 10)>>("vec10_mul", oDev, (aDev, bDev))
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(aDev)
  engine.freeDevice(bDev)
  engine.freeDevice(oDev)
  echo "  vec10_mul: ", buf
  for i in 0 ..< 10:
    doAssert buf[i] == a[i] * b[i]

  echo "  OK — Multi-kernel vec10 (CUDA)"

when isMainModule:
  runTest()
