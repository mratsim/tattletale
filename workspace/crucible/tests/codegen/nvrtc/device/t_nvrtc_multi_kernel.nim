## NVRTC: several kernels in one module, a device proc called from a global kernel.
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/device/t_nvrtc_multi_kernel.nim
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
  var engine = bkCuda.newEngine()

  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"

  var buf: array[8, uint32]
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel<<(1, 8)>>("setConstantKernel", oDev, (42'u32,))
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  echo "  setConstant(42): ", buf
  for i in 0..7: doAssert buf[i] == 42

  var a, b: array[8, uint32]
  for i in 0..7: a[i] = uint32(i); b[i] = uint32(i * 10)
  let aDev = engine.allocDevice(uint32, a.len)
  engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
  let bDev = engine.allocDevice(uint32, b.len)
  engine.copyToDeviceFromHost(bDev, addr b[0], b.len)
  engine.launchKernel<<(1, 8)>>("addKernel", oDev, (aDev, bDev))
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  echo "  add: ", buf
  for i in 0..7: doAssert buf[i] == uint32(i + i*10)
  echo "  OK (test_nvrtc_multi_kernel)"

when isMainModule:
  runTest()
