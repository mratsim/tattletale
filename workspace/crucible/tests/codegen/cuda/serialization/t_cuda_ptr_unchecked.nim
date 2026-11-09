## NVRTC: `ptr UncheckedArray` kernels chained through device procs.
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/cuda/serialization/t_cuda_ptr_unchecked.nim
import workspace/crucible

const kernelCode = cuda:
  proc fillArray(p: ptr UncheckedArray[uint32]; n: uint32) {.device.} =
    for i in 0 ..< n:
      p[i] = i + 10'u32
  proc addArrays(c, a, b: ptr UncheckedArray[uint32]; n: uint32) {.device.} =
    for i in 0 ..< n:
      c[i] = a[i] + b[i]
  proc fillKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    fillArray(output, 8)
  proc addKernel(c, a, b: ptr UncheckedArray[uint32]) {.global.} =
    addArrays(c, a, b, 8)

proc runTest() =
  var engine = newEngine(runCuda)

  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"

  var outBuf: array[8, uint32]
  let oDev = engine.allocDevice(uint32, outBuf.len)
  engine.launchKernel<<(1, 1)>>("fillKernel", oDev, ())
  engine.copyToHostFromDevice(outBuf, oDev)
  echo "  fill: ", outBuf
  for i in 0..7: doAssert outBuf[i] == uint32(i + 10)

  var a, b: array[8, uint32]
  for i in 0..7: a[i] = uint32(i); b[i] = uint32(i * 10)
  var c: array[8, uint32]
  let aDev = engine.allocDevice(uint32, a.len)
  engine.copyToDeviceFromHost(aDev, a)
  let bDev = engine.allocDevice(uint32, b.len)
  engine.copyToDeviceFromHost(bDev, b)
  let cDev = engine.allocDevice(uint32, c.len)
  engine.launchKernel<<(1, 1)>>("addKernel", cDev, (aDev, bDev))
  engine.copyToHostFromDevice(c, cDev)
  echo "  add:  ", c
  for i in 0..7: doAssert c[i] == uint32(i + i*10)
  echo "  OK (test_cuda_ptr_unchecked)"

when isMainModule:
  runTest()
