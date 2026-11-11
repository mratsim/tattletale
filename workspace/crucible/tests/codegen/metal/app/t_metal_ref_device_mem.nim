## launchDeviceFn with a ref DeviceMem argument on Metal.
##
## Test that a ref DeviceMem crosses launchDeviceFn into the kernel's
## pointer parameter and that the batch entry keeps the buffer alive.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/app/t_metal_ref_device_mem.nim

import workspace/crucible

proc saxpy(y: ptr UncheckedArray[float32], alpha: float32, x: ptr UncheckedArray[float32]) =
  for i in 0 ..< 7:
    y[i] = alpha * x[i] + y[i]

proc runTest() =
  var engine = newEngine(runMetal)
  const N = 7
  let xRef = new(DeviceMem[runMetal, float32])
  let yRef = new(DeviceMem[runMetal, float32])
  xRef[] = engine.allocDevice(float32, N)
  yRef[] = engine.allocDevice(float32, N)
  var xHost: array[N, float32]
  var yHost: array[N, float32]
  for i in 0 ..< N:
    xHost[i] = float32(i + 1)
    yHost[i] = float32(10 * (i + 1))
  engine.copyToDeviceFromHost(xRef[], xHost.toOpenArray(0, N - 1))
  engine.copyToDeviceFromHost(yRef[], yHost.toOpenArray(0, N - 1))

  engine.launchDeviceFn saxpy(yRef, 2.5'f32, xRef) do:
    grid = (1, 1)
    blk = (1, 1)

  var got: array[N, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, N - 1), yRef[])
  for i in 0 ..< N:
    let expected = 2.5'f32 * xHost[i] + yHost[i]
    doAssert got[i] == expected,
      "axpy: element " & $i & " got " & $got[i] & " want " & $expected
  echo "[OK] launchDeviceFn: ref DeviceMem argument crosses to the kernel"

runTest()
