## launchDeviceFn with a View argument on webgpu: pointer + length in one object.
##
## Test that a View argument holding DeviceMem handles crosses launchDeviceFn
## into the kernel's pointer fields.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/app/t_webgpu_axpy_view.nim

import workspace/crucible

type
  AxpyView = object
    p: ptr UncheckedArray[float32]
    n: int

  AxpyBuf[DevMem] = object
    p: DevMem
    n: int

proc axpy(y: AxpyView, alpha: float32, x: AxpyView) =
  for i in 0 ..< y.n:
    y.p[i] = alpha * x.p[i] + y.p[i]

proc runTest() =
  var engine = newEngine(runWebGPU)
  const N = 7
  let xDev = engine.allocDevice(float32, N)
  let yDev = engine.allocDevice(float32, N)
  var xHost: array[N, float32]
  var yHost: array[N, float32]
  for i in 0 ..< N:
    xHost[i] = float32(i + 1)
    yHost[i] = float32(10 * (i + 1))
  engine.copyToDeviceFromHost(xDev, addr xHost[0], N)
  engine.copyToDeviceFromHost(yDev, addr yHost[0], N)

  engine.launchDeviceFn axpy(AxpyBuf[typeof(yDev)](p: yDev, n: N), 2.5'f32,
                             AxpyBuf[typeof(xDev)](p: xDev, n: N)) do:
    grid = (1, 1)
    blk = (1, 1)

  var got: array[N, float32]
  engine.copyToHostFromDevice(addr got[0], yDev, N)
  for i in 0 ..< N:
    let expected = 2.5'f32 * xHost[i] + yHost[i]
    doAssert got[i] == expected,
      "axpy: element " & $i & " got " & $got[i] & " want " & $expected
  echo "[OK] launchDeviceFn: View argument with pointer + length crosses to the kernel"

runTest()
