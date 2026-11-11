## launchDeviceFn with a struct carrying ref DeviceMem fields on Vulkan.
##
## Test that a struct argument with ref DeviceMem fields crosses launchDeviceFn.
## The packer writes device addresses, never the ref cell's host pointer.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/app/t_vulkan_ref_device_mem.nim

import workspace/crucible

type
  AxpyView = object
    p: ptr UncheckedArray[float32]
    n: int

  AxpyRefs = object
    p: ref DeviceMem[runVulkan, float32]
    n: int

proc axpy(y: AxpyView, alpha: float32, x: AxpyView) =
  for i in 0 ..< y.n:
    y.p[i] = alpha * x.p[i] + y.p[i]

proc runTest() =
  var engine = newEngine(runVulkan)
  const N = 7
  let xRef = new(DeviceMem[runVulkan, float32])
  let yRef = new(DeviceMem[runVulkan, float32])
  xRef[] = engine.allocDevice(float32, N)
  yRef[] = engine.allocDevice(float32, N)
  var xHost: array[N, float32]
  var yHost: array[N, float32]
  for i in 0 ..< N:
    xHost[i] = float32(i + 1)
    yHost[i] = float32(10 * (i + 1))
  engine.copyToDeviceFromHost(xRef[], xHost.toOpenArray(0, N - 1))
  engine.copyToDeviceFromHost(yRef[], yHost.toOpenArray(0, N - 1))

  engine.launchDeviceFn axpy(AxpyRefs(p: yRef, n: N), 2.5'f32,
                             AxpyRefs(p: xRef, n: N)) do:
    lang = glsl
    grid = (1, 1)
    blk = (1, 1)

  var got: array[N, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, N - 1), yRef[])
  for i in 0 ..< N:
    let expected = 2.5'f32 * xHost[i] + yHost[i]
    doAssert got[i] == expected,
      "axpy: element " & $i & " got " & $got[i] & " want " & $expected
  echo "[OK] launchDeviceFn: struct ref DeviceMem fields cross to the kernel"

when isMainModule:
  runTest()
