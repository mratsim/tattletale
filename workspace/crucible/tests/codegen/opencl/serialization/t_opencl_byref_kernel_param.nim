## OpenCL: a by-ref device call can take a kernel parameter struct
## Test that a kernel calling a device function taking a struct by reference reads back exact values
##
## Kernel parameters are kernel arguments with no addressable storage.
## The call site materializes a local and passes that local's address.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/serialization/t_opencl_byref_kernel_param.nim

import workspace/crucible

type
  Count = object
    n: int32
    pad: array[16, int32]

const byrefKernelParamOcl = opencl:
  proc readCount(c: Count): int32 {.device.} =
    c.n + 1'i32

  proc kernelArgKernel(output: ptr UncheckedArray[int32], w: Count) {.global.} =
    output[0] = readCount(w)

proc runTest() =
  var engine = newEngine(runOpenCL)
  engine.ingest(byrefKernelParamOcl)
  let oDev = engine.allocDevice(int32, 1)

  var w: Count
  w.n = 41'i32
  engine.launchKernel("kernelArgKernel", oDev, (w,))
  var got: array[1, int32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 42'i32, "got " & $got[0] & ", want 41 + 1 = 42"
  echo "[OK] OpenCL: kernel parameter struct crosses to a by-ref call"

when isMainModule:
  runTest()
  echo "t_opencl_byref_kernel_param: all checks passed"
