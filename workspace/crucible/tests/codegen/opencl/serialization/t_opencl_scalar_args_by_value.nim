## OpenCL: scalar kernel arguments (alpha/beta) bind by value next to a buffer argument, not as cl_mem handles.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/serialization/t_opencl_scalar_args_by_value.nim

import workspace/crucible

const kernelCode = opencl:
  proc scaleKernel(output: ptr UncheckedArray[float32];
                   a: ptr UncheckedArray[float32];
                   alpha: float32;
                   beta: int32) {.global.} =
    output[0] = a[0] * alpha + float32(beta)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernelCode

  block:
    var engine = bkOpenCL.init()
    engine.ingest(kernelCode)
    var a: array[1, float32] = [2.5'f32]
    var alpha = 3.0'f32
    var beta = 4'i32
    var output: array[1, float32]
    let aDev = engine.allocDevice(float32, a.len)
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
    let oDev = engine.allocDevice(float32, 1)
    engine.launchKernel("scaleKernel", oDev, (aDev, alpha, beta))
    engine.copyToHostFromDevice(addr output[0], oDev, 1)
    engine.freeDevice(aDev)
    engine.freeDevice(oDev)
    doAssert output[0] == 11.5'f32, "expected 2.5 * 3.0 + 4.0 = 11.5"
    echo "  OK"

when isMainModule:
  runTest()
