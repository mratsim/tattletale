## OpenCL: empty-object construction emits `{0}`, never `{}`, and the kernel
## building such an object compiles and runs on the device.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/types/t_opencl_empty_object_init.nim

import std/strutils
import workspace/crucible

type
  Stateless = object
    discard

const kernelCode = opencl:
  proc makeStateless(): Stateless {.device.} =
    result = Stateless()
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    var s = makeStateless()
    output[0] = 42'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernelCode

  block:
    var engine = newEngine(runOpenCL)
    engine.ingest(kernelCode)
    var outVal: array[1, uint32]
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("kernelMain", oDev, ())
    engine.copyToHostFromDevice(addr outVal[0], oDev, 1)
    doAssert outVal[0] == 42
    echo "  OK"

when isMainModule:
  runTest()
