## OpenCL: in-place kernels read the output buffer's pre-launch contents,
## an in-place kernel run twice on the same buffer accumulates across runs.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/test_opencl_output_preinit.nim

import workspace/crucible

const kernelCode = opencl:
  proc incInPlace(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = output[0] + 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernelCode

  block:
    var engine = bkOpenCL.init()
    engine.ingest(kernelCode)

    # Seed the output buffer on the host, then run in place twice:
    # the device buffer keeps the previous run's contents.
    var outBuf: array[1, uint32]
    outBuf[0] = 0
    let oDev = engine.allocDevice(sizeof(uint32))
    engine.copyToDeviceFromHost(oDev, addr outBuf[0], sizeof(uint32))
    engine.run("incInPlace", oDev, ())
    engine.copyToHostFromDevice(addr outBuf[0], oDev, sizeof(uint32))
    doAssert outBuf[0] == 1, "first run must see a zeroed output buffer"

    # Second run: the buffer holds 1 and the in-place kernel reads 1 and writes 2.
    engine.run("incInPlace", oDev, ())
    engine.copyToHostFromDevice(addr outBuf[0], oDev, sizeof(uint32))
    engine.freeDevice(oDev)
    doAssert outBuf[0] == 2, "second run must see the previous output contents"
    echo "  OK"

when isMainModule:
  runTest()
