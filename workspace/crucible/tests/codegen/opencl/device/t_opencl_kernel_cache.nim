## OpenCL: kernel cache - same engine across multiple runs and re-ingest,
## a cached kernel reused across runs, re-ingesting new source invalidates it.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/device/t_opencl_kernel_cache.nim

import workspace/crucible

const kernelCode = opencl:
  proc addKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]

const kernelCode2 = opencl:
  proc addKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0] + 1000'u32

proc runTest() =   # private - tests run in a proc so engines are destroyed at return
  echo "=== OpenCL kernel cache ===\n"

  block: # same kernel, two runs with different data (cache hit on second run)
    var engine = bkOpenCL.init()
    engine.ingest(kernelCode)

    var a: array[1, uint32] = [10'u32]
    var b: array[1, uint32] = [1'u32]
    var outVal: array[1, uint32]

    let aDev = engine.allocDevice(a.len * sizeof(uint32))
    let bDev = engine.allocDevice(b.len * sizeof(uint32))
    let oDev = engine.allocDevice(sizeof(uint32))
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len * sizeof(uint32))
    engine.launchKernel("addKernel", oDev, (aDev, bDev))
    engine.copyToHostFromDevice(addr outVal[0], oDev, sizeof(uint32))
    doAssert outVal[0] == 11

    a = [100'u32]
    b = [200'u32]
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len * sizeof(uint32))
    engine.launchKernel("addKernel", oDev, (aDev, bDev))
    engine.copyToHostFromDevice(addr outVal[0], oDev, sizeof(uint32))
    engine.freeDevice(aDev)
    engine.freeDevice(bDev)
    engine.freeDevice(oDev)
    doAssert outVal[0] == 300
    echo "  OK - cached kernel reused across runs"

  block: # re-ingest invalidates the cache, new source takes effect
    var engine = bkOpenCL.init()
    engine.ingest(kernelCode)

    var a: array[1, uint32] = [1'u32]
    var b: array[1, uint32] = [2'u32]
    var outVal: array[1, uint32]

    let aDev = engine.allocDevice(a.len * sizeof(uint32))
    let bDev = engine.allocDevice(b.len * sizeof(uint32))
    let oDev = engine.allocDevice(sizeof(uint32))
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len * sizeof(uint32))
    engine.launchKernel("addKernel", oDev, (aDev, bDev))
    engine.copyToHostFromDevice(addr outVal[0], oDev, sizeof(uint32))
    doAssert outVal[0] == 3

    engine.ingest(kernelCode2)
    engine.launchKernel("addKernel", oDev, (aDev, bDev))
    engine.copyToHostFromDevice(addr outVal[0], oDev, sizeof(uint32))
    engine.freeDevice(aDev)
    engine.freeDevice(bDev)
    engine.freeDevice(oDev)
    doAssert outVal[0] == 1003
    echo "  OK - re-ingest invalidated cached kernel"

  echo "All cache tests passed"

when isMainModule:
  runTest()
