## Test: OpenCL kernel cache - same engine across multiple runs and re-ingest
##
## The engine compiles each kernel once per ingest and reuses the cached
## kernel across runs; re-ingesting new source must invalidate the cache so
## a later run uses the freshly ingested source.
##
## To compile and run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_kernel_cache.nim

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

    engine.run("addKernel", outVal, (a, b))
    doAssert outVal[0] == 11

    a = [100'u32]
    b = [200'u32]
    engine.run("addKernel", outVal, (a, b))
    doAssert outVal[0] == 300
    echo "  OK - cached kernel reused across runs"

  block: # re-ingest invalidates the cache, new source takes effect
    var engine = bkOpenCL.init()
    engine.ingest(kernelCode)

    var a: array[1, uint32] = [1'u32]
    var b: array[1, uint32] = [2'u32]
    var outVal: array[1, uint32]

    engine.run("addKernel", outVal, (a, b))
    doAssert outVal[0] == 3

    engine.ingest(kernelCode2)
    engine.run("addKernel", outVal, (a, b))
    doAssert outVal[0] == 1003
    echo "  OK - re-ingest invalidated cached kernel"

  echo "All cache tests passed"

when isMainModule:
  runTest()
