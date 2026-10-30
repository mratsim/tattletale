## OpenCL: an object constructor that omits a field must zero-initialize it,
## emitting `{0}` for array/object fields and `0` for scalars, never `{}`.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/types/t_opencl_missing_field_init.nim

import std/strutils
import workspace/crucible

type
  WorkItem = object
    value: uint32
    data: array[4, uint32]
    ptrField: ptr UncheckedArray[uint32]

const kernelCode = opencl:
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    var w = WorkItem(value: 7'u32)
    output[0] = w.value
    output[1] = w.data[0]
    output[2] = w.data[3]
    if cast[uint64](w.ptrField) == 0'u64:
      output[3] = 1'u32
    else:
      output[3] = 0'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernelCode

  # Omitted fields must zero-initialize: braced zeros for the array field,
  # a plain 0 for the pointer field, both checked at runtime below.
  # An empty initializer `{}` is invalid OpenCL C.
  doAssert not kernelCode.contains("){}"), "empty initializer list {} is invalid OpenCL C"

  block:
    var engine = newEngine(runOpenCL)
    engine.ingest(kernelCode)
    var out32: array[4, uint32]
    let oDev = engine.allocDevice(uint32, 4)
    engine.launchKernel("kernelMain", oDev, ())
    engine.copyToHostFromDevice(addr out32[0], oDev, 4)
    doAssert out32[0] == 7, "set field must keep its value"
    doAssert out32[1] == 0, "omitted array field must be zeroed"
    doAssert out32[2] == 0, "omitted array field must be zeroed"
    doAssert out32[3] == 1, "omitted pointer field must be nil"
    echo "  OK"

when isMainModule:
  runTest()
