## OpenCL: nested-array whole assignment copies every dimension.
##
## Test that a kernel assigning one whole nested array to another reads back
## all 32 elements: the copy lowers to a `__builtin_memcpy` over the nested
## type and its `sizeof` must cover all dimensions.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/types/t_opencl_nested_array_assign.nim

import workspace/crucible

const code = opencl:
  proc nestedCopyKernel(output: ptr UncheckedArray[float32]) {.global.} =
    var b: array[4, array[8, float32]]
    for i in 0 ..< 4:
      for j in 0 ..< 8:
        b[i][j] = float32(i * 8 + j) * 1.5'f32 + 0.25'f32
    var a: array[4, array[8, float32]]
    a = b
    for i in 0 ..< 4:
      for j in 0 ..< 8:
        output[i * 8 + j] = a[i][j]

proc runTest() =
  var engine = newEngine(runOpenCL)
  engine.ingest(code)
  let oDev = engine.allocDevice(float32, 32)
  engine.launchKernel("nestedCopyKernel", oDev, ())
  var res: array[32, float32]
  engine.copyToHostFromDevice(addr res[0], oDev, 32)
  for i in 0 ..< 32:
    doAssert res[i] == float32(i) * 1.5'f32 + 0.25'f32,
      "slot " & $i & " carries the copied value, got " & $res[i]
  echo "  OK - t_opencl_nested_array_assign"

when isMainModule:
  runTest()
