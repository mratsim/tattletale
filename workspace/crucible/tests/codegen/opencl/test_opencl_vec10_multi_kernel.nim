## Multi-kernel Vec10 add/mul — OpenCL backend
## Run with:
##   nim c -r workspace/crucible/tests/codegen/opencl/test_vec10_multi_kernel.nim
import workspace/crucible

const kernelCode = opencl:
  proc vec10_add(a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32];
                 output: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 10:
      output[i] = a[i] + b[i]
  proc vec10_mul(a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32];
                 output: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 10:
      output[i] = a[i] * b[i]

echo kernelCode

var a: array[10, uint32] = [1'u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]
var b: array[10, uint32] = [10'u32, 20, 30, 40, 50, 60, 70, 80, 90, 100]

echo "=== Running vec10_add ===\n"
block:
  var engine = bkOpenCL.init()
  engine.ingest(kernelCode)
  var res: array[10, uint32]
  engine.run("vec10_add", res, (a, b))
  for i in 0 ..< 10:
    doAssert res[i] == a[i] + b[i]
  echo "  OK — vec10_add"

echo "=== Running vec10_mul ===\n"
block:
  var engine = bkOpenCL.init()
  engine.ingest(kernelCode)
  var res: array[10, uint32]
  engine.run("vec10_mul", res, (a, b))
  for i in 0 ..< 10:
    doAssert res[i] == a[i] * b[i]
  echo "  OK — vec10_mul"

echo "\n  OK — Multi-kernel vec10 (OpenCL)"
