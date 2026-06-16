## Multi-kernel Vec10 add/mul — WebGPU (WGSL) backend
## Run with:
##   nim c -r workspace/crucible/tests/codegen/webgpu/test_vec10_multi_kernel.nim
import workspace/crucible/src/codegen/wgpu

const code = webgpu:
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

echo code

var a: array[10, uint32] = [1'u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]
var b: array[10, uint32] = [10'u32, 20, 30, 40, 50, 60, 70, 80, 90, 100]

echo "=== Running vec10_add ===\n"
block:
  var ctx = initWgpu()
  defer: ctx.shutdown()
  let r = execWgpu(ctx, code, "vec10_add", outputBytes=40, inputs = [
    (cast[pointer](a[0].addr), 40),
    (cast[pointer](b[0].addr), 40)
  ])
  let res = cast[ptr array[10, uint32]](r[0].addr)
  for i in 0 ..< 10:
    doAssert res[i] == a[i] + b[i]
  echo "  OK — vec10_add"

echo "=== Running vec10_mul ===\n"
block:
  var ctx = initWgpu()
  defer: ctx.shutdown()
  let r = execWgpu(ctx, code, "vec10_mul", outputBytes=40, inputs = [
    (cast[pointer](a[0].addr), 40),
    (cast[pointer](b[0].addr), 40)
  ])
  let res = cast[ptr array[10, uint32]](r[0].addr)
  for i in 0 ..< 10:
    doAssert res[i] == a[i] * b[i]
  echo "  OK — vec10_mul"

echo "\n  OK — Multi-kernel vec10 (WebGPU)"
