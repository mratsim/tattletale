## CuTe Layout  tile — OpenCL backend
## Run with: nim c -r workspace/crucible/tests/codegen/opencl/test_cute_layout_cl.nim
import std/strformat
import workspace/crucible/src/codegen/cl

type
  Layout[S: static tuple, D: static tuple] = object
    discard
  Tile[M, N: static int] = object
    data: array[M * N, uint32]

proc tileAt[M, N: static int](t: Tile[M, N]; r, c: uint32): uint32 =
  t.data[r * uint32(N) + c]

const kernelCl = opencl:
  proc cuteKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let t = Tile[2, 3](data: [10'u32, 20'u32, 30'u32, 40'u32, 50'u32, 60'u32])
    output[0] = tileAt(t, 0'u32, 0'u32)
    output[1] = tileAt(t, 0'u32, 1'u32)
    output[2] = tileAt(t, 1'u32, 2'u32)
    let a00 = 1'u32; let a01 = 2'u32; let a10 = 3'u32; let a11 = 4'u32
    let gemmB = Tile[2, 3](data: [5'u32, 6'u32, 7'u32, 8'u32, 9'u32, 10'u32])
    output[4] = a00 * tileAt(gemmB, 0'u32, 0'u32) + a01 * tileAt(gemmB, 1'u32, 0'u32)
    output[5] = a10 * tileAt(gemmB, 0'u32, 2'u32) + a11 * tileAt(gemmB, 1'u32, 2'u32)

echo "=== OpenCL CuTe Layout generation ===\n"
echo kernelCl; echo ""

echo "=== OpenCL execution ===\n"
block:
  var ctx = initOpenCL()
  defer: ctx.shutdown()
  var d: array[1, uint32]
  let r = execOpenCL(ctx, kernelCl, "cuteKernel", outputBytes=24, inputs = [(cast[pointer](d[0].addr), 4)])
  let res = cast[ptr array[6, uint32]](r[0].addr)
  doAssert res[0]==10 and res[1]==20 and res[2]==60
  doAssert res[4]==21 and res[5]==61
  echo "  OK — CuTe Layout (descriptor) + Tile  (OpenCL)"
