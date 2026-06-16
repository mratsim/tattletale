## CuTe Layout + GEMM tile — Vulkan (GLSL/SPIR-V) backend
## Run with: nim c -r -d:vulkan --hints:off --warnings:off \
##   workspace/crucible/tests/codegen/vulkan/test_cute_layout_vk.nim
import std/strformat
import workspace/crucible/src/codegen/vk

type
  Layout[S0, S1, D0, D1: static int] = object
    data: array[(S0-1)*D0 + (S1-1)*D1 + 1, uint32]

  Tile[M, N: static int] = object
    data: array[M * N, uint32]

proc at[S0, S1, D0, D1: static int](l: Layout[S0, S1, D0, D1];
                                       r, c: uint32): uint32 {.device.} =
  l.data[r * uint32(D0) + c * uint32(D1)]

proc tileAt[M, N: static int](t: Tile[M, N]; r, c: uint32): uint32 {.device.} =
  t.data[r * uint32(N) + c]

const kernelVk = vulkan:
  proc cuteKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let l1 = Layout[4, 1, 2, 1](
      data: [10'u32, 0'u32, 20'u32, 0'u32, 30'u32, 0'u32, 40'u32])
    output[0] = at(l1, 0'u32, 0'u32)
    output[1] = at(l1, 1'u32, 0'u32)
    output[2] = at(l1, 3'u32, 0'u32)
    let l2 = Layout[2, 3, 3, 1](
      data: [1'u32, 2'u32, 3'u32, 4'u32, 5'u32, 6'u32])
    output[3] = at(l2, 0'u32, 0'u32)
    output[4] = at(l2, 1'u32, 2'u32)
    let t = Tile[2, 3](data: [10'u32, 20'u32, 30'u32, 40'u32, 50'u32, 60'u32])
    output[5] = tileAt(t, 0'u32, 1'u32)
    output[6] = tileAt(t, 1'u32, 2'u32)
    let a00 = 1'u32; let a01 = 2'u32
    let a10 = 3'u32; let a11 = 4'u32
    let gemmB = Tile[2, 3](data: [5'u32, 6'u32, 7'u32, 8'u32, 9'u32, 10'u32])
    output[8] = a00 * tileAt(gemmB, 0'u32, 0'u32) + a01 * tileAt(gemmB, 1'u32, 0'u32)
    output[9] = a10 * tileAt(gemmB, 0'u32, 2'u32) + a11 * tileAt(gemmB, 1'u32, 2'u32)

echo "=== Vulkan CuTe Layout generation ===\n"
echo kernelVk; echo ""

when defined(vulkan):
  echo "=== Vulkan execution ===\n"
  block:
    var ctx = initVulkan()
    defer: ctx.shutdown()
    let r = execVulkan(ctx, kernelVk, "cuteKernel", outputBytes=40, inputs=@[])
    let res = cast[ptr array[10, uint32]](r[0].addr)
    doAssert res[0]==10 and out[1]==20 and out[2]==40
    doAssert res[3]==1 and out[4]==6
    doAssert res[5]==20 and out[6]==60
    doAssert res[8]==21 and out[9]==61
    echo "  OK — CuTe Layout + GEMM (Vulkan)"
else:
  echo "---\nTo run execution tests, recompile with -d:vulkan"
