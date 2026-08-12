## CuTe Layout + Tile dot products — Vulkan (GLSL/SPIR-V) backend
## Run with: nim c -r workspace/crucible/tests/codegen/vulkan/test_cute_layout_vk.nim
import std/strformat
import workspace/crucible

type
  Layout[S: static tuple, D: static tuple] = object
    discard
  Tile[M, N: static int] = object
    data: array[M * N, uint32]

proc tileAt[M, N: static int](t: Tile[M, N]; r, c: uint32): uint32 =
  t.data[r * uint32(N) + c]

const kernelVk = vulkan:
  proc cuteKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let t = Tile[2, 3](data: [10'u32, 20'u32, 30'u32, 40'u32, 50'u32, 60'u32])
    output[0] = tileAt(t, 0'u32, 0'u32)
    output[1] = tileAt(t, 0'u32, 1'u32)
    output[2] = tileAt(t, 1'u32, 2'u32)
    let a00 = 1'u32; let a01 = 2'u32; let a10 = 3'u32; let a11 = 4'u32
    let gemmB = Tile[2, 3](data: [5'u32, 6'u32, 7'u32, 8'u32, 9'u32, 10'u32])
    output[4] = a00 * tileAt(gemmB, 0'u32, 0'u32) + a01 * tileAt(gemmB, 1'u32, 0'u32)
    output[5] = a10 * tileAt(gemmB, 0'u32, 2'u32) + a11 * tileAt(gemmB, 1'u32, 2'u32)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo "=== Vulkan CuTe Layout generation ===\n"
  echo kernelVk; echo ""

  echo "=== Vulkan execution ===\n"
  block:
    var engine = bkVulkan.init()
    engine.ingest(kernelVk)
    var res: array[6, uint32]
    engine.run("cuteKernel", res, ())
    doAssert res[0]==10 and res[1]==20 and res[2]==60
    doAssert res[4]==21 and res[5]==61
    echo "  OK — CuTe Layout + Tile  (Vulkan)"

when isMainModule:
  runTest()
