## Vulkan 2D dispatch — the `{.workgroup: (X, Y).}` annotation bakes a 3D
## local_size_xyz into the GLSL, and the `<<(grid, blk)>>` chevrons drive
## vkCmdDispatch group counts (blk must match the baked size, validated).
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_ndrange2d.nim

import std/[unittest]
import workspace/crucible

const kernel2d = vulkan:
  proc grid2d(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    # direct builtin use — binding gl_GlobalInvocationID to a local let
    # hits an addr-lowering gap (global-let path), so index it inline
    C[gl_GlobalInvocationID[1] * 8'u32 + gl_GlobalInvocationID[0]] =
      gl_GlobalInvocationID[1] * 8'u32 + gl_GlobalInvocationID[0]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernel2d

  suite "Vulkan - multi-axis dispatch":

    test "2D: baked (4, 2) × grid (2, 4) = 8×8 work-items":
      var engine = bkVulkan.init()
      engine.ingest(kernel2d)
      var res: array[64, uint32]
      engine.run<<(grid: (2, 4), blk: (4, 2))>>("grid2d", res, ())
      for i in 0 ..< 64:
        check res[i] == uint32(i)

when isMainModule:
  runTest()
