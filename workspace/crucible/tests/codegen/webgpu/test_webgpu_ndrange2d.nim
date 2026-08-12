## WebGPU 2D dispatch — the `{.workgroup: (X, Y).}` annotation bakes a 3D
## @workgroup_size into the WGSL, and the `<<(grid, blk)>>` chevrons drive
## dispatchWorkgroups group counts (blk must match the baked size, validated).
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_ndrange2d.nim

import std/[unittest]
import workspace/crucible

const kernel2d = webgpu:
  proc grid2d(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    # inline use — a local `let gid = global_id` emits a wrong type
    # annotation (WgslGridDim vs the builtin's vec3<u32>)
    C[global_id.y * 8'u32 + global_id.x] = global_id.y * 8'u32 + global_id.x

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernel2d

  suite "WebGPU - multi-axis dispatch":

    test "2D: baked (4, 2) × grid (2, 4) = 8×8 work-items":
      var engine = bkWGSL.init()
      engine.ingest(kernel2d)
      var res: array[64, uint32]
      engine.run<<(grid: (2, 4), blk: (4, 2))>>("grid2d", res, ())
      for i in 0 ..< 64:
        check res[i] == uint32(i)

when isMainModule:
  runTest()
