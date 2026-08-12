## OpenCL 2D/3D NDRange launch extents — the `<<(grid, blk)>>` chevrons map
## to a 3D clEnqueueNDRangeKernel: global = grid·blk per axis, local = blk.
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_ndrange2d.nim

import std/[unittest]
import workspace/crucible

const kernel2d = opencl:
  proc grid2d(C: ptr UncheckedArray[uint32]) {.global.} =
    let gx = get_global_id(0'u32)
    let gy = get_global_id(1'u32)
    C[gy * 8'u32 + gx] = gy * 8'u32 + gx

const kernel3d = opencl:
  proc grid3d(C: ptr UncheckedArray[uint32]) {.global.} =
    let gx = get_global_id(0'u32)
    let gy = get_global_id(1'u32)
    let gz = get_global_id(2'u32)
    C[gz * 16'u32 + gy * 4'u32 + gx] = gz * 16'u32 + gy * 4'u32 + gx

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "OpenCL - multi-axis NDRange work sizes":

    test "2D: grid (2, 4) × blk (4, 2) = 8×8 work-items":
      var engine = bkOpenCL.init()
      engine.ingest(kernel2d)
      var res: array[64, uint32]
      engine.run<<(grid: (2, 4), blk: (4, 2))>>("grid2d", res, ())
      for i in 0 ..< 64:
        check res[i] == uint32(i)

    test "3D: grid (2, 2, 2) × blk (2, 2, 2) = 4×4×4 work-items":
      var engine = bkOpenCL.init()
      engine.ingest(kernel3d)
      var res: array[64, uint32]
      engine.run<<(grid: (2, 2, 2), blk: (2, 2, 2))>>("grid3d", res, ())
      for i in 0 ..< 64:
        check res[i] == uint32(i)

when isMainModule:
  runTest()
