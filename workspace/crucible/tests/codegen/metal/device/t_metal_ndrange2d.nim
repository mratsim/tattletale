## Metal 2-D dispatch: grid (2, 4) x blk (4, 2) runs 8x8 work-items, each
## thread writes its own slot through `thread_position_in_grid`, the 8x8
## output is the identity pattern.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_ndrange2d.nim

import std/unittest
import workspace/crucible

const kernel2d = metal:
  proc grid2d(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    let flat = thread_position_in_grid.y * 8 + thread_position_in_grid.x
    C[flat] = uint32(flat)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernel2d

  suite "Metal - multi-axis dispatch":

    test "2D: grid (2, 4) × blk (4, 2) = 8×8 work-items":
      var engine = bkMetal.init()
      engine.ingest(kernel2d)
      let oDev = engine.allocDevice(uint32, 64)
      engine.launchKernel<<(grid: (2, 4), blk: (4, 2))>>("grid2d", oDev, ())
      var res: array[64, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 64)
      for i in 0 ..< 64:
        check res[i] == uint32(i)

when isMainModule:
  runTest()
