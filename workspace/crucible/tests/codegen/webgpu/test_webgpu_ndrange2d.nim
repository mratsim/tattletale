## WebGPU 2D dispatch, the `{.workgroup: (X, Y).}` annotation bakes the workgroup size
## and the chevron form drives the dispatchWorkgroups group counts.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_ndrange2d.nim

import std/[unittest]
import workspace/crucible

const kernel2d = webgpu:
  proc grid2d(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    let flat = global_id.y * 8 + global_id.x
    C[flat] = uint32(flat)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernel2d

  suite "WebGPU - multi-axis dispatch":

    test "2D: baked (4, 2) × grid (2, 4) = 8×8 work-items":
      var engine = bkWGSL.init()
      engine.ingest(kernel2d)
      var res: array[64, uint32]
      let oDev = engine.allocDevice(res.len * sizeof(uint32))
      engine.launchKernel<<(grid: (2, 4), blk: (4, 2))>>("grid2d", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len * sizeof(uint32))
      engine.freeDevice(oDev)
      for i in 0 ..< 64:
        check res[i] == uint32(i)

when isMainModule:
  runTest()
