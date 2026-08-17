## Metal 2-D dispatch: the `{.workgroup: (X, Y).}` annotation is accepted
## for DSL parity with the webgpu suite. Metal never bakes it into the shader,
## because the threadgroup size is dispatch-time. The `<<(grid, blk)>>`
## chevrons drive `dispatchThreadgroups`, so an explicit blk is required.
## A plain `run()` dispatches blk=1 on Metal. The kernel indexes the output
## with `global_id` (emitted as `bid * bdim + tid`), so every thread writes exactly its own slot
## and the 8×8 output is the identity pattern.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_ndrange2d.nim

import std/unittest
import workspace/crucible

const kernel2d = metal:
  proc grid2d(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    C[global_id.y * 8'u32 + global_id.x] = global_id.y * 8'u32 + global_id.x

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernel2d

  suite "Metal - multi-axis dispatch":

    test "2D: grid (2, 4) × blk (4, 2) = 8×8 work-items":
      var engine = bkMetal.init()
      engine.ingest(kernel2d)
      var res: array[64, uint32]
      engine.run<<(grid: (2, 4), blk: (4, 2))>>("grid2d", res, ())
      for i in 0 ..< 64:
        check res[i] == uint32(i)

when isMainModule:
  runTest()
