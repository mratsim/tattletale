## Metal: attribute-param injection and a shadowed builtin local.
## `singleAttrKernel` uses `thread_position_in_grid` only, the printer must
## inject exactly one attribute param. `shadowKernel` declares a local
## `thread_position_in_grid`, the local must win and no attribute param may
## be injected for the shadowed name.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_shadowed_builtin_local.nim

import std/unittest
import workspace/crucible

const singleAttrMsl = metal:
  proc singleAttrKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[thread_position_in_grid.x] = uint32(thread_position_in_grid.x)

const shadowMsl = metal:
  proc shadowKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let thread_position_in_grid = 5'u32
    output[0] = thread_position_in_grid

proc runTest() =
  suite "Metal - attribute injection + shadowed local":
    test "single attr param tracks the grid x position":
      var engine = bkMetal.newEngine()
      engine.ingest(singleAttrMsl)
      let oDev = engine.allocDevice(uint32, 2)
      engine.launchKernel<<(grid: (2, 1), blk: (1, 1))>>("singleAttrKernel", oDev, ())
      var res: array[2, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 0'u32
      check res[1] == 1'u32

    test "local shadowing the attr name wins; no attr param injected":
      var engine = bkMetal.newEngine()
      engine.ingest(shadowMsl)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("shadowKernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 5'u32

when isMainModule:
  runTest()
