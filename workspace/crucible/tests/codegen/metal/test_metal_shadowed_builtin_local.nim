## Metal: attribute-param injection and a shadowed builtin local.
## `singleAttrKernel` uses only `thread_position_in_grid` — the printer must
## inject exactly one `uint3` attribute param. `shadowKernel` declares a local
## `let thread_position_in_grid = 5'u32` — the printer must inject NO attribute
## param for the shadowed name (a collision would fail MSL compilation) and the
## local must win. Both run through `engine.run()` and assert byte-exact output.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_shadowed_builtin_local.nim

import std/unittest
import workspace/crucible

const singleAttrMsl = metal:
  proc singleAttrKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[thread_position_in_grid.x] = thread_position_in_grid.x

const shadowMsl = metal:
  proc shadowKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let thread_position_in_grid = 5'u32
    output[0] = thread_position_in_grid

proc runTest() =
  suite "Metal - attribute injection + shadowed local":
    test "single attr param tracks the grid x position":
      var engine = bkMetal.init()
      engine.ingest(singleAttrMsl)
      var res: array[2, uint32]
      engine.run<<(grid: (2, 1), blk: (1, 1))>>("singleAttrKernel", res, ())
      check res[0] == 0'u32
      check res[1] == 1'u32

    test "local shadowing the attr name wins; no attr param injected":
      var engine = bkMetal.init()
      engine.ingest(shadowMsl)
      var res: array[1, uint32]
      engine.run("shadowKernel", res, ())
      check res[0] == 5'u32

when isMainModule:
  runTest()
