# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## launchDeviceFn chevron contract:
## - `<<` takes launch geometry only: sharedMem and stream do not exist
##   on the device-fn path and must be a compile error, never silently dropped
## - the plain `(grid, blk)` form must compile so the rejection is attributable
##   to the extra fields
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/runtime/t_launch_device_fn_chevron_config.nim

import workspace/crucible

proc setOne(outBuf: ptr UncheckedArray[uint32]) =
  outBuf[0] = 7'u32

proc runTest() =
  var engine = bkMetal.init()
  let outDev = engine.allocDevice(sizeof(uint32))
  let grid = (1, 1)
  let blk = (1, 1)

  block:  # the plain geometry form launches
      doAssert compiles(engine.launchDeviceFn<<(grid, blk)>> setOne(outDev))

  block:  # sharedMem/stream on the device-fn chevron are compile errors
      doAssert not compiles(
        engine.launchDeviceFn<<(grid, blk, sharedMem: 1024)>> setOne(outDev))
      doAssert not compiles(
        engine.launchDeviceFn<<(grid: (1, 1), blk: (1, 1), sharedMem: 1024)>> setOne(outDev))
      doAssert not compiles(
        engine.launchDeviceFn<<(grid, blk, 1024)>> setOne(outDev))
      doAssert not compiles(
        engine.launchDeviceFn<<(grid: (1, 1), blk: (1, 1), stream: 7)>> setOne(outDev))

  block:  # a real launch through the chevron form: geometry reaches the dispatch
      proc gridId(outBuf: ptr UncheckedArray[uint32]) =
        outBuf[thread_position_in_grid.x] = thread_position_in_grid.x
      let geomDev = engine.allocDevice(3 * sizeof(uint32))
      engine.launchDeviceFn<<(grid: (3, 1), blk: (1, 1))>>(gridId(geomDev))
      var g: array[3, uint32]
      engine.copyToHostFromDevice(addr g[0], geomDev, sizeof(g))
      doAssert g == [0'u32, 1, 2],
        "the chevron grid must reach the dispatch, one write per grid position: " & $g
      engine.freeDevice(geomDev)

  engine.freeDevice(outDev)

when isMainModule:
  runTest()
