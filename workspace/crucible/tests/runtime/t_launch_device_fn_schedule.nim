# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## launchDeviceFn `do:` schedule substitution:
##
## - `grid = (n, 1)` with `n` a launch value: the schedule expression binds
##   to each launch's value before that launch
## - each grid position writes `thread_position_in_grid.x` into its own slot
## - the buffer contents prove the substituted geometry reached the dispatch
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/runtime/t_launch_device_fn_schedule.nim

import workspace/crucible

proc gridId(outBuf: ptr UncheckedArray[uint32], n: uint32) =
  outBuf[thread_position_in_grid.x] = thread_position_in_grid.x

proc runTest() =   # private, tests run in a proc so engines are destroyed at return
  var engine = bkMetal.init()
  let dev = engine.allocDevice(4 * sizeof(uint32))

  engine.launchDeviceFn gridId(dev, 3) do:
    grid = (n, 1)
    blk = (1, 1)
  var g: array[4, uint32]
  engine.copyToHostFromDevice(addr g[0], dev, sizeof(g))
  doAssert g[0 ..< 3] == [0'u32, 1, 2],
    "grid = (n, 1) must substitute the launch value n=3: " & $g

  engine.launchDeviceFn gridId(dev, 2) do:
    grid = (n, 1)
    blk = (1, 1)
  engine.copyToHostFromDevice(addr g[0], dev, sizeof(g))
  doAssert g[0 ..< 2] == [0'u32, 1],
    "grid = (n, 1) must substitute the launch value n=2: " & $g

  engine.freeDevice(dev)

when isMainModule:
  runTest()
  echo "  All launchDeviceFn schedule tests passed."
