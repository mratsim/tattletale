# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## forEach over a fully collapsed tile: every mode has length 1, under Metal.
## The SDPA rowStats tile at the shipped MT=8 constants has this shape.
## The reduced layout leaves no mode to iterate, so the flat index is 0.
##
## Two kernels exercise the collapsed index:
## - `_` matches the SDPA rescale-skip loop shape. On the unfixed tree
##   codegenMetal refuses the compile: "type not supported: NimNode (ntyRef)"
## - `j` reads the index. On the unfixed tree the sem check fails on `j: NimNode`
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/ceramic/tests/tile_algebra/t_foreach_metal.nim

import std/unittest
import workspace/crucible
import workspace/ceramic/src/tile_algebra

const forEachCollapsedMsl = metal:
  proc forEachCollapsedKernel(dst: ptr UncheckedArray[float32]) {.global.} =
    var t: tile(float32, (1, 1))
    forEach _, var d in t:
      d = 7.5'f32
    dst[0] = t.data[0]

const forEachCollapsedIndexedMsl = metal:
  proc forEachCollapsedIndexedKernel(dst: ptr UncheckedArray[float32]) {.global.} =
    var t: tile(float32, (1, 1))
    forEach j, var d in t:
      d = 7.5'f32
    forEach k, d in t:
      dst[k] = d

proc runTest() =
  # private: tests run in a proc so engines are destroyed at return
  suite "Ceramic - forEach over an all length-1 tile":
    test "single element write and read through the collapsed index":
      var engine = newEngine(runMetal)
      engine.ingest(forEachCollapsedMsl)
      let dstDev = engine.allocDevice(float32, 1)
      engine.launchKernel("forEachCollapsedKernel", dstDev, ())
      var res: array[1, float32]
      engine.copyToHostFromDevice(addr res[0], dstDev, 1)
      check res[0] == 7.5'f32

    test "collapsed index is the flat offset 0":
      var engine = newEngine(runMetal)
      engine.ingest(forEachCollapsedIndexedMsl)
      let dstDev = engine.allocDevice(float32, 1)
      engine.launchKernel("forEachCollapsedIndexedKernel", dstDev, ())
      var res: array[1, float32]
      engine.copyToHostFromDevice(addr res[0], dstDev, 1)
      check res[0] == 7.5'f32

when isMainModule:
  runTest()
