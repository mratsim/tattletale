# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal: `reduceAdd` warp-synchronous reduction.
## Test that `reduceAdd` sums the 32 lanes and broadcasts the total to every lane.
##
## Run:
##   cd tattletale && \
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/warp_sync_reductions/t_metal_warp_sync_reductions.nim

import std/[strutils, unittest]
import workspace/crucible

const scalarMsl = metal:
  proc scalarReduce(outBuf: ptr UncheckedArray[float32]) {.global.} =
    let lane = thread_index_in_threadgroup
    let v = float32(lane)
    outBuf[lane] = reduceAdd(v)

const slotsMsl = metal:
  proc slotsReduce(outBuf: ptr UncheckedArray[float32]) {.global.} =
    let lane = thread_index_in_threadgroup
    for j in 0 ..< 4:
      let v = float32(lane) + float32(j)
      outBuf[lane * 4 + j] = reduceAdd(v)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - warp-synchronous reductions":

    test "reduceAdd emits simd_sum":
      check "simd_sum(" in scalarMsl

    test "scalar reduce sums the 32 lanes and broadcasts to each":
      var engine = bkMetal.newEngine()
      engine.ingest(scalarMsl)
      let oDev = engine.allocDevice(float32, 32)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("scalarReduce", oDev, ())
      var res: array[32, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 32)
      # sum of lanes 0..31 = 496, every lane holds the total
      for lane in 0 ..< 32:
        check res[lane] == 496.0

    test "per-slot reduceAdd sums each slot across the 32 lanes":
      var engine = bkMetal.newEngine()
      engine.ingest(slotsMsl)
      let oDev = engine.allocDevice(float32, 32 * 4)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("slotsReduce", oDev, ())
      var res: array[32 * 4, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 32 * 4)
      # slot j: sum over lanes of (lane + j) = 496 + 32*j
      for lane in 0 ..< 32:
        for j in 0 ..< 4:
          check res[lane * 4 + j] == float32(496 + 32 * j)

when isMainModule:
  runTest()
