# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal: `atomicFetchAdd` on device memory (fp32 needs Metal 3+).
## Test that every lane's add lands exactly once: the returned old
## values form the multiset 0..31 and the counter ends at the exact sum.
##
## Run:
##   cd tattletale && \
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/atomic_fetch_add/t_metal_atomic_fetch_add.nim

import std/[strutils, unittest, algorithm]
import workspace/crucible

const addFpMsl = metal:
  proc addFp(counter: ptr float32, outBuf: ptr UncheckedArray[float32]) {.global.} =
    let lane = thread_index_in_threadgroup
    outBuf[lane] = atomicFetchAdd(counter, float32(lane))

const addFpSameMsl = metal:
  proc addFpSame(counter: ptr float32, outBuf: ptr UncheckedArray[float32]) {.global.} =
    let lane = thread_index_in_threadgroup
    outBuf[lane] = atomicFetchAdd(counter, 1.0'f32)

const addIntMsl = metal:
  proc addInt(counter: ptr int32) {.global.} =
    discard atomicFetchAdd(counter, int32(1))

const addMultiTgMsl = metal:
  proc addMultiTg(counter: ptr float32) {.global.} =
    discard atomicFetchAdd(counter, 1.0'f32)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - atomicFetchAdd":

    test "atomicFetchAdd emits atomic_fetch_add_explicit over an atomic_float cast":
      check "atomic_fetch_add_explicit((volatile device atomic_float*)" in addFpMsl

    test "fp: 32 lanes add 1.0, each reads a distinct old value 0..31":
      var engine = bkMetal.newEngine()
      engine.ingest(addFpSameMsl)
      let cDev = engine.allocDevice(float32, 1)
      let oDev = engine.allocDevice(float32, 32)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("addFpSame", cDev.mem, (oDev.mem, ()))
      var old: array[32, float32]
      engine.copyToHostFromDevice(addr old[0], oDev, 32)
      var counter: array[1, float32]
      engine.copyToHostFromDevice(addr counter[0], cDev, 1)
      # equal adds serialize into distinct counts, whatever the order:
      # the returned old values are exactly 0..31
      old.sort()
      for lane in 0 ..< 32:
        check old[lane] == float32(lane)
      check counter[0] == 32.0

    test "fp: 32 lanes add their lane index, counter ends at 496":
      var engine = bkMetal.newEngine()
      engine.ingest(addFpMsl)
      let cDev = engine.allocDevice(float32, 1)
      let oDev = engine.allocDevice(float32, 32)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("addFp", cDev.mem, (oDev.mem, ()))
      var old: array[32, float32]
      engine.copyToHostFromDevice(addr old[0], oDev, 32)
      var counter: array[1, float32]
      engine.copyToHostFromDevice(addr counter[0], cDev, 1)
      # sum of lanes 0..31 = 496, exact in fp32 and order-independent
      # distinct adds make the per-lane old values order-dependent,
      # bounded by the exact sum
      for lane in 0 ..< 32:
        check old[lane] >= 0.0 and old[lane] <= 496.0
      check counter[0] == 496.0

    test "int: 32 lanes increment, counter ends at 32":
      var engine = bkMetal.newEngine()
      engine.ingest(addIntMsl)
      let cDev = engine.allocDevice(int32, 1)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("addInt", cDev.mem, ())
      var counter: array[1, int32]
      engine.copyToHostFromDevice(addr counter[0], cDev, 1)
      check counter[0] == 32

    test "fp: 4 threadgroups x 32 lanes, counter ends at 128":
      var engine = bkMetal.newEngine()
      engine.ingest(addMultiTgMsl)
      let cDev = engine.allocDevice(float32, 1)
      engine.launchKernel<<(grid: (4, 1), blk: (32, 1))>>("addMultiTg", cDev.mem, ())
      var counter: array[1, float32]
      engine.copyToHostFromDevice(addr counter[0], cDev, 1)
      check counter[0] == 128.0

when isMainModule:
  runTest()
