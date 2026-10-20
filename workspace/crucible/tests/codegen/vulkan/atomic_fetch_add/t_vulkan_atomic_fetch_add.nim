# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan: `atomicFetchAdd` on device memory.
##
## - fp needs GL_EXT_shader_atomic_float
## - MoltenVK exposes it on Metal 3+ hardware
## - equal adds serialize into distinct counts, the returned old
##   values are 0..31 and the counter ends at the exact sum
##
## Run:
##   cd tattletale && \
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/atomic_fetch_add/t_vulkan_atomic_fetch_add.nim

import std/[strutils, unittest, algorithm]
import workspace/crucible

const addFpVk = vulkan:
  proc addFp(counter: ptr float32, outBuf: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    let lane = thread_index_in_threadgroup
    outBuf[lane] = atomicFetchAdd(counter, float32(lane))

const addFpSameVk = vulkan:
  proc addFpSame(counter: ptr float32, outBuf: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    let lane = thread_index_in_threadgroup
    outBuf[lane] = atomicFetchAdd(counter, 1.0'f32)

const addIntVk = vulkan:
  proc addInt(counter: ptr int32) {.global, workgroup: (32, 1, 1).} =
    discard atomicFetchAdd(counter, int32(1))

const addMultiTgVk = vulkan:
  proc addMultiTg(counter: ptr float32) {.global, workgroup: (32, 1, 1).} =
    discard atomicFetchAdd(counter, 1.0'f32)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Vulkan - atomicFetchAdd":

    test "atomicFetchAdd emits atomicAdd and the atomic float extension":
      check "atomicAdd(" in addFpVk
      check "#extension GL_EXT_shader_atomic_float : enable" in addFpVk

    test "fp: 32 lanes add 1.0, each reads a distinct old value 0..31":
      var engine = bkVulkan.newEngine()
      engine.ingest(addFpSameVk)
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
      var engine = bkVulkan.newEngine()
      engine.ingest(addFpVk)
      let cDev = engine.allocDevice(float32, 1)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("addFp", cDev.mem, ())
      var counter: array[1, float32]
      engine.copyToHostFromDevice(addr counter[0], cDev, 1)
      # sum of lanes 0..31 = 496, exact in fp32 and order-independent
      check counter[0] == 496.0

    test "int: 32 lanes increment, counter ends at 32":
      var engine = bkVulkan.newEngine()
      engine.ingest(addIntVk)
      let cDev = engine.allocDevice(int32, 1)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("addInt", cDev.mem, ())
      var counter: array[1, int32]
      engine.copyToHostFromDevice(addr counter[0], cDev, 1)
      check counter[0] == 32

    test "fp: 4 threadgroups x 32 lanes, counter ends at 128":
      var engine = bkVulkan.newEngine()
      engine.ingest(addMultiTgVk)
      let cDev = engine.allocDevice(float32, 1)
      engine.launchKernel<<(grid: (4, 1), blk: (32, 1))>>("addMultiTg", cDev.mem, ())
      var counter: array[1, float32]
      engine.copyToHostFromDevice(addr counter[0], cDev, 1)
      check counter[0] == 128.0

when isMainModule:
  runTest()
