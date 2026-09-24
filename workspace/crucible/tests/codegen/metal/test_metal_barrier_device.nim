## Metal device-scope barrier emission check.
##
## - `threadgroup_barrier_device` spells `threadgroup_barrier(mem_flags::mem_device)`, ordering the device address space
## - `threadgroup_barrier` spells `threadgroup_barrier(mem_flags::mem_threadgroup)`, ordering the threadgroup address space
## - a device exchange behind the threadgroup fence stays unordered
##   (see `test_metal_cross_vocabulary` for the canonical shape)
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests
##   (from tattletale) against workspace/crucible/tests/codegen/metal/test_metal_barrier_device.nim

import std/[strutils, unittest]
import workspace/crucible

const deviceExchangeMsl = metal:
  proc deviceExchangeKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Thread t writes the slot belonging to thread (63 - t), reads back
    # the slot written by thread (63 - t), the exchange the fence orders.
    output[blockIdx.x * 64'u32 + 63'u32 - thread_position_in_threadgroup.x] =
      blockIdx.x * 64'u32 + thread_position_in_threadgroup.x
    threadgroup_barrier_device()
    output[blockIdx.x * 64'u32 + thread_position_in_threadgroup.x] =
      output[blockIdx.x * 64'u32 + 63'u32 - thread_position_in_threadgroup.x]

proc runTest() =
  suite "Metal - device-scope barrier":

    test "threadgroup_barrier_device emits the mem_device spelling and runs":
      var engine = bkMetal.init()
      engine.ingest(deviceExchangeMsl)
      let msl = engine.getArtifact()
      check "threadgroup_barrier(mem_flags::mem_device)" in msl
      check "threadgroup_barrier(mem_flags::mem_threadgroup)" notin msl
      var res: array[128, uint32]
      engine.run<<(grid: (2, 1), blk: (64, 1))>>("deviceExchangeKernel", res, ())
      for i in 0 ..< 128:
        check res[i] == uint32(i)

when isMainModule:
  runTest()
