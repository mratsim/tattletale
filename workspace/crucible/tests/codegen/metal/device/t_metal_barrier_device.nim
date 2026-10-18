## Metal device-scope barrier: `threadgroup_barrier_device` must emit
## `threadgroup_barrier(mem_flags::mem_device)` in the MSL and order
## the device address space, each thread reads the slot its peer thread wrote.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_barrier_device.nim

import std/[strutils, unittest]
import workspace/crucible

const deviceExchangeMsl = metal:
  proc deviceExchangeKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Thread t writes its own slot, then copies the slot written by thread
    # (63 - t) into its own slot, the exchange the fence orders.
    let t = thread_position_in_threadgroup.x
    let base = blockIdx.x * 64
    output[base + t] = uint32(base + t)
    threadgroup_barrier_device()
    output[base + t] = output[base + 63 - t]

proc runTest() =
  suite "Metal - device-scope barrier":

    test "threadgroup_barrier_device emits the mem_device spelling and runs":
      var engine = bkMetal.init()
      engine.ingest(deviceExchangeMsl)
      let msl = engine.getArtifact()
      check "threadgroup_barrier(mem_flags::mem_device)" in msl
      check "threadgroup_barrier(mem_flags::mem_threadgroup)" notin msl
      let oDev = engine.allocDevice(uint32, 128)
      engine.launchKernel<<(grid: (2, 1), blk: (64, 1))>>("deviceExchangeKernel", oDev, ())
      var res: array[128, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 128)
      engine.freeDevice(oDev)
      for i in 0 ..< 128:
        check res[i] == uint32((i div 64) * 64 + 63 - (i mod 64))

when isMainModule:
  runTest()
