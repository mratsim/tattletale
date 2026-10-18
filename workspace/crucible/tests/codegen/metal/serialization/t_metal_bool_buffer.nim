## Metal bool buffers marshal at their Nim width, 1 byte per element (scalar bools stay 4-byte i32 on the host).
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_bool_buffer.nim

import std/unittest
import workspace/crucible

# ── Kernels ─────────────────────────────────────────────────────────────────

const kernelBoolRead = metal:
  proc boolReadKernel(output: ptr UncheckedArray[uint32]; flags: ptr UncheckedArray[bool]) {.global.} =
    let i = thread_position_in_grid.x
    if flags[i]:
      output[i] = 1'u32
    else:
      output[i] = 0'u32

const kernelBoolWrite = metal:
  proc boolWriteKernel(output: ptr UncheckedArray[bool]; vals: ptr UncheckedArray[uint32]) {.global.} =
    let i = thread_position_in_grid.x
    output[i] = vals[i] != 0'u32

# ── Host side ───────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - bool buffer marshalling":

    test "array[bool] input reads 1-byte elements":
      var engine = bkMetal.init()
      engine.ingest(kernelBoolRead)
      let flags = [true, false, true, false]
      let fDev = engine.allocDevice(bool, flags.len)
      engine.copyToDeviceFromHost(fDev, unsafeAddr flags[0], flags.len)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel<<(grid: (4, 1), blk: (1, 1))>>("boolReadKernel", oDev, (fDev,))
      var res: array[4, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      engine.freeDevice(fDev)
      engine.freeDevice(oDev)
      check res == [1'u32, 0'u32, 1'u32, 0'u32]

    test "seq[bool] input reads 1-byte elements":
      var engine = bkMetal.init()
      engine.ingest(kernelBoolRead)
      let flags = @[true, false, true, false]
      let fDev = engine.allocDevice(bool, flags.len)
      engine.copyToDeviceFromHost(fDev, addr flags[0], flags.len)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel<<(grid: (4, 1), blk: (1, 1))>>("boolReadKernel", oDev, (fDev,))
      var res: array[4, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      engine.freeDevice(fDev)
      engine.freeDevice(oDev)
      check res == [1'u32, 0'u32, 1'u32, 0'u32]

    test "array[bool] output writes 1-byte elements":
      var engine = bkMetal.init()
      engine.ingest(kernelBoolWrite)
      let vals = [1'u32, 0'u32, 1'u32, 0'u32]
      let vDev = engine.allocDevice(uint32, vals.len)
      engine.copyToDeviceFromHost(vDev, unsafeAddr vals[0], vals.len)
      let oDev = engine.allocDevice(bool, 4)
      engine.launchKernel<<(grid: (4, 1), blk: (1, 1))>>("boolWriteKernel", oDev, (vDev,))
      var outBools: array[4, bool]
      engine.copyToHostFromDevice(addr outBools[0], oDev, 4)
      engine.freeDevice(vDev)
      engine.freeDevice(oDev)
      check outBools == [true, false, true, false]

when isMainModule:
  runTest()
