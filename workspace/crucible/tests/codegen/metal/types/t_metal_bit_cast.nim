## Metal: the bit-reinterpret cast reinterprets the operand's bits.
##
## Test that `cast[uint32](f)` and `cast[float32](u)` move bit patterns, not numeric values.
## The kernels cast device inputs both ways and the readback matches the host Nim `cast` bit for bit, which is the semantics `cast` states.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_bit_cast.nim

import std/unittest
import workspace/crucible

const bitsToUintMsl = metal:
  proc bitsToUintKernel(outBuf: ptr UncheckedArray[uint32], inBuf: ptr UncheckedArray[float32]) {.global.} =
    let lane = thread_index_in_threadgroup
    outBuf[lane] = cast[uint32](inBuf[lane])

const uintToBitsMsl = metal:
  proc uintToBitsKernel(outBuf: ptr UncheckedArray[float32], inBuf: ptr UncheckedArray[uint32]) {.global.} =
    let lane = thread_index_in_threadgroup
    outBuf[lane] = cast[float32](inBuf[lane])

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - bit-reinterpret cast":

    test "cast[uint32](float) stores the exact bit pattern":
      var engine = newEngine(runMetal)
      engine.ingest(bitsToUintMsl)
      var inp: array[32, float32]
      var expected: array[32, uint32]
      for lane in 0 ..< 32:
        inp[lane] = float32(-2.0) - float32(lane)
        expected[lane] = cast[uint32](inp[lane])
      let iDev = engine.allocDevice(float32, 32)
      let oDev = engine.allocDevice(uint32, 32)
      engine.copyToDeviceFromHost(iDev, addr inp[0], 32)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("bitsToUintKernel", oDev, (iDev,))
      var res: array[32, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 32)
      for lane in 0 ..< 32:
        check res[lane] == expected[lane]

    test "cast[float32](uint32) reads the bit pattern back as the float":
      var engine = newEngine(runMetal)
      engine.ingest(uintToBitsMsl)
      var inp: array[32, uint32]
      var expected: array[32, float32]
      for lane in 0 ..< 32:
        inp[lane] = cast[uint32](float32(-2.0) - float32(lane))
        expected[lane] = float32(-2.0) - float32(lane)
      let iDev = engine.allocDevice(uint32, 32)
      let oDev = engine.allocDevice(float32, 32)
      engine.copyToDeviceFromHost(iDev, addr inp[0], 32)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("uintToBitsKernel", oDev, (iDev,))
      var res: array[32, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 32)
      for lane in 0 ..< 32:
        check res[lane] == expected[lane]

when isMainModule:
  runTest()
  echo "t_metal_bit_cast: all checks passed"
