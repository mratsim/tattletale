## GLSL: the bit-reinterpret cast emits the float-bits builtins and reinterprets bits.
##
## Test that the launched kernel returns the host cast's bit pattern.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/types/t_glsl_bit_cast.nim

import std/unittest
import workspace/crucible

const bitsToUintGlsl = glsl:
  proc bitsToUintKernel(outBuf: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 1, 1).} =
    let lane = thread_index_in_threadgroup
    let f = float32(-2.0) - float32(lane)
    outBuf[lane] = cast[uint32](f)

const uintToBitsGlsl = glsl:
  proc uintToBitsKernel(outBuf: ptr UncheckedArray[float32]) {.global, workgroup: (4, 1, 1).} =
    let lane = thread_index_in_threadgroup
    let u = 0xC0000000'u32 + uint32(lane)
    outBuf[lane] = cast[float32](u)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "GLSL - bit-reinterpret cast emission":

    test "cast[uint32](float) reinterprets the bits on the device":
      # host cast is the independent truth: -2.0-lane gives
      # 0xC0000000 - lane * 0x00400000, a numeric conversion would round instead
      var engine = newEngine(runVulkan)
      engine.ingest(bitsToUintGlsl)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel<<(grid: (1, 1, 1), blk: (4, 1, 1))>>("bitsToUintKernel", oDev, ())
      var res: array[4, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      for lane in 0 ..< 4:
        let expected = cast[uint32](float32(-2.0) - float32(lane))
        check res[lane] == expected

    test "cast[float32](uint32) reinterprets the bits on the device":
      var engine = newEngine(runVulkan)
      engine.ingest(uintToBitsGlsl)
      let oDev = engine.allocDevice(float32, 4)
      engine.launchKernel<<(grid: (1, 1, 1), blk: (4, 1, 1))>>("uintToBitsKernel", oDev, ())
      var res: array[4, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      for lane in 0 ..< 4:
        let expected = 0xC0000000'u32 + uint32(lane)
        check cast[uint32](res[lane]) == expected

when isMainModule:
  runTest()
  echo "t_glsl_bit_cast: all checks passed"
