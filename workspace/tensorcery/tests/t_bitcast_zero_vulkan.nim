# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan (MoltenVK): the bit-reinterpret cast moves +0.0 exactly.
##
## Test that zero bit patterns survive the device bitcast round trip:
## f32 +0.0/-0.0 to u32 0x00000000/0x80000000 and back, and the top 16
## bits of f32 +0.0 (the bf16 view) are 0x0000. The f16 and bf16 formats
## store zeros as these bit patterns.
##
## Run (from the worktree root):
##   nim c -r --hints:off --warnings:off --outdir:build/tests/t_bitcast_zero_vulkan \
##     --nimcache:nimcache/tests/t_bitcast_zero_vulkan workspace/tensorcery/tests/t_bitcast_zero_vulkan.nim

import std/strutils
import workspace/crucible

const zerosToUintGlsl = glsl:
  proc zerosToUintKernel(outBuf: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 1, 1).} =
    let lane = thread_index_in_threadgroup
    let f = if lane mod 2 == 0: float32(+0.0) else: float32(-0.0)
    outBuf[lane] = cast[uint32](f)

const uintToZerosGlsl = glsl:
  proc uintToZerosKernel(outBuf: ptr UncheckedArray[float32]) {.global, workgroup: (4, 1, 1).} =
    let lane = thread_index_in_threadgroup
    let u = if lane mod 2 == 0: 0x00000000'u32 else: 0x80000000'u32
    outBuf[lane] = cast[float32](u)

proc runTest() =
  var engine = newEngine(runVulkan)
  block f32ToU32:
    engine.ingest(zerosToUintGlsl)
    let oDev = engine.allocDevice(uint32, 4)
    engine.launchKernel<<(grid: (1, 1, 1), blk: (4, 1, 1))>>("zerosToUintKernel", oDev, ())
    var res: array[4, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 4)
    for lane in 0 ..< 4:
      let want = if lane mod 2 == 0: 0x00000000'u32 else: 0x80000000'u32
      doAssert res[lane] == want,
        "f32 zero bitcast lane " & $lane & ": " & res[lane].toHex & " != " & want.toHex
    doAssert (res[0] shr 16) == 0x0000'u32, "bf16 view of f32 +0.0 must be 0x0000"
    echo "  OK: f32 +0.0/-0.0 bitcast to 0x00000000/0x80000000, bf16 view of +0.0 is 0x0000"

  block u32ToF32:
    engine.ingest(uintToZerosGlsl)
    let oDev = engine.allocDevice(float32, 4)
    engine.launchKernel<<(grid: (1, 1, 1), blk: (4, 1, 1))>>("uintToZerosKernel", oDev, ())
    var res: array[4, float32]
    engine.copyToHostFromDevice(addr res[0], oDev, 4)
    for lane in 0 ..< 4:
      let want = if lane mod 2 == 0: 0x00000000'u32 else: 0x80000000'u32
      doAssert cast[uint32](res[lane]) == want,
        "u32 zero bitcast lane " & $lane & " wrong bit pattern"
      doAssert res[lane] == 0.0'f32
    echo "  OK: u32 0x00000000/0x80000000 bitcast to f32 +0.0/-0.0"

  echo "t_bitcast_zero_vulkan: ALL GREEN"

when isMainModule:
  runTest()
