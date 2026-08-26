## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan: SSBO dedup by position (not by param name).
##
## SSBO bindings are assigned by param POSITION, not param name, so kernels
## with differently-named ptr params at the same position share the same
## buffer. kernel1's `output`/`a` and kernel2's `y`/`x` are both lowered to
## binding 0 / binding 1, and both kernels are ingested and run through the
## engine with real buffers to prove the shared bindings work.
##
## Run:
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/tests/vulkan --nimcache:nimcache/tests/vulkan \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_ssbo_dedup.nim

import std/[strutils, unittest]
import workspace/crucible

const code = vulkan:
  proc kernel1(
      output: ptr UncheckedArray[uint32],
      a: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + 1'u32
  proc kernel2(
      y: ptr UncheckedArray[uint32],
      x: ptr UncheckedArray[uint32]) {.global.} =
    y[0] = x[0] * 2'u32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Vulkan - SSBO dedup by position":
    test "2 SSBOs, binding 0 = first param, binding 1 = second, shared across kernels":
      # Two kernels × 2 ptr params each must produce exactly 2 SSBOs, not 4:
      # kernel2's `y`/`x` are renamed to kernel1's canonical `output`/`a`,
      # so both kernels reference the same binding-0 / binding-1 buffers.
      # (`output` is a GLSL reserved keyword — emitted as `output_vk`.)
      let ssboCount = code.count("layout(set = 0, binding =")
      check ssboCount == 2
      check "layout(set = 0, binding = 0) buffer Buf0 { uint output_vk[]; };" in code
      check "layout(set = 0, binding = 1) buffer Buf1 { uint a[]; };" in code
      check "void kernel1()" in code
      check "void kernel2()" in code
      # kernel2's body reads the shared canonical names (position 0 and 1),
      # proving the bindings are shared across differently-named kernels.
      check "output_vk[0] = (a[0] * 2U);" in code

      # Runtime proof: both kernels run on the engine with real buffers.
      # kernel1: output = a + 1; kernel2: y = x * 2 — the second param
      # (binding 1) is shared, so the same input value fed to `a` and `x`
      # lands in the same buffer slot and each kernel reads it back.
      var engine = bkVulkan.init()
      engine.ingest(code)
      var res1: array[1, uint32]
      let a = [41'u32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("kernel1", res1, (a,))
      check res1[0] == 42'u32
      var res2: array[1, uint32]
      let x = [41'u32]
      engine.run << (grid: (1, 1), blk: (1, 1)) >> ("kernel2", res2, (x,))
      check res2[0] == 82'u32

when isMainModule:
  runTest()
