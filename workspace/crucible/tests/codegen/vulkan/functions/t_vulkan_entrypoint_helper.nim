# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan: entry-point parse with a void-returning device helper.
## parseEntryPoint must return the first kernel's name, never a device-helper name.
## Each kernel opens with its own `layout(local_size_x ...) in;` preamble.
## Entry point resolution anchors on that preamble, not on the first bare `void `.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/functions/t_vulkan_entrypoint_helper.nim

import std/[strformat, strutils]
import workspace/crucible

# The kernel calls a void-returning device function: the codegen forward-
# declares it (`void helper(uint x);`) before the kernel's preamble, which is
# exactly the shape that breaks a naive first-`void ` entry-point scan.
const helperVk = vulkan:
  proc helper(x: uint32) {.device.} =
    let y = x + 1

  proc entryKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    helper(41'u32)
    output[0] = 42'u32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo "=== Vulkan: entry point with void-returning device helper ===\n"

  echo helperVk
  echo ""

  # Shape check: the helper forward declaration must precede the first kernel
  # preamble in the generated GLSL (this is what breaks the old parse).
  let fwdIdx = helperVk.find("void helper(")
  let preIdx = helperVk.find("layout(local_size_x")
  doAssert fwdIdx >= 0, "expected a `void helper(` forward declaration in the GLSL"
  doAssert preIdx > fwdIdx,
    &"helper forward declaration (idx {fwdIdx}) must precede the kernel " &
    &"preamble (idx {preIdx})"

  var engine = bkVulkan.init()
  engine.ingest(helperVk)

  # Sentinel: a wrong entry point either fails ingest (glslang `-e helper`) or
  # silently recompiles per launch; correct output proves the fast path parsed
  # the kernel name.
  var gpuOut: array[1, uint32] = [0xDEADBEEF'u32]
  let oDev = engine.allocDevice(uint32, 1)
  engine.copyToDeviceFromHost(oDev, addr gpuOut[0], 1)
  engine.launchKernel("entryKernel", oDev, ())
  engine.copyToHostFromDevice(addr gpuOut[0], oDev, 1)
  engine.freeDevice(oDev)
  echo &"  entryKernel: got {gpuOut[0]}, expected 42"
  doAssert gpuOut[0] == 42, &"entryKernel: got {gpuOut[0]}, expected 42"

  echo "  OK: entry point parsed as the kernel name, fast path fires"

when isMainModule:
  runTest()
