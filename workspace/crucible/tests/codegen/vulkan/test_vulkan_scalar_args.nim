## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan: by-value (scalar) kernel params as push constants — anti-regression.
##
## Codegen emits scalar params as `layout(push_constant) uniform KernelParams`
## (see test_vulkan_kernel_pushconst.nim, codegen-only). The runtime must
## populate that push-constant range and only bind pointer args as SSBOs.
##
## Failure mode this test prevents: a runtime that skips vkCmdPushConstants and
## allocates an SSBO fallback for the scalar leaves the shader's push-constant
## block unwritten and shifts the output off binding 0, so the kernel returns the
## output's pre-initialized bytes with no error. The runtime must pack by-value
## scalars into the push-constant range and bind only pointer args as SSBOs,
## with the output at binding 0.

import std/strformat
import workspace/crucible

# One kernel per source: the codegen emits a single file-scope KernelParams
# push-constant block per source = the union of every kernel's by-value params.
# Mixing kernels with different scalar signatures in one source misaligns the
# block (kernel 2's params land after kernel 1's) — a codegen contract caveat;
# one kernel per ingest is the supported pattern.
const code1 = vulkan:
  proc kernelWithVal(output: ptr UncheckedArray[uint32];
                     val: uint32) {.global.} =
    output[0] = val

const code2 = vulkan:
  proc kernelWithTwoVals(output: ptr UncheckedArray[uint32];
                         a: uint32; b: uint32) {.global.} =
    output[0] = a + b

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo "=== Vulkan: scalar by-value param via push constants ===\n"

  var engine = bkVulkan.init()
  engine.ingest(code1)

  # Sentinel: with the misbinding regression the kernel never writes this
  # buffer and the runtime reads it back unchanged — silent wrong result.
  var gpuOut: array[1, uint32] = [0xDEADBEEF'u32]
  engine.run("kernelWithVal", gpuOut, (7'u32,))
  doAssert gpuOut[0] == 7, &"push-constant scalar: got {gpuOut[0]:#x}, expected 7"

  echo "  OK — scalar by-value param via push constants"

  # Re-ingest (RAII field replacement) + two scalars: proves packing order
  # matches the KernelParams block member order.
  engine.ingest(code2)
  engine.run("kernelWithTwoVals", gpuOut, (3'u32, 4'u32,))
  doAssert gpuOut[0] == 7, &"two push-constant scalars: got {gpuOut[0]}, expected 7"

  echo "  OK — two scalar by-value params (packing order, re-ingest)"

when isMainModule:
  runTest()
