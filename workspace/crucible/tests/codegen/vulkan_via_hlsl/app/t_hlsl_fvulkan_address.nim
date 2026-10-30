## Vulkan: the fvVulkan HLSL flavor, zero-descriptor device-address route.
##
## Test that an `hlsl(fvVulkan)` kernel with buffer parameters produces
## validated SPIR-V through dxc and executes through the engine's address route
##
## - every parameter is a `[[vk::push_constant]]` block member, buffer
##   parameters hold `uint64_t` device addresses on locked member offsets
## - a buffer access constructs `vk::BufferPointer` from the address per use
## - readback [16, 27] for a = [10, 20], b = [1, 2], z = 5, byte-exact
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_fvulkan_address.nim

import std/strutils
import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const addVulkanHlsl = hlsl(fvVulkan):
  proc addKernel(output, a, b: ptr UncheckedArray[uint32], z: uint32) {.global.} =
    output[0] = a[0] + b[0] + z
    output[1] = a[1] + b[1] + z

proc runTest() =   # private: the engine is destroyed at return
  # fvVulkan binds no descriptor resources: every parameter sits in the push block,
  # buffers as device addresses, so the compile check needs no binding expectations.
  # Push-constant block plus BufferPointer accesses lock the flavor to the emitted text.
  doAssert "vk::BufferPointer" in addVulkanHlsl,
    "fvVulkan output has no BufferPointer access:\n" & addVulkanHlsl
  doAssert "[[vk::push_constant]]" in addVulkanHlsl,
    "fvVulkan output has no push-constant block:\n" & addVulkanHlsl
  doAssert "[[vk::offset(8)]] uint64_t a;" in addVulkanHlsl,
    "fvVulkan buffer members are not uint64_t on locked offsets:\n" & addVulkanHlsl
  let words = compileFvVulkanSpirV("fvvulkan_add", addVulkanHlsl)
  echo "  fvVulkan add: dxc -spirv + spirv-val ok, ", words.len, " words"

  # Engine launch on the address route: buffers go out as device addresses
  # in the push block, matching the packing that the fvVulkan offsets lock against.
  var engine = newEngine(runVulkan)
  engine.ingest(addVulkanHlsl)

  var a: array[2, uint32] = [10'u32, 20'u32]
  var b: array[2, uint32] = [1'u32, 2'u32]
  let aDev = engine.allocDevice(uint32, a.len)
  let bDev = engine.allocDevice(uint32, b.len)
  let oDev = engine.allocDevice(uint32, 2)
  engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
  engine.copyToDeviceFromHost(bDev, addr b[0], b.len)

  engine.launchKernel("addKernel", oDev, (aDev, bDev, 5'u32))

  var out32: array[2, uint32]
  engine.copyToHostFromDevice(addr out32[0], oDev, 2)
  doAssert out32[0] == 16, "slot 0 got " & $out32[0] & ", want 10 + 1 + 5 = 16"
  doAssert out32[1] == 27, "slot 1 got " & $out32[1] & ", want 20 + 2 + 5 = 27"
  echo "  fvVulkan address route: [", out32[0], ", ", out32[1], "] exact readback on MoltenVK"

when isMainModule:
  runTest()
  echo "t_hlsl_fvulkan_address: all checks passed"
