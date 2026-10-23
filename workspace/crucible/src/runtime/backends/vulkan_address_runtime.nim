# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
## Address route: buffers reach the kernel as 64-bit buffer-device addresses packed into push constants, no descriptor set.
## The shader declares buffer_reference blocks, the GLSL printer output.

import workspace/crucible/src/runtime/backends/vulkan_common
import workspace/crucible/src/abis/vulkan_abi as vk

proc deviceAddress*(buf: VulkanBuffer, ctx: VulkanContext): uint64 =
  ## Shader-visible 64-bit address of a buffer, the value a pointer param carries.
  let vkGetBufferDeviceAddress = cast[
    proc(device: VkDevice, pInfo: ptr VkBufferDeviceAddressInfo): VkDeviceAddress {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkGetBufferDeviceAddress"))
  var info = VkBufferDeviceAddressInfo(
    sType: VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
    buffer: buf.handle)
  result = uint64(vkGetBufferDeviceAddress(ctx.device, info.addr))
