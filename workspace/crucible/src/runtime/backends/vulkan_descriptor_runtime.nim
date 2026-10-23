# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/crucible/src/runtime/backends/vulkan_common
import ../errors
import workspace/crucible/src/abis/vulkan_abi as vk

proc createDescriptorSetLayout*(ctx: VulkanContext, bindingCount: uint32, uniformCount: uint32 = 0): VkDescriptorSetLayout =
  ## Descriptor set layout for the descriptor route, compute stage only:
  ## - bindings 0 ..< bindingCount: STORAGE_BUFFER, one per buffer parameter
  ##   in declaration order, matching the printer's register(tN/uN, space0) numbering
  ## - bindings bindingCount ..< bindingCount + uniformCount: UNIFORM_BUFFER,
  ##   matching the printer's register(bN, space0) ordinals for the scalar
  ##   ConstantBuffer blocks, the first block sits after all buffers
  let vkCreateDescriptorSetLayout = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkDescriptorSetLayoutCreateInfo,
         pAllocator: pointer,
         pSetLayout: ptr VkDescriptorSetLayout): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreateDescriptorSetLayout"))
  var bindings = newSeq[VkDescriptorSetLayoutBinding](bindingCount.int + uniformCount.int)
  for i in 0 ..< bindingCount.int:
    bindings[i] = VkDescriptorSetLayoutBinding(
      binding: uint32(i),
      descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      descriptorCount: 1,
      stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
      pImmutableSamplers: nil
    )
  for i in bindingCount.int ..< bindingCount.int + uniformCount.int:
    bindings[i] = VkDescriptorSetLayoutBinding(
      binding: uint32(i),
      descriptorType: VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      descriptorCount: 1,
      stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
      pImmutableSamplers: nil
    )
  var slCI = VkDescriptorSetLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    bindingCount: bindingCount + uniformCount
  )
  if bindingCount + uniformCount > 0:
    slCI.pBindings = bindings[0].addr
  check vkCreateDescriptorSetLayout(ctx.device, slCI.addr, nil, result.addr)

proc destroyDescriptorSetLayout*(ctx: VulkanContext, setLayout: VkDescriptorSetLayout) =
  let vkDestroyDescriptorSetLayout = cast[
    proc(device: VkDevice, descriptorSetLayout: VkDescriptorSetLayout, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyDescriptorSetLayout"))
  if setLayout != nil:
    vkDestroyDescriptorSetLayout(ctx.device, setLayout, nil)

proc createDescriptorPool*(ctx: VulkanContext, bindingCount: uint32, uniformCount: uint32 = 0): VkDescriptorPool =
  ## Set lifetime is owned by the pool: destroyDescriptorPool frees the sets.
  let vkCreateDescriptorPool = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkDescriptorPoolCreateInfo,
         pAllocator: pointer,
         pDescriptorPool: ptr VkDescriptorPool): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreateDescriptorPool"))
  var poolSizes = [
    VkDescriptorPoolSize(
      `type`: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      descriptorCount: bindingCount
    ),
    VkDescriptorPoolSize(
      `type`: VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      descriptorCount: uniformCount
    )
  ]
  var poolCI = VkDescriptorPoolCreateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    maxSets: 1,
    poolSizeCount: 2,
    pPoolSizes: poolSizes[0].addr
  )
  check vkCreateDescriptorPool(ctx.device, poolCI.addr, nil, result.addr)

proc destroyDescriptorPool*(ctx: VulkanContext, pool: VkDescriptorPool) =
  let vkDestroyDescriptorPool = cast[
    proc(device: VkDevice, descriptorPool: VkDescriptorPool, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyDescriptorPool"))
  if pool != nil:
    vkDestroyDescriptorPool(ctx.device, pool, nil)

proc allocateDescriptorSet*(ctx: VulkanContext, pool: VkDescriptorPool, setLayout: VkDescriptorSetLayout): VkDescriptorSet =
  let vkAllocateDescriptorSets = cast[
    proc(device: VkDevice, pAllocateInfo: ptr VkDescriptorSetAllocateInfo, pDescriptorSets: ptr VkDescriptorSet): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkAllocateDescriptorSets"))
  var allocInfo = VkDescriptorSetAllocateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    descriptorPool: pool,
    descriptorSetCount: 1,
    pSetLayouts: setLayout.addr
  )
  check vkAllocateDescriptorSets(ctx.device, allocInfo.addr, result.addr)

proc updateDescriptorSet*(ctx: VulkanContext, setHandle: VkDescriptorSet, buffers: openArray[VulkanBuffer], bindingCount: uint32, uniforms: openArray[VulkanBuffer] = []) =
  ## One write per binding: binding i covers the whole i-th buffer, the uniform buffers
  ## write bindings bindingCount ..< bindingCount + uniforms.len.
  ##
  ## Precondition:
  ## - `buffers` enumerates the kernel's buffer parameters in declaration order,
  ##   binding i writes the i-th buffer, matching the printer's register(tN/uN, space0) numbering
  ## - `uniforms` enumerates the kernel's scalar ConstantBuffer blocks in declaration order,
  ##   binding bindingCount + j writes the j-th block, matching the printer's
  ##   register(bN, space0) ordinals, the first block after all buffers
  ## - `bindingCount` is the set layout's storage-binding count from createDescriptorSetLayout:
  ##   fewer buffers leaves bindings unwritten (reads hit stale device state),
  ##   more writes bindings the layout does not declare
  ##
  ## A mismatched order is silent on the device: reads and writes reach the wrong buffers, no validation error.
  if buffers.len != bindingCount.int:
    failLoud("Vulkan updateDescriptorSet: " & $buffers.len & " buffers for a set layout of " &
      $bindingCount & " bindings, unwritten or undeclared bindings read stale device state")
  let vkUpdateDescriptorSets = cast[
    proc(device: VkDevice, descriptorWriteCount: uint32,
         pDescriptorWrites: ptr VkWriteDescriptorSet,
         descriptorCopyCount: uint32,
         pDescriptorCopies: ptr VkWriteDescriptorSet) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkUpdateDescriptorSets"))
  var bufInfos = newSeq[VkDescriptorBufferInfo](buffers.len + uniforms.len)
  var writes = newSeq[VkWriteDescriptorSet](buffers.len + uniforms.len)
  for i, buf in buffers:
    bufInfos[i] = VkDescriptorBufferInfo(
      buffer: buf.handle,
      offset: 0,
      range: VkDeviceSize(buf.size)
    )
    writes[i] = VkWriteDescriptorSet(
      sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      dstSet: setHandle,
      dstBinding: uint32(i),
      dstArrayElement: 0,
      descriptorCount: 1,
      descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      pBufferInfo: bufInfos[i].addr
    )
  for j, buf in uniforms:
    let i = buffers.len + j
    bufInfos[i] = VkDescriptorBufferInfo(
      buffer: buf.handle,
      offset: 0,
      range: VkDeviceSize(buf.size)
    )
    writes[i] = VkWriteDescriptorSet(
      sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      dstSet: setHandle,
      dstBinding: uint32(bindingCount.int + j),
      dstArrayElement: 0,
      descriptorCount: 1,
      descriptorType: VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      pBufferInfo: bufInfos[i].addr
    )
  if writes.len > 0:
    vkUpdateDescriptorSets(ctx.device, uint32(writes.len), writes[0].addr, 0, nil)
