# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Minimal Vulkan triad: one device buffer alloc, one upload, one GLSL kernel dispatch, one readback.
##
## Test that the k_add GLSL source compiles to SPIR-V, dispatches in one
## vkQueueSubmit and adds the uploaded operands element for element.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" --outdir:build/tests workspace/tensorcery/tests/t_vulkan_minimal.nim

import std/strutils
import
  workspace/crucible/src/runtime/backends/vulkan_common,
  workspace/crucible/src/runtime/backends/vulkan_compile {.all.},
  workspace/crucible/src/runtime/backends/vulkan_descriptor_runtime,
  workspace/crucible/src/abis/vulkan_abi as vk,
  workspace/tensorcery/src/kernels/k_add,
  workspace/tensorcery/src/backends/vulkan,
  workspace/tensorcery/src/kernels/k_vulkan

const
  vkStructureTypeMemoryBarrier = vk.VkStructureType(7)
  # VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
  stageCompute = 1'u32 shl 11
  # VK_ACCESS_SHADER_READ_BIT
  accessShaderRead = 1'u32 shl 5
  # VK_ACCESS_SHADER_WRITE_BIT
  accessShaderWrite = 1'u32 shl 6

type
  VkMemoryBarrier = object
    ## VkMemoryBarrier is absent from the vendored ABI. The C struct is
    ## {sType, pNext, srcAccessMask, dstAccessMask}.
    sType: vk.VkStructureType
    pNext: pointer
    srcAccessMask: uint32
    dstAccessMask: uint32

  PipeBundle = object
    ## Compute pipeline plus the set layout its descriptor sets bind against.
    handle: vk.VkPipeline
    layout: vk.VkPipelineLayout
    setLayout: vk.VkDescriptorSetLayout

proc load(ctx: VulkanContext, name: cstring): pointer =
  ## Loads one device-level Vulkan function through vkGetInstanceProcAddr.
  ctx.gpaAddr(ctx.instance, name)

proc buildAddPipeline(ctx: VulkanContext, setLayout: vk.VkDescriptorSetLayout): PipeBundle =
  ## Compiles the k_add GLSL source and builds the compute pipeline over 4 SSBO bindings.
  let spirv = compileGlslToSpirvViaGlslang(kAddGlsl % $workgroupSize, "k_add")
  let shaderModule = ctx.createShaderModule(spirv)
  let vkCreatePipelineLayout = cast[
    proc(device: vk.VkDevice, pCI: ptr vk.VkPipelineLayoutCreateInfo, pAlloc: pointer, pLayout: ptr vk.VkPipelineLayout): vk.VkResult {.cdecl.}
  ](ctx.load("vkCreatePipelineLayout"))
  let vkCreateComputePipelines = cast[
    proc(device: vk.VkDevice, cache: vk.VkPipelineCache, count: uint32, pCIs: ptr vk.VkComputePipelineCreateInfo, pAlloc: pointer, pPipelines: ptr vk.VkPipeline): vk.VkResult {.cdecl.}
  ](ctx.load("vkCreateComputePipelines"))
  var plCI = vk.VkPipelineLayoutCreateInfo(
    sType: vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    setLayoutCount: 1)
  plCI.pSetLayouts = setLayout.unsafeAddr
  var playout: vk.VkPipelineLayout
  check vkCreatePipelineLayout(ctx.device, plCI.addr, nil, playout.addr)
  var stageInfo = vk.VkPipelineShaderStageCreateInfo(
    sType: vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    stage: vk.VK_SHADER_STAGE_COMPUTE_BIT,
    module: shaderModule,
    pName: "k_add")
  var cpCI = vk.VkComputePipelineCreateInfo(
    sType: vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    stage: stageInfo,
    layout: playout)
  check vkCreateComputePipelines(ctx.device, ctx.pipelineCache, 1, cpCI.addr, nil, result.handle.addr)
  ctx.destroyShaderModule(shaderModule)
  result.layout = playout
  result.setLayout = setLayout

proc main() =
  var ctx = newVulkanContext()
  let setLayout = ctx.createDescriptorSetLayout(4)
  let pipeline = ctx.buildAddPipeline(setLayout)

  let n = 1024
  var outBuf, aBuf, bBuf, sBuf: VulkanBuffer
  outBuf = ctx.allocBuffer(n * sizeof(float32))
  aBuf = ctx.allocBuffer(n * sizeof(float32))
  bBuf = ctx.allocBuffer(n * sizeof(float32))
  sBuf = ctx.allocBuffer(sizeof(AddScalars))

  var a, b: seq[float32]
  newSeq(a, n)
  newSeq(b, n)
  for i in 0 ..< n:
    a[i] = float32(i)
    b[i] = 2.0
  ctx.writeBuffer(aBuf, a[0].addr, n * sizeof(float32))
  ctx.writeBuffer(bBuf, b[0].addr, n * sizeof(float32))

  var scalars: AddScalars
  scalars.aOff = 0
  scalars.bOff = 0
  scalars.shape = [int32(n), 1'i32, 1'i32, 1'i32]
  scalars.sa = [1'i32, 0, 0, 0]
  scalars.sb = [1'i32, 0, 0, 0]
  scalars.n = int32(n)
  ctx.writeBuffer(sBuf, scalars.addr, sizeof(AddScalars))

  let vkAllocateCommandBuffers = cast[
    proc(device: vk.VkDevice, pAllocateInfo: ptr vk.VkCommandBufferAllocateInfo, pCommandBuffers: ptr vk.VkCommandBuffer): vk.VkResult {.cdecl.}
  ](ctx.load("vkAllocateCommandBuffers"))
  let vkBeginCommandBuffer = cast[
    proc(cb: vk.VkCommandBuffer, pBeginInfo: ptr vk.VkCommandBufferBeginInfo): vk.VkResult {.cdecl.}
  ](ctx.load("vkBeginCommandBuffer"))
  let vkCmdBindPipeline = cast[
    proc(cb: vk.VkCommandBuffer, bindPoint: vk.VkPipelineBindPoint, pipeline: vk.VkPipeline) {.cdecl.}
  ](ctx.load("vkCmdBindPipeline"))
  let vkCmdBindDescriptorSets = cast[
    proc(cb: vk.VkCommandBuffer, bindPoint: vk.VkPipelineBindPoint, layout: vk.VkPipelineLayout, firstSet: uint32, setCount: uint32, pSets: ptr vk.VkDescriptorSet, dynOffCount: uint32, pDynOffs: ptr uint32) {.cdecl.}
  ](ctx.load("vkCmdBindDescriptorSets"))
  let vkCmdDispatch = cast[
    proc(cb: vk.VkCommandBuffer, gx: uint32, gy: uint32, gz: uint32) {.cdecl.}
  ](ctx.load("vkCmdDispatch"))
  let vkEndCommandBuffer = cast[
    proc(cb: vk.VkCommandBuffer): vk.VkResult {.cdecl.}
  ](ctx.load("vkEndCommandBuffer"))
  let vkCreateFence = cast[
    proc(device: vk.VkDevice, pCI: ptr vk.VkFenceCreateInfo, pAlloc: pointer, fence: ptr vk.VkFence): vk.VkResult {.cdecl.}
  ](ctx.load("vkCreateFence"))
  let vkQueueSubmit = cast[
    proc(queue: vk.VkQueue, submitCount: uint32, pSubmit: ptr vk.VkSubmitInfo, fence: vk.VkFence): vk.VkResult {.cdecl.}
  ](ctx.load("vkQueueSubmit"))
  let vkWaitForFences = cast[
    proc(device: vk.VkDevice, fenceCount: uint32, pFences: ptr vk.VkFence, waitAll: vk.VkBool32, timeout: uint64): vk.VkResult {.cdecl.}
  ](ctx.load("vkWaitForFences"))
  let vkDestroyFence = cast[
    proc(device: vk.VkDevice, fence: vk.VkFence, pAlloc: pointer) {.cdecl.}
  ](ctx.load("vkDestroyFence"))
  let vkFreeCommandBuffers = cast[
    proc(device: vk.VkDevice, pool: vk.VkCommandPool, count: uint32, pCBs: ptr vk.VkCommandBuffer) {.cdecl.}
  ](ctx.load("vkFreeCommandBuffers"))
  let vkUpdateDescriptorSets = cast[
    proc(device: vk.VkDevice, count: uint32, pWrites: ptr vk.VkWriteDescriptorSet,
         copyCount: uint32, pCopies: ptr vk.VkWriteDescriptorSet) {.cdecl.}
  ](ctx.load("vkUpdateDescriptorSets"))

  var cmdBuf: vk.VkCommandBuffer
  var allocInfo = vk.VkCommandBufferAllocateInfo(
    sType: vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    commandPool: ctx.commandPool,
    level: 0,
    commandBufferCount: 1)
  check vkAllocateCommandBuffers(ctx.device, allocInfo.addr, cmdBuf.addr)
  var beginInfo = vk.VkCommandBufferBeginInfo(
    sType: vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    flags: vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
  check vkBeginCommandBuffer(cmdBuf, beginInfo.addr)

  let pool = ctx.createDescriptorPool(4)
  let ds = ctx.allocateDescriptorSet(pool, setLayout)
  var infos: array[4, vk.VkDescriptorBufferInfo]
  var writes: array[4, vk.VkWriteDescriptorSet]
  let bufs = [outBuf.handle, aBuf.handle, bBuf.handle, sBuf.handle]
  for i, h in bufs:
    infos[i] = vk.VkDescriptorBufferInfo(
      buffer: h, offset: 0, range: vk.VkDeviceSize.high)  # VK_WHOLE_SIZE
    writes[i] = vk.VkWriteDescriptorSet(
      sType: vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      dstSet: ds, dstBinding: uint32(i), dstArrayElement: 0,
      descriptorCount: 1, descriptorType: vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      pBufferInfo: infos[i].addr)
  vkUpdateDescriptorSets(ctx.device, 4, writes[0].addr, 0, nil)

  vkCmdBindPipeline(cmdBuf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle)
  vkCmdBindDescriptorSets(cmdBuf, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline.layout, 0, 1, ds.addr, 0, nil)
  vkCmdDispatch(cmdBuf, uint32((n + 255) div 256), 1, 1)

  check vkEndCommandBuffer(cmdBuf)
  var fence: vk.VkFence
  var fci = vk.VkFenceCreateInfo(sType: vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
  check vkCreateFence(ctx.device, fci.addr, nil, fence.addr)
  var submitInfo = vk.VkSubmitInfo(
    sType: vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
    commandBufferCount: 1,
    pCommandBuffers: cmdBuf.addr)
  var submits = 0
  check vkQueueSubmit(ctx.queue, 1, submitInfo.addr, fence)
  submits.inc
  check vkWaitForFences(ctx.device, 1, fence.addr, vk.VK_TRUE, uint64.high)
  vkDestroyFence(ctx.device, fence, nil)
  vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, cmdBuf.addr)
  ctx.destroyDescriptorPool(pool)

  doAssert submits == 1, "the triad runs in one vkQueueSubmit, got " & $submits

  let res = readBuffer[byte](ctx, outBuf)
  var got: seq[float32]
  newSeq(got, n)
  copyMem(got[0].addr, res[0].unsafeAddr, n * sizeof(float32))
  for i in 0 ..< n:
    doAssert got[i] == float32(i) + 2.0,
        "element " & $i & " got " & $got[i] & ", expected " & $(float32(i) + 2.0)

main()
