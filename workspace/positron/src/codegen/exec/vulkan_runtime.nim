## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Vulkan compute execution DSL.
##
## Provides a high-level wrapper over the Vulkan compute API.
## SPIR-V compilation via glslangValidator.
## ICD loaded directly (bypasses Vulkan loader) to work around NVIDIA 595 + loader 1.4.
##
## Usage:
##   import workspace/positron/src/codegen/vk
##   const code = vulkan:
##     proc add(a: ptr UncheckedArray[uint32];
##              b: ptr UncheckedArray[uint32];
##              output: ptr UncheckedArray[uint32]) {.global.} =
##       output[0] = a[0] + b[0]
##   var ctx = initVulkan()
##   let res = execVulkan(ctx, code, "main", 8,
##                        inputs = [([10'u32, 20'u32], 8u)])
##   ctx.shutdown()

import std/[os, osproc, strutils, sequtils]
import std/dynlib
import workspace/positron/src/abis/vulkan_abi as vk
import workspace/positron/src/abis/shaderc_abi
type
  VulkanError* = ref object of CatchableError

  VulkanBuffer* = object
    size*: int
    handle: VkBuffer
    memory: VkDeviceMemory
    device: VkDevice

  VulkanPipeline* = object
    device: VkDevice
    handle: VkPipeline
    layout: VkPipelineLayout
    descriptorSetLayout: VkDescriptorSetLayout
    descriptorPool: VkDescriptorPool
    descriptorSet: VkDescriptorSet
    ssboCount: int

  VulkanContext* = object
    instance: VkInstance
    physicalDevice: VkPhysicalDevice
    device: VkDevice
    queue: VkQueue
    queueFamilyIndex: uint32
    commandPool: VkCommandPool
    pipelineCache: VkPipelineCache
    vkLib: LibHandle
    getProcAddr*: pointer  ## vkGetInstanceProcAddr

proc check(res: VkResult) =
  if res != VK_SUCCESS:
    raise VulkanError(msg: "Vulkan error: " & $res)

proc gpaAddr*(ctx: VulkanContext, instance: VkInstance, name: cstring): pointer =
  ## Call vkGetInstanceProcAddr to load a Vulkan function pointer.
  ## Pass `nil` for `instance` to get instance-level functions
  ## before creating the instance.
  let fn = cast[
    proc(instance: VkInstance, name: cstring): pointer {.cdecl.}
  ](ctx.getProcAddr)
  if fn == nil: return nil
  fn(instance, name)
# ═══════════════════════════════════════════════════════════════════════
# Vulkan loader — dlopen libvulkan.so.1 (like ash does)
# ═══════════════════════════════════════════════════════════════════════
# We load the Vulkan loader (libvulkan.so.1), get vkGetInstanceProcAddr,
# and use that to load all other Vulkan functions. The loader handles
# ICD discovery, interface negotiation, and dispatch internally.
# This matches the approach used by ash, vulkano, and all standard
# Vulkan applications.

proc loadVulkanLoader(): tuple[lib: LibHandle, gpa: pointer] =
  result.lib = loadLib("libvulkan.so.1")
  if result.lib == nil:
    raise VulkanError(msg: "Cannot load libvulkan.so.1 — install a Vulkan loader (e.g. libvulkan1)")
  result.gpa = result.lib.symAddr("vkGetInstanceProcAddr")
  if result.gpa == nil:
    raise VulkanError(msg: "libvulkan.so.1 missing vkGetInstanceProcAddr")

# ═══════════════════════════════════════════════════════════════════════
# SPIR-V compilation
# ═══════════════════════════════════════════════════════════════════════

proc compileGlslToSpirV*(glsl: string): seq[uint32] =
  let tmpDir = getTempDir()
  let srcPath = tmpDir / "vk_shader_comp" & ".comp"
  let spvPath = tmpDir / "vk_shader_comp" & ".spv"
  try:
    writeFile(srcPath, glsl)
    let exitCode = execCmd("glslangValidator -V -o " & spvPath & " " & srcPath)
    if exitCode != 0:
      raise VulkanError(msg: "glslangValidator failed: exit=" & $exitCode)
    let raw = readFile(spvPath)
    result = newSeq[uint32](raw.len div 4)
    copyMem(result[0].addr, raw[0].addr, raw.len)
  finally:
    removeFile(srcPath)
    removeFile(spvPath)

# ═══════════════════════════════════════════════════════════════════════
# Vulkan initialization
# ═══════════════════════════════════════════════════════════════════════

proc initVulkan*(): VulkanContext =
  let (lib, gpaRaw) = loadVulkanLoader()
  result.vkLib = lib
  result.getProcAddr = gpaRaw

  let gpaFn = cast[
    proc(instance: VkInstance, pName: cstring): pointer {.cdecl.}
  ](gpaRaw)

  # Load vkCreateInstance via GPA(NULL, name) — pre-instance, same as ash
  let vkCreateInstance = cast[
    proc(pCreateInfo: ptr VkInstanceCreateInfo,
         pAllocator: pointer,
         pInstance: ptr VkInstance): VkResult {.cdecl.}
  ](gpaFn(nil, "vkCreateInstance"))
  if vkCreateInstance == nil:
    raise VulkanError(msg: "Cannot load vkCreateInstance")

  var appInfo = VkApplicationInfo(
    sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pApplicationName: "PositronVk",
    applicationVersion: 1,
    pEngineName: "Positron",
    engineVersion: 1,
    apiVersion: 0x00403000
  )
  var instCI = VkInstanceCreateInfo(
    sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pApplicationInfo: appInfo.addr
  )
  check vkCreateInstance(instCI.addr, nil, result.instance.addr)

  # Helper: load instance-level functions via GPA(instance, name)
  template gpa(name: cstring): pointer = gpaFn(result.instance, name)

  # Enumerate physical devices
  var ep = cast[
    proc(instance: VkInstance, pPhysicalDeviceCount: ptr uint32,
         pPhysicalDevices: ptr VkPhysicalDevice): VkResult {.cdecl.}
  ](gpa("vkEnumeratePhysicalDevices"))
  var devCount: uint32 = 0
  check ep(result.instance, devCount.addr, nil)
  if devCount == 0:
    raise VulkanError(msg: "No Vulkan devices found")
  var devices = newSeq[VkPhysicalDevice](devCount.int)
  check ep(result.instance, devCount.addr, devices[0].addr)
  result.physicalDevice = devices[0]

  # Queue families
  var gpqfp = cast[
    proc(physicalDevice: VkPhysicalDevice,
         pQueueFamilyPropertyCount: ptr uint32,
         pQueueFamilyProperties: pointer) {.cdecl.}
  ](gpa("vkGetPhysicalDeviceQueueFamilyProperties"))
  var qfCount: uint32 = 0
  gpqfp(result.physicalDevice, qfCount.addr, nil)
  if qfCount == 0:
    raise VulkanError(msg: "No queue families found")
  var qfProps = newSeq[VkQueueFamilyProperties](qfCount.int)
  gpqfp(result.physicalDevice, qfCount.addr, qfProps[0].addr)
  result.queueFamilyIndex = 0
  for i, f in qfProps:
    if f.queueCount > 0 and (f.queueFlags and VK_QUEUE_COMPUTE_BIT.uint32) != 0:
      result.queueFamilyIndex = i.uint32
      break

  # Create device
  var cd = cast[
    proc(physicalDevice: VkPhysicalDevice,
         pCreateInfo: ptr VkDeviceCreateInfo,
         pAllocator: pointer,
         pDevice: ptr VkDevice): VkResult {.cdecl.}
  ](gpa("vkCreateDevice"))
  var qprio: cfloat = 1.0
  var qci = VkDeviceQueueCreateInfo(
    sType: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    queueFamilyIndex: result.queueFamilyIndex,
    queueCount: 1,
    pQueuePriorities: qprio.addr
  )
  var dci = VkDeviceCreateInfo(
    sType: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    queueCreateInfoCount: 1,
    pQueueCreateInfos: qci.addr
  )
  check cd(result.physicalDevice, dci.addr, nil, result.device.addr)

  # Get compute queue
  var gdq = cast[
    proc(device: VkDevice, queueFamilyIndex: uint32,
         queueIndex: uint32, pQueue: ptr VkQueue) {.cdecl.}
  ](gpa("vkGetDeviceQueue"))
  gdq(result.device, result.queueFamilyIndex, 0, result.queue.addr)

  # Command pool
  var ccpool = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkCommandPoolCreateInfo,
         pAllocator: pointer,
         pCommandPool: ptr VkCommandPool): VkResult {.cdecl.}
  ](gpa("vkCreateCommandPool"))
  var cpc = VkCommandPoolCreateInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    queueFamilyIndex: result.queueFamilyIndex
  )
  check ccpool(result.device, cpc.addr, nil, result.commandPool.addr)

  # Pipeline cache
  var cpcache = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkPipelineCacheCreateInfo,
         pAllocator: pointer,
         pPipelineCache: ptr VkPipelineCache): VkResult {.cdecl.}
  ](gpa("vkCreatePipelineCache"))
  var pcc = VkPipelineCacheCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO
  )
  check cpcache(result.device, pcc.addr, nil, result.pipelineCache.addr)

proc shutdown*(ctx: var VulkanContext) =
  let L = ctx.vkLib
  let vkDestroyPipelineCachePtr = cast[
    proc(device: VkDevice, pipelineCache: VkPipelineCache,
         pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyPipelineCache"))
  let vkDestroyCommandPoolPtr = cast[
    proc(device: VkDevice, commandPool: VkCommandPool,
         pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyCommandPool"))
  let vkDestroyDevicePtr = cast[
    proc(device: VkDevice, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyDevice"))
  let vkDestroyInstancePtr = cast[
    proc(instance: VkInstance, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyInstance"))
  if ctx.pipelineCache != nil and vkDestroyPipelineCachePtr != nil:
    vkDestroyPipelineCachePtr(ctx.device, ctx.pipelineCache, nil)
  if ctx.commandPool != nil and vkDestroyCommandPoolPtr != nil:
    vkDestroyCommandPoolPtr(ctx.device, ctx.commandPool, nil)
  if ctx.device != nil and vkDestroyDevicePtr != nil:
    vkDestroyDevicePtr(ctx.device, nil)
  if ctx.instance != nil and vkDestroyInstancePtr != nil:
    vkDestroyInstancePtr(ctx.instance, nil)

# ═══════════════════════════════════════════════════════════════════════
# Buffer management
# ═══════════════════════════════════════════════════════════════════════

proc allocBuffer*(ctx: var VulkanContext, size: int): VulkanBuffer =
  result.size = size
  result.device = ctx.device

  let vkCreateBuffer = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkBufferCreateInfo,
         pAllocator: pointer, pBuffer: ptr VkBuffer): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreateBuffer"))
  let vkGetBufferMemoryRequirements = cast[
    proc(device: VkDevice, buffer: VkBuffer,
         pMemoryRequirements: ptr VkMemoryRequirements) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkGetBufferMemoryRequirements"))
  let vkAllocateMemory = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkMemoryAllocateInfo,
         pAllocator: pointer, pMemory: ptr VkDeviceMemory): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkAllocateMemory"))
  let vkBindBufferMemory = cast[
    proc(device: VkDevice, buffer: VkBuffer, memory: VkDeviceMemory,
         memoryOffset: VkDeviceSize): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkBindBufferMemory"))

  var bci = VkBufferCreateInfo(
    sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    size: VkDeviceSize(size),
    usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
           VK_BUFFER_USAGE_TRANSFER_SRC_BIT or
           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    sharingMode: VK_SHARING_MODE_EXCLUSIVE
  )
  check vkCreateBuffer(ctx.device, bci.addr, nil, result.handle.addr)

  var memReq: VkMemoryRequirements
  vkGetBufferMemoryRequirements(ctx.device, result.handle, memReq.addr)

  let vkGetPhysicalDeviceMemoryProperties = cast[
    proc(physicalDevice: VkPhysicalDevice,
         pMemoryProperties: ptr VkPhysicalDeviceMemoryProperties) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkGetPhysicalDeviceMemoryProperties"))
  var memProps: VkPhysicalDeviceMemoryProperties
  vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, memProps.addr)

  # Find memory type — inline loop, no closure (avoids Nim capture issues).
  # Preference:
  #  1. DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (integrated/ReBAR)
  #  2. HOST_VISIBLE | HOST_COHERENT (discrete via PCIe BAR)
  #  3. DEVICE_LOCAL only (VRAM)
  #  4. Any compatible (last resort)
  var memTypeIdx = high(uint32)
  for i in 0'u32 ..< memProps.memoryTypeCount:
    if (memReq.memoryTypeBits and (1'u32 shl i)) == 0:
      continue
    if memTypeIdx == high(uint32):
      memTypeIdx = i  # fallback: any compatible
    let flags = cast[uint32](memProps.memoryTypes[i].propertyFlags)
    let best = cast[uint32](
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT or
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    if (flags and best) == best:
      memTypeIdx = i; break
    let std = cast[uint32](
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    if (flags and std) == std:
      memTypeIdx = i
  if memTypeIdx == high(uint32):
    var details = "memoryTypeBits=" & $memReq.memoryTypeBits & " memTypeCount=" & $memProps.memoryTypeCount
    for i in 0'u32 ..< memProps.memoryTypeCount:
      details &= " [" & $i & ": flags=" & $(cast[int](memProps.memoryTypes[i].propertyFlags)) & " heap=" & $memProps.memoryTypes[i].heapIndex & "]"
    raise VulkanError(msg: "No suitable memory type found — " & details)

  var allocInfo = VkMemoryAllocateInfo(
    sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    allocationSize: memReq.size,
    memoryTypeIndex: memTypeIdx
  )
  check vkAllocateMemory(ctx.device, allocInfo.addr, nil, result.memory.addr)
  check vkBindBufferMemory(ctx.device, result.handle, result.memory, 0)

proc dealloc*(buffer: var VulkanBuffer) =
  if buffer.handle != nil:
    discard

proc writeBuffer*(ctx: VulkanContext, buf: var VulkanBuffer, data: pointer, size: int) =
  if size > buf.size:
    raise VulkanError(msg: "writeBuffer overflow")
  let vkMapMemory = cast[
    proc(device: VkDevice, memory: VkDeviceMemory,
         offset: VkDeviceSize, size: VkDeviceSize,
         flags: VkFlags, ppData: ptr pointer): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkMapMemory"))
  let vkUnmapMemory = cast[
    proc(device: VkDevice, memory: VkDeviceMemory) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkUnmapMemory"))
  var mapped: pointer
  check vkMapMemory(buf.device, buf.memory, 0, VkDeviceSize(size), 0, mapped.addr)
  copyMem(mapped, data, size)
  vkUnmapMemory(buf.device, buf.memory)

proc readBuffer*[T](ctx: VulkanContext, buf: VulkanBuffer): seq[T] =
  if buf.size mod sizeof(T) != 0:
    raise VulkanError(msg: "Buffer size not divisible by type size")
  if buf.size > 0:
    let vkMapMemory = cast[
      proc(device: VkDevice, memory: VkDeviceMemory,
           offset: VkDeviceSize, size: VkDeviceSize,
           flags: VkFlags, ppData: ptr pointer): VkResult {.cdecl.}
    ](ctx.gpaAddr(ctx.instance, "vkMapMemory"))
    let vkUnmapMemory = cast[
      proc(device: VkDevice, memory: VkDeviceMemory) {.cdecl.}
    ](ctx.gpaAddr(ctx.instance, "vkUnmapMemory"))
    result = newSeq[T](buf.size div sizeof(T))
    var mapped: pointer
    check vkMapMemory(buf.device, buf.memory, 0, VkDeviceSize(buf.size), 0, mapped.addr)
    copyMem(result[0].addr, mapped, buf.size)
    vkUnmapMemory(buf.device, buf.memory)

# ═══════════════════════════════════════════════════════════════════════
# Pipeline management
# ═══════════════════════════════════════════════════════════════════════

proc createPipeline*(ctx: var VulkanContext, shaderModule: VkShaderModule,
                     ssboCount: int): VulkanPipeline =
  result.device = ctx.device
  result.ssboCount = ssboCount

  let vkCreateDescriptorSetLayout = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkDescriptorSetLayoutCreateInfo,
         pAllocator: pointer,
         pSetLayout: ptr VkDescriptorSetLayout): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreateDescriptorSetLayout"))
  let vkCreatePipelineLayout = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkPipelineLayoutCreateInfo,
         pAllocator: pointer,
         pPipelineLayout: ptr VkPipelineLayout): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreatePipelineLayout"))
  let vkCreateDescriptorPool = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkDescriptorPoolCreateInfo,
         pAllocator: pointer,
         pDescriptorPool: ptr VkDescriptorPool): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreateDescriptorPool"))
  let vkAllocateDescriptorSets = cast[
    proc(device: VkDevice, pAllocateInfo: ptr VkDescriptorSetAllocateInfo,
         pDescriptorSets: ptr VkDescriptorSet): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkAllocateDescriptorSets"))
  let vkCreateComputePipelines = cast[
    proc(device: VkDevice, pipelineCache: VkPipelineCache,
         createInfoCount: uint32,
         pCreateInfos: ptr VkComputePipelineCreateInfo,
         pAllocator: pointer,
         pPipelines: ptr VkPipeline): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreateComputePipelines"))

  var bindings = newSeq[VkDescriptorSetLayoutBinding](ssboCount)
  for i in 0 ..< ssboCount:
    bindings[i] = VkDescriptorSetLayoutBinding(
      binding: uint32(i),
      descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      descriptorCount: 1,
      stageFlags: VK_SHADER_STAGE_COMPUTE_BIT
    )
  var dslCI = VkDescriptorSetLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    bindingCount: uint32(ssboCount),
    pBindings: bindings[0].addr
  )
  check vkCreateDescriptorSetLayout(ctx.device, dslCI.addr, nil, result.descriptorSetLayout.addr)

  var plCI = VkPipelineLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    setLayoutCount: 1,
    pSetLayouts: result.descriptorSetLayout.addr
  )
  check vkCreatePipelineLayout(ctx.device, plCI.addr, nil, result.layout.addr)

  var poolSize = VkDescriptorPoolSize(
    `type`: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    descriptorCount: uint32(ssboCount)
  )
  var dpCI = VkDescriptorPoolCreateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    maxSets: 1,
    poolSizeCount: 1,
    pPoolSizes: poolSize.addr
  )
  check vkCreateDescriptorPool(ctx.device, dpCI.addr, nil, result.descriptorPool.addr)

  var dsAI = VkDescriptorSetAllocateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    descriptorPool: result.descriptorPool,
    descriptorSetCount: 1,
    pSetLayouts: result.descriptorSetLayout.addr
  )
  check vkAllocateDescriptorSets(ctx.device, dsAI.addr, result.descriptorSet.addr)

  var stageInfo = VkPipelineShaderStageCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    stage: VK_SHADER_STAGE_COMPUTE_BIT,
    module: shaderModule,
    pName: "main"
  )
  var cpCI = VkComputePipelineCreateInfo(
    sType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    stage: stageInfo,
    layout: result.layout
  )
  check vkCreateComputePipelines(ctx.device, ctx.pipelineCache, 1, cpCI.addr, nil, result.handle.addr)

proc destroyPipeline*(pipeline: var VulkanPipeline, ctx: VulkanContext) =
  let vkDestroyPipeline = cast[
    proc(device: VkDevice, pipeline: VkPipeline, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyPipeline"))
  let vkDestroyPipelineLayout = cast[
    proc(device: VkDevice, pipelineLayout: VkPipelineLayout,
         pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyPipelineLayout"))
  let vkDestroyDescriptorSetLayout = cast[
    proc(device: VkDevice, descriptorSetLayout: VkDescriptorSetLayout,
         pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyDescriptorSetLayout"))
  let vkDestroyDescriptorPool = cast[
    proc(device: VkDevice, descriptorPool: VkDescriptorPool,
         pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyDescriptorPool"))
  if pipeline.handle != nil and vkDestroyPipeline != nil:
    vkDestroyPipeline(pipeline.device, pipeline.handle, nil)
  if pipeline.layout != nil and vkDestroyPipelineLayout != nil:
    vkDestroyPipelineLayout(pipeline.device, pipeline.layout, nil)
  if pipeline.descriptorSetLayout != nil and vkDestroyDescriptorSetLayout != nil:
    vkDestroyDescriptorSetLayout(pipeline.device, pipeline.descriptorSetLayout, nil)
  if pipeline.descriptorPool != nil and vkDestroyDescriptorPool != nil:
    vkDestroyDescriptorPool(pipeline.device, pipeline.descriptorPool, nil)

proc setArg*(pipeline: var VulkanPipeline, index: int, buf: VulkanBuffer,
            ctx: VulkanContext) =
  let vkUpdateDescriptorSets = cast[
    proc(device: VkDevice, descriptorWriteCount: uint32,
         pDescriptorWrites: ptr VkWriteDescriptorSet,
         descriptorCopyCount: uint32,
         pDescriptorCopies: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkUpdateDescriptorSets"))
  var binfo = VkDescriptorBufferInfo(
    buffer: buf.handle,
    offset: 0,
    range: VkDeviceSize(buf.size)
  )
  # Workaround removed — the real bug was VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
  # having the wrong enum value (2 = SAMPLED_IMAGE) in the Nim Vulkan bindings.
  # The correct value is 7 per the Vulkan spec.
  var wd = VkWriteDescriptorSet(
    sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    pNext: nil,
    dstSet: pipeline.descriptorSet,
    dstBinding: uint32(index),
    dstArrayElement: 0,
    descriptorCount: 1,
    descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    pImageInfo: nil,
    pBufferInfo: binfo.addr,
    pTexelBufferView: nil
  )
  vkUpdateDescriptorSets(pipeline.device, 1, wd.addr, 0, nil)

# ═══════════════════════════════════════════════════════════════════════
# Kernel execution
# ═══════════════════════════════════════════════════════════════════════

proc runKernel*(ctx: VulkanContext, pipeline: VulkanPipeline,
                globalWorkSize, localWorkSize: openArray[uint32]) =
  let vkAllocateCommandBuffers = cast[
    proc(device: VkDevice, pAllocateInfo: ptr VkCommandBufferAllocateInfo,
         pCommandBuffers: ptr VkCommandBuffer): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkAllocateCommandBuffers"))
  let vkBeginCommandBuffer = cast[
    proc(commandBuffer: VkCommandBuffer,
         pBeginInfo: ptr VkCommandBufferBeginInfo): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkBeginCommandBuffer"))
  let vkCmdBindPipeline = cast[
    proc(commandBuffer: VkCommandBuffer,
         pipelineBindPoint: VkPipelineBindPoint,
         pipeline: VkPipeline) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCmdBindPipeline"))
  let vkCmdBindDescriptorSets = cast[
    proc(commandBuffer: VkCommandBuffer,
         pipelineBindPoint: VkPipelineBindPoint,
         layout: VkPipelineLayout,
         firstSet: uint32,
         descriptorSetCount: uint32,
         pDescriptorSets: ptr VkDescriptorSet,
         dynamicOffsetCount: uint32,
         pDynamicOffsets: ptr uint32) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCmdBindDescriptorSets"))
  let vkCmdDispatch = cast[
    proc(commandBuffer: VkCommandBuffer,
         groupCountX: uint32, groupCountY: uint32,
         groupCountZ: uint32) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCmdDispatch"))
  let vkEndCommandBuffer = cast[
    proc(commandBuffer: VkCommandBuffer): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkEndCommandBuffer"))
  let vkCreateFence = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkFenceCreateInfo,
         pAllocator: pointer, pFence: ptr VkFence): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreateFence"))
  let vkQueueSubmit = cast[
    proc(queue: VkQueue, submitCount: uint32,
         pSubmits: ptr VkSubmitInfo,
         fence: VkFence): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkQueueSubmit"))
  let vkWaitForFences = cast[
    proc(device: VkDevice, fenceCount: uint32,
         pFences: ptr VkFence,
         waitAll: VkBool32,
         timeout: uint64): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkWaitForFences"))
  let vkDestroyFence = cast[
    proc(device: VkDevice, fence: VkFence, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyFence"))
  let vkFreeCommandBuffers = cast[
    proc(device: VkDevice, commandPool: VkCommandPool,
         commandBufferCount: uint32,
         pCommandBuffers: ptr VkCommandBuffer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkFreeCommandBuffers"))

  var allocInfo = VkCommandBufferAllocateInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    commandPool: ctx.commandPool,
    level: 0,
    commandBufferCount: 1
  )
  var cmdBuf: VkCommandBuffer
  check vkAllocateCommandBuffers(ctx.device, allocInfo.addr, cmdBuf.addr)

  var beginInfo = VkCommandBufferBeginInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
  )
  check vkBeginCommandBuffer(cmdBuf, beginInfo.addr)

  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle)
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout,
                          0, 1, pipeline.descriptorSet.addr, 0, nil)

  var gx = 1'u32; var gy = 1'u32; var gz = 1'u32
  var lx = 1'u32; var ly = 1'u32; var lz = 1'u32
  if globalWorkSize.len > 0: gx = globalWorkSize[0]
  if globalWorkSize.len > 1: gy = globalWorkSize[1]
  if globalWorkSize.len > 2: gz = globalWorkSize[2]
  if localWorkSize.len > 0: lx = localWorkSize[0]
  if localWorkSize.len > 1: ly = localWorkSize[1]
  if localWorkSize.len > 2: lz = localWorkSize[2]

  let gcx = (gx + lx - 1) div lx
  let gcy = (gy + ly - 1) div ly
  let gcz = (gz + lz - 1) div lz
  vkCmdDispatch(cmdBuf, gcx, gcy, gcz)

  check vkEndCommandBuffer(cmdBuf)

  var fence: VkFence
  var fci = VkFenceCreateInfo(sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
  check vkCreateFence(ctx.device, fci.addr, nil, fence.addr)

  var submitInfo = VkSubmitInfo(
    sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
    commandBufferCount: 1,
    pCommandBuffers: cmdBuf.addr
  )
  check vkQueueSubmit(ctx.queue, 1, submitInfo.addr, fence)
  check vkWaitForFences(ctx.device, 1, fence.addr, VK_TRUE, uint64.high)
  vkDestroyFence(ctx.device, fence, nil)
  vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, cmdBuf.addr)

# ═══════════════════════════════════════════════════════════════════════
# High-level helper: execVulkan
# ═══════════════════════════════════════════════════════════════════════

proc execVulkan*(
  ctx: var VulkanContext,
  source: string,
  entryPoint: string,
  outputBytes: int,
  inputs: openArray[tuple[data: pointer, size: int]]
): seq[byte] =
  let numInputs = inputs.len
  let totalSsboCount = numInputs + 1
  let spirv = compileGlslToSpirV(source)
  # Replace inline gpa calls with ctx.gpaAddr(ctx.instance, ...) below
  let vkCreateShaderModule = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkShaderModuleCreateInfo,
         pAllocator: pointer,
         pShaderModule: ptr VkShaderModule): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreateShaderModule"))
  var smCI = VkShaderModuleCreateInfo(
    sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    codeSize: csize_t(spirv.len * sizeof(uint32)),
    pCode: spirv[0].addr
  )
  var shaderModule: VkShaderModule
  check vkCreateShaderModule(ctx.device, smCI.addr, nil, shaderModule.addr)

  var inputBuffers = newSeq[VulkanBuffer](numInputs)
  for i in 0 ..< numInputs:
    inputBuffers[i] = ctx.allocBuffer(inputs[i].size)
    ctx.writeBuffer(inputBuffers[i], inputs[i].data, inputs[i].size)
  var outBuf = ctx.allocBuffer(outputBytes)

  var pipeline = ctx.createPipeline(shaderModule, totalSsboCount)
  for i in 0 ..< numInputs:
    pipeline.setArg(i, inputBuffers[i], ctx)
  pipeline.setArg(numInputs, outBuf, ctx)

  let wgs: uint32 = 256
  let nwg = ((outputBytes div 4).uint32 + wgs - 1) div wgs
  ctx.runKernel(pipeline, [nwg * wgs], [wgs, 1'u32, 1'u32])

  result = readBuffer[byte](ctx, outBuf)

  let vkDestroyShaderModule = cast[
    proc(device: VkDevice, shaderModule: VkShaderModule,
         pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyShaderModule"))
  if shaderModule != nil and vkDestroyShaderModule != nil:
    vkDestroyShaderModule(ctx.device, shaderModule, nil)
