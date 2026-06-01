## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## Minimal Vulkan type definitions — no dynlib imports.
## Used by vulkan_runtime.nim to avoid loading the Vulkan loader.
## All Vulkan functions are loaded dynamically via vk_icdGetInstanceProcAddr.

import std/macros

type
  VkFlags* = uint32
  VkBool32* = uint32
  VkDeviceSize* = uint64
  VkQueueFlags* = uint32
  VkSampleMask* = uint32
  VkResult* = int32
  VkStructureType* = int32
  VkShaderStageFlagBits* = int32
  VkDescriptorType* = int32
  VkDescriptorSetLayoutCreateFlagBits* = int32
  VkBufferUsageFlagBits* = int32
  VkMemoryPropertyFlagBits* = int32
  VkCommandBufferUsageFlagBits* = int32
  VkPipelineBindPoint* = int32
  VkSharingMode* = int32
  VkImageLayout* = int32
  VkImageAspectFlagBits* = int32
  VkFormat* = int32
  VkImageType* = int32
  VkImageViewType* = int32
  VkComponentSwizzle* = int32
  VkAttachmentLoadOp* = int32
  VkAttachmentStoreOp* = int32
  VkPipelineStageFlagBits* = int32
  VkAccessFlagBits* = int32
  VkDependencyFlagBits* = int32
  VkSubpassContents* = int32
  VkCommandBufferLevel* = int32
  VkDescriptorUpdateTemplateType* = int32
  VkObjectType* = int32
  VkIndexType* = int32

  VkInstance* = pointer
  VkPhysicalDevice* = pointer
  VkDevice* = pointer
  VkQueue* = pointer
  VkBuffer* = pointer
  VkDeviceMemory* = pointer
  VkShaderModule* = pointer
  VkPipeline* = pointer
  VkPipelineLayout* = pointer
  VkDescriptorSetLayout* = pointer
  VkDescriptorSet* = pointer
  VkDescriptorPool* = pointer
  VkCommandPool* = pointer
  VkCommandBuffer* = pointer
  VkFence* = pointer
  VkPipelineCache* = pointer
  VkSampler* = pointer
  VkSemaphore* = pointer
  VkEvent* = pointer
  VkQueryPool* = pointer
  VkSurfaceKHR* = pointer
  VkSwapchainKHR* = pointer
  VkImageView* = pointer
  VkFramebuffer* = pointer
  VkRenderPass* = pointer

  VkExtent3D* = object
    width*: uint32
    height*: uint32
    depth*: uint32

  VkQueueFamilyProperties* = object
    queueFlags*: VkQueueFlags
    queueCount*: uint32
    timestampValidBits*: uint32
    minImageTransferGranularity*: VkExtent3D

  VkApplicationInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    pApplicationName*: cstring
    applicationVersion*: uint32
    pEngineName*: cstring
    engineVersion*: uint32
    apiVersion*: uint32

  VkInstanceCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkFlags
    pApplicationInfo*: ptr VkApplicationInfo
    enabledLayerCount*: uint32
    ppEnabledLayerNames*: cstringArray
    enabledExtensionCount*: uint32
    ppEnabledExtensionNames*: cstringArray

  VkDeviceQueueCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkFlags
    queueFamilyIndex*: uint32
    queueCount*: uint32
    pQueuePriorities*: ptr cfloat

  VkDeviceCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkFlags
    queueCreateInfoCount*: uint32
    pQueueCreateInfos*: ptr VkDeviceQueueCreateInfo
    enabledLayerCount*: uint32
    ppEnabledLayerNames*: cstringArray
    enabledExtensionCount*: uint32
    ppEnabledExtensionNames*: cstringArray
    pEnabledFeatures*: pointer

  VkBufferCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkFlags
    size*: VkDeviceSize
    usage*: VkBufferUsageFlagBits
    sharingMode*: VkSharingMode
    queueFamilyIndexCount*: uint32
    pQueueFamilyIndices*: ptr uint32

  VkMemoryRequirements* = object
    size*: VkDeviceSize
    alignment*: VkDeviceSize
    memoryTypeBits*: uint32

  VkPhysicalDeviceProperties* = object
    apiVersion*: uint32
    driverVersion*: uint32
    vendorID*: uint32
    deviceID*: uint32
    deviceType*: int32
    deviceName*: array[256, char]

  VkPhysicalDeviceMemoryProperties* = object
    memoryTypeCount*: uint32
    memoryTypes*: array[32, VkMemoryType]
    memoryHeapCount*: uint32
    memoryHeaps*: array[16, VkMemoryHeap]

  VkMemoryType* = object
    propertyFlags*: VkMemoryPropertyFlagBits
    heapIndex*: uint32

  VkMemoryHeap* = object
    size*: VkDeviceSize
    flags*: VkFlags

  VkMemoryAllocateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    allocationSize*: VkDeviceSize
    memoryTypeIndex*: uint32

  VkShaderModuleCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkFlags
    codeSize*: csize_t
    pCode*: ptr uint32

  VkDescriptorSetLayoutBinding* = object
    binding*: uint32
    descriptorType*: VkDescriptorType
    descriptorCount*: uint32
    stageFlags*: VkShaderStageFlagBits
    pImmutableSamplers*: ptr VkSampler

  VkDescriptorSetLayoutCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkDescriptorSetLayoutCreateFlagBits
    bindingCount*: uint32
    pBindings*: ptr VkDescriptorSetLayoutBinding

  VkDescriptorPoolSize* = object
    `type`*: VkDescriptorType
    descriptorCount*: uint32

  VkDescriptorPoolCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkFlags
    maxSets*: uint32
    poolSizeCount*: uint32
    pPoolSizes*: ptr VkDescriptorPoolSize

  VkDescriptorSetAllocateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    descriptorPool*: VkDescriptorPool
    descriptorSetCount*: uint32
    pSetLayouts*: ptr VkDescriptorSetLayout

  VkDescriptorBufferInfo* = object
    buffer*: VkBuffer
    offset*: VkDeviceSize
    range*: VkDeviceSize

  VkWriteDescriptorSet* = object
    sType*: VkStructureType
    pNext*: pointer
    dstSet*: VkDescriptorSet
    dstBinding*: uint32
    dstArrayElement*: uint32
    descriptorCount*: uint32
    descriptorType*: VkDescriptorType
    pImageInfo*: pointer
    pBufferInfo*: ptr VkDescriptorBufferInfo
    pTexelBufferView*: pointer

  VkPipelineLayoutCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkFlags
    setLayoutCount*: uint32
    pSetLayouts*: ptr VkDescriptorSetLayout
    pushConstantRangeCount*: uint32
    pPushConstantRanges*: pointer

  VkPipelineShaderStageCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkFlags
    stage*: VkShaderStageFlagBits
    module*: VkShaderModule
    pName*: cstring
    pSpecializationInfo*: pointer

  VkComputePipelineCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkFlags
    stage*: VkPipelineShaderStageCreateInfo
    layout*: VkPipelineLayout
    basePipelineHandle*: VkPipeline
    basePipelineIndex*: int32

  VkPipelineCacheCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkFlags
    initialDataSize*: csize_t
    pInitialData*: pointer

  VkCommandPoolCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkFlags
    queueFamilyIndex*: uint32

  VkCommandBufferAllocateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    commandPool*: VkCommandPool
    level*: int32
    commandBufferCount*: uint32

  VkCommandBufferBeginInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkCommandBufferUsageFlagBits
    pInheritanceInfo*: pointer

  VkSubmitInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    waitSemaphoreCount*: uint32
    pWaitSemaphores*: pointer
    pWaitDstStageMask*: pointer
    commandBufferCount*: uint32
    pCommandBuffers*: ptr VkCommandBuffer
    signalSemaphoreCount*: uint32
    pSignalSemaphores*: pointer

  VkFenceCreateInfo* = object
    sType*: VkStructureType
    pNext*: pointer
    flags*: VkFlags

  VkBufferCopy* = object
    srcOffset*: VkDeviceSize
    dstOffset*: VkDeviceSize
    size*: VkDeviceSize

# Constants
const
  VK_STRUCTURE_TYPE_APPLICATION_INFO* = VkStructureType(0)
  VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO* = VkStructureType(1)
  VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO* = VkStructureType(2)
  VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO* = VkStructureType(3)
  VK_STRUCTURE_TYPE_SUBMIT_INFO* = VkStructureType(4)
  VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO* = VkStructureType(5)
  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO* = VkStructureType(6)
  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO* = VkStructureType(7)
  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO* = VkStructureType(8)
  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO* = VkStructureType(10)
  VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO* = VkStructureType(11)
  VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO* = VkStructureType(39)
  VK_STRUCTURE_TYPE_FENCE_CREATE_INFO* = VkStructureType(14)
  VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO* = VkStructureType(23)
  VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET* = VkStructureType(35)
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO* = VkStructureType(40)
  VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO* = VkStructureType(41)
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO* = VkStructureType(42)
  VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO* = VkStructureType(40)
  VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO* = VkStructureType(42)

  VK_SHADER_STAGE_COMPUTE_BIT* = VkShaderStageFlagBits(1 shl 5)
  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER* = VkDescriptorType(7)

  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT* = VkBufferUsageFlagBits(1 shl 4)
  VK_BUFFER_USAGE_TRANSFER_SRC_BIT* = VkBufferUsageFlagBits(1 shl 1)
  VK_BUFFER_USAGE_TRANSFER_DST_BIT* = VkBufferUsageFlagBits(1 shl 0)

  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT* = VkMemoryPropertyFlagBits(1 shl 0)
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT* = VkMemoryPropertyFlagBits(1 shl 1)
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT* = VkMemoryPropertyFlagBits(1 shl 2)

  VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT* = VkCommandBufferUsageFlagBits(1 shl 0)
  VK_PIPELINE_BIND_POINT_COMPUTE* = VkPipelineBindPoint(1)
  VK_SHARING_MODE_EXCLUSIVE* = VkSharingMode(0)

  VK_QUEUE_COMPUTE_BIT* = VkQueueFlags(1 shl 1)
  VK_QUEUE_GRAPHICS_BIT* = VkQueueFlags(1 shl 0)
