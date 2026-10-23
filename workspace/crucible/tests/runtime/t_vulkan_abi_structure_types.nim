## VkStructureType constants in the Vulkan ABI
##
## Test that every VK_STRUCTURE_TYPE_* constant matches the vulkan_core.h numbering
## (Vulkan 1.4, header version 363), and that COPY_BUFFER_INFO does not exist:
## vkCmdCopyBuffer takes plain parameters, there is no such structure.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/runtime/t_vulkan_abi_structure_types.nim

import workspace/crucible/src/abis/vulkan_abi

static:
  doAssert VK_STRUCTURE_TYPE_APPLICATION_INFO == VkStructureType(0)
  doAssert VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO == VkStructureType(1)
  doAssert VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO == VkStructureType(2)
  doAssert VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO == VkStructureType(3)
  doAssert VK_STRUCTURE_TYPE_SUBMIT_INFO == VkStructureType(4)
  doAssert VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO == VkStructureType(5)
  doAssert VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO == VkStructureType(16)
  doAssert VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO == VkStructureType(30)
  doAssert VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO == VkStructureType(18)
  doAssert VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO == VkStructureType(29)
  doAssert VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO == VkStructureType(12)
  doAssert VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO == VkStructureType(39)
  doAssert VK_STRUCTURE_TYPE_FENCE_CREATE_INFO == VkStructureType(8)
  doAssert VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO == VkStructureType(1000157000)
  doAssert VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO == VkStructureType(17)
  doAssert VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2 == VkStructureType(1000146000)
  doAssert VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET == VkStructureType(35)
  doAssert VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO == VkStructureType(32)
  doAssert VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO == VkStructureType(33)
  doAssert VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO == VkStructureType(34)
  doAssert VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO == VkStructureType(40)
  doAssert VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO == VkStructureType(42)
  doAssert VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE == VkStructureType(6)
  doAssert VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 == VkStructureType(1000059000)
  doAssert VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES == VkStructureType(1000082000)
  doAssert VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES == VkStructureType(1000083000)
  doAssert VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES == VkStructureType(1000257000)
  doAssert VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO == VkStructureType(1000244001)
  doAssert VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO == VkStructureType(1000060000)
  doAssert not compiles(VK_STRUCTURE_TYPE_COPY_BUFFER_INFO),
    "COPY_BUFFER_INFO does not exist in vulkan_core.h, vkCmdCopyBuffer takes plain parameters"
