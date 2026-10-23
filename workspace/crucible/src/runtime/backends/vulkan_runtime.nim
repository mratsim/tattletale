## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[dynlib, os, osproc, hashes, streams, strutils, tempfiles]
import ../errors
import workspace/crucible/src/abis/vulkan_abi as vk
import ./runtime_utils

type
  VulkanBuffer* = object
    size*: int
    handle: VkBuffer
    memory: VkDeviceMemory
    device: VkDevice
    # resolved at alloc time so the free is engine-free
    destroyBuffer: proc(device: VkDevice, buffer: VkBuffer, pAllocator: pointer) {.cdecl.}
    freeMemory: proc(device: VkDevice, memory: VkDeviceMemory, pAllocator: pointer) {.cdecl.}

  # Not in the vendored ABI — the C struct is {stageFlags: uint32, offset, size}
  VkPushConstantRange = object
    stageFlags: uint32
    offset: uint32
    size: uint32

  VulkanPipeline* = ref VulkanPipelineObj
  VulkanPipelineObj = object
    device: VkDevice
    handle: VkPipeline
    layout: VkPipelineLayout
    setLayout: VkDescriptorSetLayout  ## Set layout from createPipeline, nil on the BDA route
    pushConstSize: int  ## The push-constant range size from createPipeline, 0 when the kernel takes none
    vkDestroyPipeline: proc(device: VkDevice, pipeline: VkPipeline, pAllocator: pointer) {.cdecl.}
    vkDestroyPipelineLayout: proc(device: VkDevice, pipelineLayout: VkPipelineLayout, pAllocator: pointer) {.cdecl.}

  VulkanContext* = ref VulkanContextObj
  VulkanContextObj = object
    instance*: VkInstance
    physicalDevice*: VkPhysicalDevice
    device*: VkDevice
    queue*: VkQueue
    queueFamilyIndex*: uint32
    commandPool*: VkCommandPool
    pipelineCache*: VkPipelineCache
    vkLib*: LibHandle
    getProcAddr*: pointer  ## vkGetInstanceProcAddr


proc `=destroy`(p: var VulkanPipelineObj) =
  ## Releases the pipeline and its layout, nil fields skip.
  if p.device == nil:
    return
  if p.handle != nil and p.vkDestroyPipeline != nil:
    p.vkDestroyPipeline(p.device, p.handle, nil)
  if p.layout != nil and p.vkDestroyPipelineLayout != nil:
    p.vkDestroyPipelineLayout(p.device, p.layout, nil)

template check*(res: VkResult) =
  ## Unified error policy: stacktrace + stderr + quit(1).
  ## No exceptions as the public contract.
  ## A template so instantiationInfo() reports the caller's location.
  let code = res
  if code != VK_SUCCESS:
    writeStackTrace()
    stderr.write($instantiationInfo() & " exited with error: Vulkan error " & $code & '\n')
    quit 1

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
# Vulkan loader
# ═══════════════════════════════════════════════════════════════════════
# vkGetInstanceProcAddr suffices for every other function:
# the loader handles ICD discovery, interface negotiation, and dispatch internally.

proc `=copy`(dst: var VulkanContextObj, src: VulkanContextObj) {.error:
    "VulkanContext is ref-only, copy the ref not the object".}

proc `=destroy`(c: var VulkanContextObj) =
  if c.instance == nil:
    return
  let gpa = cast[proc(instance: VkInstance, name: cstring): pointer {.cdecl.}](c.getProcAddr)
  let vkDestroyPipelineCachePtr = cast[proc(device: VkDevice, pipelineCache: VkPipelineCache, pAllocator: pointer) {.cdecl.}](gpa(c.instance, "vkDestroyPipelineCache"))
  let vkDestroyCommandPoolPtr = cast[proc(device: VkDevice, commandPool: VkCommandPool, pAllocator: pointer) {.cdecl.}](gpa(c.instance, "vkDestroyCommandPool"))
  let vkDestroyDevicePtr = cast[
    proc(device: VkDevice, pAllocator: pointer) {.cdecl.}
  ](gpa(c.instance, "vkDestroyDevice"))
  let vkDestroyInstancePtr = cast[
    proc(instance: VkInstance, pAllocator: pointer) {.cdecl.}
  ](gpa(c.instance, "vkDestroyInstance"))
  if c.pipelineCache != nil and vkDestroyPipelineCachePtr != nil:
    vkDestroyPipelineCachePtr(c.device, c.pipelineCache, nil)
    c.pipelineCache = nil
  if c.commandPool != nil and vkDestroyCommandPoolPtr != nil:
    vkDestroyCommandPoolPtr(c.device, c.commandPool, nil)
    c.commandPool = nil
  if c.device != nil and vkDestroyDevicePtr != nil:
    vkDestroyDevicePtr(c.device, nil)
    c.device = nil
  if c.instance != nil and vkDestroyInstancePtr != nil:
    vkDestroyInstancePtr(c.instance, nil)
    c.instance = nil
  if c.vkLib != nil:
    unloadLib(c.vkLib)
    c.vkLib = nil


proc loadVulkanLoader(): tuple[lib: LibHandle, gpa: pointer] =
  # macOS: dyld does not search /opt/homebrew/lib by default, so dlopen by
  # name misses the Homebrew loader. Try the name first, then the Homebrew
  # path (LunarG SDK installs are handled later if needed).
  when defined(macosx):
    result.lib = loadLib(vk.VulkanLib)
    if result.lib == nil:
      result.lib = loadLib("/opt/homebrew/lib/" & vk.VulkanLib)
  else:
    result.lib = loadLib(vk.VulkanLib)
  if result.lib == nil:
    failLoud("Cannot load " & vk.VulkanLib & " — install a Vulkan loader (e.g. libvulkan1)")
  result.gpa = result.lib.symAddr("vkGetInstanceProcAddr")
  if result.gpa == nil:
    failLoud(vk.VulkanLib & " missing vkGetInstanceProcAddr")

# ═══════════════════════════════════════════════════════════════════════
# SPIR-V compilation
# ═══════════════════════════════════════════════════════════════════════

proc compileGlslToSpirV*(glsl: string; entryPoint: string = "main"): seq[uint32] =
  ## Compiles GLSL to SPIR-V via ``glslangValidator``.
  ## (``libshaderc_shared`` does not support compute shaders on this platform.)
  ## Subgroup shuffles are SPIR-V 1.3 instructions (the KHR subgroup
  ## extensions the codegen emits), so their sources compile with the
  ## vulkan1.1 target env; all other sources stay on the SPIR-V 1.0 default.
  let tmpDir = getKernelDir("vulkan")
  # Private temp dir: 0700 so other local users cannot plant symlinks at
  # deterministic paths (TOCTOU), and unique names so concurrent compiles
  # never collide on the same file.
  setFilePermissions(tmpDir, {fpUserExec, fpUserWrite, fpUserRead})
  let srcPath = genTempPath(sanitizePath(entryPoint) & "_", ".comp", tmpDir)
  let spvPath = genTempPath(sanitizePath(entryPoint) & "_", ".spv", tmpDir)

  writeFile(srcPath, glsl)
  defer:
    if fileExists(srcPath): removeFile(srcPath)
    if fileExists(spvPath): removeFile(spvPath)
  var args = @["-V", "-e", entryPoint, "--source-entrypoint", "main"]
  if "GL_KHR_shader_subgroup" in glsl:
    args.add "--target-env"
    args.add "vulkan1.1"
  args.add "-o"
  args.add spvPath
  args.add srcPath
  let p = startProcess("glslangValidator", args = args,
    options = {poUsePath, poStdErrToStdOut})
  let compOut = p.outputStream.readAll()
  let exitCode = p.waitForExit()
  defer: p.close()
  if exitCode != 0:
    failLoud("glslangValidator failed (exit=" & $exitCode & "):\n" & compOut)

  let raw = readFile(spvPath)
  result = newSeq[uint32](raw.len div 4)
  copyMem(result[0].addr, raw[0].addr, raw.len)
# ═══════════════════════════════════════════════════════════════════════
# Physical device queries
# ═══════════════════════════════════════════════════════════════════════

proc deviceProps(ctx: VulkanContext, dev: VkPhysicalDevice): VkPhysicalDeviceProperties =
  ## Read VkPhysicalDeviceProperties via a padded buffer. The ABI struct is
  ## truncated to the fields we read (apiVersion..deviceName), but the driver
  ## writes the FULL struct including the ~3KB limits/sparseProperties.
  let vkgpd = cast[
    proc(physicalDevice: VkPhysicalDevice,
         pProperties: ptr VkPhysicalDeviceProperties) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkGetPhysicalDeviceProperties"))
  if vkgpd != nil:
    var propsBuf: array[4096, byte]
    vkgpd(dev, cast[ptr VkPhysicalDeviceProperties](propsBuf[0].addr))
    result = cast[ptr VkPhysicalDeviceProperties](propsBuf[0].addr)[]

proc deviceName*(ctx: VulkanContext, dev: VkPhysicalDevice): string =
  ## The physical device name (e.g. "NVIDIA RTX PRO 6000 Blackwell ...").
  for c in deviceProps(ctx, dev).deviceName:
    if c == '\0': break
    result.add c

# ═══════════════════════════════════════════════════════════════════════
# Vulkan initialization
# ═══════════════════════════════════════════════════════════════════════

proc newVulkanContext*(): VulkanContext =
  result = VulkanContext()
  let (lib, gpaRaw) = loadVulkanLoader()
  result.vkLib = lib
  result.getProcAddr = gpaRaw

  let gpaFn = cast[
    proc(instance: VkInstance, pName: cstring): pointer {.cdecl.}
  ](gpaRaw)

  # Load vkCreateInstance via GPA(NULL, name), pre-instance
  let vkCreateInstance = cast[
    proc(pCreateInfo: ptr VkInstanceCreateInfo,
         pAllocator: pointer,
         pInstance: ptr VkInstance): VkResult {.cdecl.}
  ](gpaFn(nil, "vkCreateInstance"))
  if vkCreateInstance == nil:
    failLoud("Cannot load vkCreateInstance")

  var appInfo = VkApplicationInfo(
    sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pApplicationName: "PositronVk",
    applicationVersion: 1,
    pEngineName: "Positron",
    engineVersion: 1,
    apiVersion: 0x00404000
  )
  # MoltenVK needs the portability enumeration opt-in or vkCreateInstance fails
  # with VK_ERROR_INCOMPATIBLE_DRIVER (-9). The extension is a no-op outside macOS.
  # flags 0x00000001 = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR.
  var portabilityExt = [cstring"VK_KHR_portability_enumeration"]
  var instCI = VkInstanceCreateInfo(
    sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    flags: 0x00000001,
    pApplicationInfo: appInfo.addr,
    enabledExtensionCount: 1,
    ppEnabledExtensionNames: cast[cstringArray](portabilityExt.addr)
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
    failLoud("No Vulkan devices found")
  var devices = newSeq[VkPhysicalDevice](devCount.int)
  check ep(result.instance, devCount.addr, devices[0].addr)

  # Queue families
  var gpqfp = cast[
    proc(physicalDevice: VkPhysicalDevice,
         pQueueFamilyPropertyCount: ptr uint32,
         pQueueFamilyProperties: pointer) {.cdecl.}
  ](gpa("vkGetPhysicalDeviceQueueFamilyProperties"))
  var gpdf2 = cast[
    proc(physicalDevice: VkPhysicalDevice, pFeatures: ptr VkPhysicalDeviceFeatures2): void {.cdecl.}
  ](gpa("vkGetPhysicalDeviceFeatures2"))
  # Software renderers (Mesa Xe KMD) can enumerate first and fail vkMapMemory,
  # prefer a discrete GPU with a compute queue
  let initCtx = result   # nested proc shadows `result` — capture the ctx
  proc deviceScore(dev: VkPhysicalDevice): tuple[score: int, name: string] =
    let props = deviceProps(initCtx, dev)
    let name = deviceName(initCtx, dev)
    var qfCount: uint32 = 0
    gpqfp(dev, qfCount.addr, nil)
    var hasCompute = false
    if qfCount > 0:
      var qfProps = newSeq[VkQueueFamilyProperties](qfCount.int)
      gpqfp(dev, qfCount.addr, qfProps[0].addr)
      for f in qfProps:
        if f.queueCount > 0 and (f.queueFlags and VK_QUEUE_COMPUTE_BIT.uint32) != 0:
          hasCompute = true
          break
    if not hasCompute:
      # A graphics-only device dispatches on queue family 0 and fails there
      return (-1, name)
    # Kernel launches pass BDA pointers, the device must enable the feature
    var bdaFeat = VkPhysicalDeviceBufferDeviceAddressFeatures(
      sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES)
    var f2 = VkPhysicalDeviceFeatures2(
      sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
    f2.pNext = bdaFeat.addr
    gpdf2(dev, f2.addr)
    if bdaFeat.bufferDeviceAddress != VK_TRUE:
      return (-1, name)
    let typeScore = case props.deviceType
      of VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: 100
      of VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: 50
      else: 0
    result = (typeScore + 10, name)

  var bestIdx = 0
  var bestScore = -1
  for i, dev in devices:
    let (score, _) = deviceScore(dev)
    if score > bestScore:
      bestScore = score
      bestIdx = i
  if bestScore < 0:
    failLoud("No Vulkan device with a compute-capable queue family found")
  result.physicalDevice = devices[bestIdx]
  echo "  Vulkan device: ", deviceScore(result.physicalDevice).name

  var qfCount: uint32 = 0
  gpqfp(result.physicalDevice, qfCount.addr, nil)
  if qfCount == 0:
    failLoud("No queue families found")
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
  # 16-bit float arithmetic + 16-bit storage (core features since Vulkan 1.1/1.2).
  var f16feat = VkPhysicalDeviceShaderFloat16Int8Features(
    sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES)
  var s16feat = VkPhysicalDevice16BitStorageFeatures(
    sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES)
  var bdaFeat = VkPhysicalDeviceBufferDeviceAddressFeatures(
    sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
    bufferDeviceAddress: VK_TRUE)
  var f2 = VkPhysicalDeviceFeatures2(
    sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
  f16feat.pNext = s16feat.addr
  s16feat.pNext = bdaFeat.addr
  f2.pNext = f16feat.addr
  gpdf2(result.physicalDevice, f2.addr)
  var dci = VkDeviceCreateInfo(
    sType: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    pNext: f2.addr,
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


# ═══════════════════════════════════════════════════════════════════════
# Buffer management
# ═══════════════════════════════════════════════════════════════════════

proc allocBuffer*(ctx: var VulkanContext, size: int): VulkanBuffer =
  result.size = size
  result.device = ctx.device
  result.destroyBuffer = cast[
    proc(device: VkDevice, buffer: VkBuffer, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyBuffer"))
  result.freeMemory = cast[
    proc(device: VkDevice, memory: VkDeviceMemory, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkFreeMemory"))

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
           VK_BUFFER_USAGE_TRANSFER_DST_BIT or
           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
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
  #  3. HOST_VISIBLE only
  #  4. Unspecified flags (driver-declared host-accessible, e.g. llvmpipe)
  #  5. DEVICE_LOCAL only (VRAM) — cannot be vkMapMemory'd; last resort
  var memTypeIdx = high(uint32)
  var devLocalOnly = high(uint32)
  for i in 0'u32 ..< memProps.memoryTypeCount:
    if (memReq.memoryTypeBits and (1'u32 shl i)) == 0:
      continue
    let flags = cast[uint32](memProps.memoryTypes[i].propertyFlags)
    let best = cast[uint32](
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT or
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    if (flags and best) == best:
      memTypeIdx = i; break
    if (flags and cast[uint32](VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) != 0:
      if memTypeIdx == high(uint32):
        memTypeIdx = i
      continue
    if (flags and cast[uint32](VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) == 0:
      if memTypeIdx == high(uint32):
        memTypeIdx = i
    else:
      if devLocalOnly == high(uint32):
        devLocalOnly = i
  if memTypeIdx == high(uint32):
    memTypeIdx = devLocalOnly
  if memTypeIdx == high(uint32):
    var details = "memoryTypeBits=" & $memReq.memoryTypeBits & " memTypeCount=" & $memProps.memoryTypeCount
    for i in 0'u32 ..< memProps.memoryTypeCount:
      details &= " [" & $i & ": flags=" & $(cast[int](memProps.memoryTypes[i].propertyFlags)) & " heap=" & $memProps.memoryTypes[i].heapIndex & "]"
    failLoud("No suitable memory type found — " & details)
  if (cast[uint32](memProps.memoryTypes[memTypeIdx].propertyFlags) and
      cast[uint32](VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) == 0 and
     (cast[uint32](memProps.memoryTypes[memTypeIdx].propertyFlags) and
      cast[uint32](VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) != 0:
    # Only a DEVICE_LOCAL-only type matched: vkMapMemory (used by
    # writeBuffer/readBuffer) fails on it with VK_ERROR_MEMORY_MAP_FAILED.
    # Some drivers (e.g. broken/experimental Vulkan in containers, Mesa Xe KMD)
    # report only DEVICE_LOCAL types.
    var details = "memoryTypeBits=" & $memReq.memoryTypeBits & " memTypeCount=" & $memProps.memoryTypeCount
    for i in 0'u32 ..< memProps.memoryTypeCount:
      details &= " [" & $i & ": flags=" & $(cast[int](memProps.memoryTypes[i].propertyFlags)) & " heap=" & $memProps.memoryTypes[i].heapIndex & "]"
    failLoud("No host-visible memory type available (vkMapMemory would fail) — " & details)

  when defined(debug):
    echo "  [allocBuffer] size=", size, " memTypeBits=", memReq.memoryTypeBits,
         " chosen type ", memTypeIdx,
         " heap=", memProps.memoryTypes[memTypeIdx].heapIndex
  var allocFlags = VkMemoryAllocateFlagsInfo(
    sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
    flags: VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
  var allocInfo = VkMemoryAllocateInfo(
    sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    pNext: allocFlags.addr,
    allocationSize: memReq.size,
    memoryTypeIndex: memTypeIdx
  )
  check vkAllocateMemory(ctx.device, allocInfo.addr, nil, result.memory.addr)
  check vkBindBufferMemory(ctx.device, result.handle, result.memory, 0)

proc deviceAddress*(buf: VulkanBuffer, ctx: VulkanContext): uint64 =
  ## Shader-visible 64-bit address of a buffer, the value a pointer param carries.
  let vkGetBufferDeviceAddress = cast[
    proc(device: VkDevice, pInfo: ptr VkBufferDeviceAddressInfo): VkDeviceAddress {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkGetBufferDeviceAddress"))
  var info = VkBufferDeviceAddressInfo(
    sType: VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
    buffer: buf.handle)
  result = uint64(vkGetBufferDeviceAddress(ctx.device, info.addr))

proc dealloc*(buffer: var VulkanBuffer) =
  if buffer.handle == nil:
    return
  if buffer.handle != nil and buffer.destroyBuffer != nil:
    buffer.destroyBuffer(buffer.device, buffer.handle, nil)
    buffer.handle = nil
  if buffer.memory != nil and buffer.freeMemory != nil:
    buffer.freeMemory(buffer.device, buffer.memory, nil)
    buffer.memory = nil
proc writeBuffer*(ctx: VulkanContext, buf: var VulkanBuffer, data: pointer, size: int) =
  if size > buf.size:
    failLoud("writeBuffer overflow")
  let vkMapMemory = cast[
    proc(device: VkDevice, memory: VkDeviceMemory,
         offset: VkDeviceSize, size: VkDeviceSize,
         flags: VkFlags, ppData: ptr pointer): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkMapMemory"))
  let vkUnmapMemory = cast[
    proc(device: VkDevice, memory: VkDeviceMemory) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkUnmapMemory"))
  var mapped: pointer
  let mres = vkMapMemory(buf.device, buf.memory, 0, VkDeviceSize(size), 0, mapped.addr)
  if mres != VK_SUCCESS:
    failLoud("vkMapMemory failed (" & $mres & ")")
  copyMem(mapped, data, size)
  vkUnmapMemory(buf.device, buf.memory)

proc readBuffer*[T](ctx: VulkanContext, buf: VulkanBuffer): seq[T] =
  if buf.size mod sizeof(T) != 0:
    failLoud("Buffer size not divisible by type size")
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
# Shader modules and descriptor sets
# ═══════════════════════════════════════════════════════════════════════
# Descriptor route: buffers bind as STORAGE_BUFFER descriptors.
# One binding per buffer parameter: binding i is the i-th buffer parameter in declaration order,
# in descriptor set 0, matching the printer's [[vk::binding(N, 0)]] emission.
# Push-constant parameters take no binding.
# createPipeline carries the set layout, runKernel binds the set.

proc createShaderModule*(ctx: VulkanContext, spirv: seq[uint32]): VkShaderModule =
  ## vkCreateShaderModule over validated SPIR-V words.
  ## Caller releases the module with destroyShaderModule.
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
  check vkCreateShaderModule(ctx.device, smCI.addr, nil, result.addr)

proc destroyShaderModule*(ctx: VulkanContext, shaderModule: VkShaderModule) =
  ## Releases a module from createShaderModule, nil skips.
  let vkDestroyShaderModule = cast[
    proc(device: VkDevice, shaderModule: VkShaderModule, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyShaderModule"))
  if shaderModule != nil:
    vkDestroyShaderModule(ctx.device, shaderModule, nil)

proc createDescriptorSetLayout*(ctx: VulkanContext, bindingCount: uint32): VkDescriptorSetLayout =
  ## One STORAGE_BUFFER binding per buffer argument, compute stage only:
  ## binding i is the i-th buffer parameter in declaration order, in descriptor set 0,
  ## matching the printer's [[vk::binding(N, 0)]] numbering. Push-constant parameters take no binding.
  let vkCreateDescriptorSetLayout = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkDescriptorSetLayoutCreateInfo,
         pAllocator: pointer,
         pSetLayout: ptr VkDescriptorSetLayout): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreateDescriptorSetLayout"))
  var bindings = newSeq[VkDescriptorSetLayoutBinding](bindingCount.int)
  for i in 0 ..< bindingCount.int:
    bindings[i] = VkDescriptorSetLayoutBinding(
      binding: uint32(i),
      descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      descriptorCount: 1,
      stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
      pImmutableSamplers: nil
    )
  var slCI = VkDescriptorSetLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    bindingCount: bindingCount
  )
  if bindingCount > 0:
    slCI.pBindings = bindings[0].addr
  check vkCreateDescriptorSetLayout(ctx.device, slCI.addr, nil, result.addr)

proc destroyDescriptorSetLayout*(ctx: VulkanContext, setLayout: VkDescriptorSetLayout) =
  ## Releases a set layout from createDescriptorSetLayout, nil skips.
  let vkDestroyDescriptorSetLayout = cast[
    proc(device: VkDevice, descriptorSetLayout: VkDescriptorSetLayout, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyDescriptorSetLayout"))
  if setLayout != nil:
    vkDestroyDescriptorSetLayout(ctx.device, setLayout, nil)

proc createDescriptorPool*(ctx: VulkanContext, bindingCount: uint32): VkDescriptorPool =
  ## Pool sized to exactly one set of bindingCount storage-buffer descriptors.
  ## Set lifetime is owned by the pool: destroyDescriptorPool frees the sets.
  let vkCreateDescriptorPool = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkDescriptorPoolCreateInfo,
         pAllocator: pointer,
         pDescriptorPool: ptr VkDescriptorPool): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreateDescriptorPool"))
  var poolSize = VkDescriptorPoolSize(
    `type`: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    descriptorCount: bindingCount
  )
  var poolCI = VkDescriptorPoolCreateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    maxSets: 1,
    poolSizeCount: 1,
    pPoolSizes: poolSize.addr
  )
  check vkCreateDescriptorPool(ctx.device, poolCI.addr, nil, result.addr)

proc destroyDescriptorPool*(ctx: VulkanContext, pool: VkDescriptorPool) =
  ## Releases a pool and every set allocated from it, nil skips.
  let vkDestroyDescriptorPool = cast[
    proc(device: VkDevice, descriptorPool: VkDescriptorPool, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyDescriptorPool"))
  if pool != nil:
    vkDestroyDescriptorPool(ctx.device, pool, nil)

proc allocateDescriptorSet*(ctx: VulkanContext, pool: VkDescriptorPool, setLayout: VkDescriptorSetLayout): VkDescriptorSet =
  ## One descriptor set from `pool`, bound against `setLayout`.
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

proc updateDescriptorSet*(ctx: VulkanContext, setHandle: VkDescriptorSet, buffers: openArray[VulkanBuffer], bindingCount: uint32) =
  ## One write per binding: binding i covers the whole i-th buffer.
  ##
  ## Precondition:
  ## - `buffers` enumerates the kernel's buffer parameters in declaration order,
  ##   binding i writes the i-th buffer, matching the printer's [[vk::binding(N, 0)]] numbering
  ##   (buffers only, push-constant parameters unbound)
  ## - `bindingCount` is the set layout's binding count from createDescriptorSetLayout:
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
  var bufInfos = newSeq[VkDescriptorBufferInfo](buffers.len)
  var writes = newSeq[VkWriteDescriptorSet](buffers.len)
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
  if writes.len > 0:
    vkUpdateDescriptorSets(ctx.device, uint32(writes.len), writes[0].addr, 0, nil)

# ═══════════════════════════════════════════════════════════════════════
# Pipeline management
# ═══════════════════════════════════════════════════════════════════════

proc createPipeline*(ctx: var VulkanContext, shaderModule: VkShaderModule,
                     entryPoint: string = "main",
                     pushConstSize: int = 0,
                     setLayout: VkDescriptorSetLayout = nil): VulkanPipeline =
  ## Compute pipeline over a shader module.
  ##
  ## Default layout: push constants only, buffers bind by device address (BDA).
  ## With `setLayout` the layout carries that descriptor set, the dispatch binds it (runKernel's `descriptorSet`).
  ##
  ## Returns: the pipeline, `=destroy` releases the pipeline and its layout.
  ## runKernel checks its `descriptorSet` and `pushConstSize` arguments against the recorded
  ## setLayout and pushConstSize, a mismatch quits loudly.
  result = VulkanPipeline()
  result.device = ctx.device
  result.setLayout = setLayout
  result.pushConstSize = pushConstSize
  result.vkDestroyPipeline = cast[
    proc(device: VkDevice, pipeline: VkPipeline, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyPipeline"))
  result.vkDestroyPipelineLayout = cast[
    proc(device: VkDevice, pipelineLayout: VkPipelineLayout, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyPipelineLayout"))

  let vkCreatePipelineLayout = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkPipelineLayoutCreateInfo,
         pAllocator: pointer,
         pPipelineLayout: ptr VkPipelineLayout): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreatePipelineLayout"))
  let vkCreateComputePipelines = cast[
    proc(device: VkDevice, pipelineCache: VkPipelineCache,
         createInfoCount: uint32,
         pCreateInfos: ptr VkComputePipelineCreateInfo,
         pAllocator: pointer,
         pPipelines: ptr VkPipeline): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreateComputePipelines"))

  var plCI = VkPipelineLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
  var pushConstRange: VkPushConstantRange
  if setLayout != nil:
    plCI.setLayoutCount = 1
    plCI.pSetLayouts = setLayout.addr
  if pushConstSize > 0:
    pushConstRange = VkPushConstantRange(
      stageFlags: uint32(VK_SHADER_STAGE_COMPUTE_BIT),
      offset: 0,
      size: uint32(pushConstSize)
    )
    plCI.pushConstantRangeCount = 1
    plCI.pPushConstantRanges = pushConstRange.addr
  check vkCreatePipelineLayout(ctx.device, plCI.addr, nil, result.layout.addr)

  var stageInfo = VkPipelineShaderStageCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    stage: VK_SHADER_STAGE_COMPUTE_BIT,
    module: shaderModule,
    pName: entryPoint
  )
  var cpCI = VkComputePipelineCreateInfo(
    sType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    stage: stageInfo,
    layout: result.layout
  )
  check vkCreateComputePipelines(ctx.device, ctx.pipelineCache, 1, cpCI.addr, nil, result.handle.addr)

# ═══════════════════════════════════════════════════════════════════════
# Kernel execution
# ═══════════════════════════════════════════════════════════════════════

proc runKernel*(ctx: VulkanContext, pipeline: VulkanPipeline,
                globalWorkSize, localWorkSize: openArray[uint32],
                pushConstData: pointer = nil, pushConstSize: int = 0,
                descriptorSet: VkDescriptorSet = nil) =
  ## One dispatch: bind pipeline, push constants, dispatch, fence wait.
  ##
  ## With `descriptorSet` that set binds as set 0 against the pipeline layout, which must carry the matching set layout.
  ##
  ## Returns: nothing, the fence wait blocks until the dispatch completes.
  ##
  ## Precondition: `descriptorSet` is given iff the pipeline was created with a setLayout,
  ## and `pushConstSize` matches the pipeline's recorded push-constant range.
  ## A violation is silent on the device (stale descriptor state, wrong push-constant bytes):
  ## the checks quit loudly instead.
  if (descriptorSet != nil) != (pipeline.setLayout != nil):
    failLoud("Vulkan runKernel: descriptorSet and the pipeline's set layout must be given together, got " &
      (if descriptorSet != nil: "a descriptor set" else: "no descriptor set") & " against " &
      (if pipeline.setLayout != nil: "a set layout" else: "no set layout"))
  if pushConstSize != pipeline.pushConstSize:
    failLoud("Vulkan runKernel: pushConstSize " & $pushConstSize &
      " does not match the pipeline's recorded push-constant range " & $pipeline.pushConstSize)
  let vkAllocateCommandBuffers = cast[
    proc(device: VkDevice, pAllocateInfo: ptr VkCommandBufferAllocateInfo, pCommandBuffers: ptr VkCommandBuffer): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkAllocateCommandBuffers"))
  let vkBeginCommandBuffer = cast[
    proc(commandBuffer: VkCommandBuffer, pBeginInfo: ptr VkCommandBufferBeginInfo): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkBeginCommandBuffer"))
  let vkCmdBindPipeline = cast[
    proc(commandBuffer: VkCommandBuffer, pipelineBindPoint: VkPipelineBindPoint, pipeline: VkPipeline) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCmdBindPipeline"))
  let vkCmdDispatch = cast[
    proc(commandBuffer: VkCommandBuffer, groupCountX: uint32, groupCountY: uint32, groupCountZ: uint32) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCmdDispatch"))
  let vkEndCommandBuffer = cast[
    proc(commandBuffer: VkCommandBuffer): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkEndCommandBuffer"))
  let vkCreateFence = cast[
    proc(device: VkDevice, pCreateInfo: ptr VkFenceCreateInfo, pAllocator: pointer, pFence: ptr VkFence): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCreateFence"))
  let vkQueueSubmit = cast[
    proc(queue: VkQueue, submitCount: uint32, pSubmits: ptr VkSubmitInfo, fence: VkFence): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkQueueSubmit"))
  let vkWaitForFences = cast[
    proc(device: VkDevice, fenceCount: uint32, pFences: ptr VkFence, waitAll: VkBool32, timeout: uint64): VkResult {.cdecl.}
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

  if descriptorSet != nil:
    let vkCmdBindDescriptorSets = cast[
      proc(commandBuffer: VkCommandBuffer, pipelineBindPoint: VkPipelineBindPoint,
           layout: VkPipelineLayout, firstSet: uint32, descriptorSetCount: uint32,
           pDescriptorSets: ptr VkDescriptorSet, dynamicOffsetCount: uint32,
           pDynamicOffsets: ptr uint32) {.cdecl.}
    ](ctx.gpaAddr(ctx.instance, "vkCmdBindDescriptorSets"))
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout,
                            0, 1, descriptorSet.addr, 0, nil)

  if pushConstSize > 0:
    let vkCmdPushConstants = cast[
      proc(commandBuffer: VkCommandBuffer, layout: VkPipelineLayout, stageFlags: uint32, offset: uint32, size: uint32, pValues: pointer) {.cdecl.}
    ](ctx.gpaAddr(ctx.instance, "vkCmdPushConstants"))
    vkCmdPushConstants(cmdBuf, pipeline.layout, uint32(VK_SHADER_STAGE_COMPUTE_BIT),
                       0, uint32(pushConstSize), pushConstData)

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
