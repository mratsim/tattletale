# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, tables]

import workspace/crucible/src/abis/vulkan_abi as vk
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/crucible_codegen
import workspace/crucible/src/runtime/backends/vulkan_common
import workspace/crucible/src/runtime/backends/vulkan_address_runtime {.all.}
import workspace/crucible/src/runtime/backends/vulkan_compile {.all.}
import workspace/crucible/src/runtime/backends/runtime_utils
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/launch/launch_device_fn {.all.}
import workspace/crucible/src/runtime/launch/launch_serialization

export engines_common.LaunchConfig, engines_common.dim3

# ############################################################
#
#                   Vulkan batch submission
#
# ############################################################

type
  BatchEntry = object
    entrypoint: string
    spv: seq[uint32]
    pipelineState: VkPipeline
    fleet, squadron: Dim3
    borrowedBuffers: seq[VulkanBuffer]
    scalars: seq[byte]

  BatchPipeline = object
    device: VkDevice
    handle: VkPipeline
    layout: VkPipelineLayout
    destroyPipeline: proc(device: VkDevice, pipeline: VkPipeline, pAllocator: pointer) {.cdecl.}
    destroyLayout: proc(device: VkDevice, pipelineLayout: VkPipelineLayout, pAllocator: pointer) {.cdecl.}

  VulkanBatchObj = object
    ctx: VulkanContext
    entries: seq[BatchEntry]
    pipelines: Table[(string, seq[uint32]), BatchPipeline]
      # entrypoint + module words: one name can bake different LocalSizes per entry
    recording: VkCommandBuffer
    cmdPool: VkCommandPool
    destroyCommandPool: proc(device: VkDevice, commandPool: VkCommandPool, pAllocator: pointer) {.cdecl.}

  VulkanBatch* = ref VulkanBatchObj

  VkMemoryBarrier = object
    sType: VkStructureType
    pNext: pointer
    srcAccessMask: uint32
    dstAccessMask: uint32

  VkPushConstantRange = object
    stageFlags: uint32
    offset: uint32
    size: uint32

const
  VK_STRUCTURE_TYPE_MEMORY_BARRIER = VkStructureType(46)
  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800
  VK_ACCESS_SHADER_READ_BIT = 0x00000020
  VK_ACCESS_SHADER_WRITE_BIT = 0x00000040

proc `=destroy`(batch: var VulkanBatchObj) =
  ## Releases the batch's own objects, never the borrowed context or buffers.
  # Destroying the pool frees the recorded command buffer allocated from it.
  if batch.cmdPool != nil and batch.destroyCommandPool != nil:
    batch.destroyCommandPool(batch.ctx.device, batch.cmdPool, nil)
  for p in batch.pipelines.values:
    if p.handle != nil and p.destroyPipeline != nil:
      p.destroyPipeline(p.device, p.handle, nil)
    if p.layout != nil and p.destroyLayout != nil:
      p.destroyLayout(p.device, p.layout, nil)

proc checkPushRoom(pushConstantCursor, need: int) {.inline.} =
  if pushConstantCursor + need > 128:
    raiseAssert("Vulkan push constants exceed the spec-guaranteed 128-byte max")

type
  EntryPack = object
    ## Packs one entry's push-constant block in device function signature order:
    ## device addresses at 8-byte alignment, scalars at their device layout.
    vctx: VulkanContext
    pushConstantCursor: int
    scalars: seq[byte]
    buffers: seq[VulkanBuffer]

proc write8(pack: var EntryPack, x: uint64) =
  pack.pushConstantCursor = pack.pushConstantCursor.round_multiple_up(8)
  checkPushRoom(pack.pushConstantCursor, 8)
  pack.scalars.setLen(pack.pushConstantCursor + 8)
  copyMem(pack.scalars[pack.pushConstantCursor].addr, x.unsafeAddr, 8)
  inc pack.pushConstantCursor, 8

proc store(pack: var EntryPack, v: VulkanBuffer, _: int) =
  pack.buffers.add v
  pack.write8(deviceAddress(v, pack.vctx))

proc store[T](pack: var EntryPack, v: T, _: int) =
  when T is ptr|pointer:
    {.error: "Pointer device function args are not supported, pass the DeviceMem handle from allocDevice, got " & $T.}
  const lay = computeDeviceLayout(T, cgGlsl)
  if lay.align > 1:
    pack.pushConstantCursor = pack.pushConstantCursor.round_multiple_up(lay.align)
  checkPushRoom(pack.pushConstantCursor, lay.size)
  pack.scalars.setLen(pack.pushConstantCursor + lay.size)
  var blob: array[lay.size, byte]
  var off = 0
  when T is object|tuple and hasBufferLeaf(T, VulkanBuffer):
    for f in fields(v):
      when typeof(f) is DeviceMem:
        off = off.round_multiple_up(8)
        pack.buffers.add f.mem
        let a = deviceAddress(f.mem, pack.vctx)
        copyMem(blob[off].addr, a.unsafeAddr, 8)
        inc off, 8
      else:
        blob.packFrom(off, f, cgGlsl)
    off = off.round_multiple_up(lay.align)
    doAssert off == lay.size,
      "host packing of " & $T & " does not match the device layout"
  else:
    blob.packFrom(off, v, cgGlsl)
  if lay.size > 0:
    copyMem(pack.scalars[pack.pushConstantCursor].addr, blob[0].addr, lay.size)
  inc pack.pushConstantCursor, lay.size

proc recordGlobalFn*(batch: VulkanBatch, spv: seq[uint32], entrypoint: string, cfg: LaunchConfig, output: auto, args: tuple) =
  if batch.recording != nil:
    raiseAssert("recordGlobalFn: the batch is finalized, entries are frozen")
  if cfg.sharedMem != 0 or cfg.stream != 0:
    raiseAssert("Vulkan batch: launch-time sharedMem/stream are unsupported, they are Metal/CUDA features")
  if cfg.grid.x < 1 or cfg.grid.y < 1 or cfg.grid.z < 1:
    raiseAssert("Vulkan batch: cfg.grid must be >= 1 per axis, got " &
                $cfg.grid.x & "x" & $cfg.grid.y & "x" & $cfg.grid.z)
  let sq = parseSpirvEntryDescriptor(spv)
  if sq.entryPoints != 1 or sq.name != entrypoint:
    raiseAssert("Vulkan batch: the module for \"" & entrypoint &
             "\" must declare exactly one OpEntryPoint named after the device function, got " &
             $sq.entryPoints & " entrypoints" &
             (if sq.entryPoints == 1: " (\"" & sq.name & "\")" else: ""))
  if sq.localSize.x == 0 or sq.localSize.y == 0 or sq.localSize.z == 0:
    raiseAssert("Vulkan batch: the module for \"" & entrypoint & "\" bakes no workgroup size")
  # A blk of (1, 1, 1) drops the workgroup pragma from the wrapper.
  # Printed modules then bake the printer's default workgroup size.
  let squadron = if cfg.blk.x == 1 and cfg.blk.y == 1 and cfg.blk.z == 1:
              sq.localSize
            else:
              cfg.blk
  if squadron.x != sq.localSize.x or squadron.y != sq.localSize.y or
     squadron.z != sq.localSize.z:
    raiseAssert("Vulkan batch entry " & entrypoint & ": squadron " &
         $squadron.x & "x" & $squadron.y & "x" & $squadron.z &
         " != baked squadron " & $sq.localSize.x & "x" & $sq.localSize.y &
         "x" & $sq.localSize.z &
         ": launch config mismatch, squadron is shader-baked on Vulkan")
  var pack = EntryPack(vctx: batch.ctx)
  bindArgs(pack, output, args, store, VulkanBuffer, cgGlsl)
  # VUID-VkPushConstantRange-size-00310 and VUID-vkCmdPushConstants-size-00370:
  # push-constant range and update sizes must be multiples of 4
  pack.scalars.setLen(pack.scalars.len.round_multiple_up(4))
  batch.entries.add BatchEntry(
    entrypoint: entrypoint,
    spv: spv,
    fleet: cfg.grid,
    squadron: squadron,
    borrowedBuffers: pack.buffers,
    scalars: pack.scalars)

macro recordGlobalFn*(batch: VulkanBatch, deviceFn: typed, cfg: static LaunchConfig, output: typed, args: typed): untyped =
  var argNodes = args
  if argNodes.kind == nnkPar and argNodes.len == 1:
    argNodes = argNodes[0]
  doAssert argNodes.kind in {nnkTupleConstr, nnkPar},
    "batch enqueue: args must be a tuple, got " & args.repr
  var callN = nnkCall.newTree(deviceFn)
  callN.add(output)
  for a in argNodes:
    callN.add(a)
  let k = callN.resolveDeviceFnOverload()
  let impl = k.getImpl()
  impl.expectKind(nnkProcDef)
  let formals = impl[3]
  var formalTypes: seq[NimNode]
  for i in 1 ..< formals.len:
    let f = formals[i]
    for j in 0 ..< f.len - 2:
      formalTypes.add(f[^2])
  doAssert argNodes.len + 1 <= formalTypes.len,
    "batch enqueue: " & $(argNodes.len + 1) & " args for " &
    $formalTypes.len & " parameters"
  var statics = nnkBracket.newTree()
  var valuesN = nnkBracket.newTree()
  var valuesTuple = nnkTupleConstr.newTree()
  var allArgs = @[output]
  for i in 0 ..< argNodes.len:
    allArgs.add(argNodes[i])
  for i, a in allArgs:
    if isCompileTime(formalTypes[i]):
      statics.add(a)
    else:
      valuesN.add(a)
      if i > 0:
        valuesTuple.add(a)
  let procdef = buildGlobalFn(k, statics, valuesN)
  let wg = getWorkgroupPragma(nnkPar.newTree(
    newLit(cfg.blk.x), newLit(cfg.blk.y), newLit(cfg.blk.z)))
  if not wg.isNil:
    procdef[4].add(wg)
  procdef[6][0][0] = k
  let kname = newLit($procdef[0])
  let configCall = nnkObjConstr.newTree(
    bindSym"LaunchConfig",
    nnkExprColonExpr.newTree(ident"grid",
      bindSym"dim3".newCall(newLit(cfg.grid.x), newLit(cfg.grid.y), newLit(cfg.grid.z))),
    nnkExprColonExpr.newTree(ident"blk",
      bindSym"dim3".newCall(newLit(cfg.blk.x), newLit(cfg.blk.y), newLit(cfg.blk.z))),
    nnkExprColonExpr.newTree(ident"sharedMem", newLit(cfg.sharedMem)),
    nnkExprColonExpr.newTree(ident"stream", newLit(cfg.stream)))
  let spirvN = bindSym"spirv"
  let implN = bindSym"recordGlobalFn"
  result = quote do:
    block:
      const batchModule = `spirvN`(`procdef`)
      `implN`(`batch`, batchModule, `kname`, `configCall`, `output`, `valuesTuple`)

proc newVulkanBatch*(ctx: VulkanContext): VulkanBatch =
  ## Borrowed: the context, its device and queue, and every bound buffer.
  VulkanBatch(ctx: ctx)

func getScalarSize*(batch: VulkanBatch): int =
  for e in batch.entries:
    result += e.scalars.len

proc createBatchPipeline(batch: VulkanBatch, entry: BatchEntry): BatchPipeline =
  ## Compute pipeline over the entry's module: push constants only, buffers bind by device address, zero descriptor sets.
  let ctx = batch.ctx
  result.device = ctx.device
  result.destroyPipeline = cast[
    proc(device: VkDevice, pipeline: VkPipeline, pAllocator: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkDestroyPipeline"))
  result.destroyLayout = cast[
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
  var pushConstRange: VkPushConstantRange
  var plCI = VkPipelineLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO)
  if entry.scalars.len > 0:
    pushConstRange = VkPushConstantRange(
      stageFlags: uint32(VK_SHADER_STAGE_COMPUTE_BIT),
      offset: 0,
      size: uint32(entry.scalars.len))
    plCI.pushConstantRangeCount = 1
    plCI.pPushConstantRanges = pushConstRange.addr
  check vkCreatePipelineLayout(ctx.device, plCI.addr, nil, result.layout.addr)
  let shaderModule = ctx.createShaderModule(entry.spv)
  var stageInfo = VkPipelineShaderStageCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    stage: VK_SHADER_STAGE_COMPUTE_BIT,
    module: shaderModule,
    pName: entry.entrypoint
  )
  var cpCI = VkComputePipelineCreateInfo(
    sType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    stage: stageInfo,
    layout: result.layout
  )
  check vkCreateComputePipelines(ctx.device, ctx.pipelineCache, 1, cpCI.addr, nil, result.handle.addr)
  ctx.destroyShaderModule(shaderModule)

proc finalize*(batch: VulkanBatch) =
  if batch.recording != nil:
    return
  if batch.entries.len == 0:
    raiseAssert("finalize: the batch has no entries")
  let ctx = batch.ctx
  let vkCreateCommandPool = cast[proc(device: VkDevice, pCreateInfo: ptr VkCommandPoolCreateInfo, pAllocator: pointer, pCommandPool: ptr VkCommandPool): VkResult {.cdecl.}](ctx.gpaAddr(ctx.instance, "vkCreateCommandPool"))
  let vkAllocateCommandBuffers = cast[
    proc(device: VkDevice, pAllocateInfo: ptr VkCommandBufferAllocateInfo, pCommandBuffers: ptr VkCommandBuffer): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkAllocateCommandBuffers"))
  let vkBeginCommandBuffer = cast[
    proc(commandBuffer: VkCommandBuffer, pBeginInfo: ptr VkCommandBufferBeginInfo): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkBeginCommandBuffer"))
  let vkEndCommandBuffer = cast[
    proc(commandBuffer: VkCommandBuffer): VkResult {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkEndCommandBuffer"))
  let vkCmdBindPipeline = cast[
    proc(commandBuffer: VkCommandBuffer, pipelineBindPoint: VkPipelineBindPoint, pipeline: VkPipeline) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCmdBindPipeline"))
  let vkCmdPushConstants = cast[
    proc(commandBuffer: VkCommandBuffer, layout: VkPipelineLayout, stageFlags: uint32, offset: uint32, size: uint32, pValues: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCmdPushConstants"))
  let vkCmdDispatch = cast[
    proc(commandBuffer: VkCommandBuffer, groupCountX: uint32, groupCountY: uint32, groupCountZ: uint32) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCmdDispatch"))
  let vkCmdPipelineBarrier = cast[
    proc(commandBuffer: VkCommandBuffer, srcStageMask: VkFlags, dstStageMask: VkFlags,
         dependencyFlags: VkFlags, memoryBarrierCount: uint32, pMemoryBarriers: ptr VkMemoryBarrier,
         bufferMemoryBarrierCount: uint32, pBufferMemoryBarriers: pointer,
         imageMemoryBarrierCount: uint32, pImageMemoryBarriers: pointer) {.cdecl.}
  ](ctx.gpaAddr(ctx.instance, "vkCmdPipelineBarrier"))
  if batch.cmdPool == nil:
    var cpc = VkCommandPoolCreateInfo(
      sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      queueFamilyIndex: ctx.queueFamilyIndex
    )
    check vkCreateCommandPool(ctx.device, cpc.addr, nil, batch.cmdPool.addr)
    batch.destroyCommandPool = cast[
      proc(device: VkDevice, commandPool: VkCommandPool, pAllocator: pointer) {.cdecl.}
    ](ctx.gpaAddr(ctx.instance, "vkDestroyCommandPool"))
  var allocInfo = VkCommandBufferAllocateInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    commandPool: batch.cmdPool,
    level: 0,
    commandBufferCount: 1
  )
  var cmdBuf: VkCommandBuffer
  check vkAllocateCommandBuffers(ctx.device, allocInfo.addr, cmdBuf.addr)
  # No VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT:
  #  a replayable command buffer resubmits verbatim, one submit per replay.
  var beginInfo = VkCommandBufferBeginInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
  )
  check vkBeginCommandBuffer(cmdBuf, beginInfo.addr)
  for i, entry in batch.entries.mpairs:
    let pkey = (entry.entrypoint, entry.spv)
    if pkey notin batch.pipelines:
      batch.pipelines[pkey] = createBatchPipeline(batch, entry)
    entry.pipelineState = batch.pipelines[pkey].handle
    if i > 0:
      # Vulkan gives no ordering guarantee between dispatches in one command buffer.
      # Barrier insertion makes each dispatch read the previous one's writes.
      var memBarrier = VkMemoryBarrier(
        sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        srcAccessMask: VK_ACCESS_SHADER_WRITE_BIT,
        dstAccessMask: VK_ACCESS_SHADER_READ_BIT or VK_ACCESS_SHADER_WRITE_BIT)
      vkCmdPipelineBarrier(cmdBuf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, memBarrier.addr, 0, nil, 0, nil)
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, entry.pipelineState)
    if entry.scalars.len > 0:
      vkCmdPushConstants(cmdBuf, batch.pipelines[pkey].layout,
                         uint32(VK_SHADER_STAGE_COMPUTE_BIT),
                         0, uint32(entry.scalars.len), entry.scalars[0].addr)
    vkCmdDispatch(cmdBuf, uint32(entry.fleet.x), uint32(entry.fleet.y),
                  uint32(entry.fleet.z))
  check vkEndCommandBuffer(cmdBuf)
  batch.recording = cmdBuf

proc replay*(batch: VulkanBatch) =
  if batch.recording == nil:
    raiseAssert("replay: finalize the batch first")
  let ctx = batch.ctx
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
  var fence: VkFence
  var fci = VkFenceCreateInfo(sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
  check vkCreateFence(ctx.device, fci.addr, nil, fence.addr)
  var submitInfo = VkSubmitInfo(
    sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
    commandBufferCount: 1,
    pCommandBuffers: batch.recording.addr
  )
  check vkQueueSubmit(ctx.queue, 1, submitInfo.addr, fence)
  check vkWaitForFences(ctx.device, 1, fence.addr, VK_TRUE, uint64.high)
  vkDestroyFence(ctx.device, fence, nil)
