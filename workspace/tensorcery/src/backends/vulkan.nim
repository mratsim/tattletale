# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vulkan backend: device storage for TensorImpl[T, bkVulkan] and the batched submission path.
##
## One session owns one VulkanContext. The GLSL kernel sandbox (src/kernels/k_vulkan.nim) compiles to SPIR-V pipelines per kernel name.
##
## eval() encodes every pending entry into ONE command buffer. Dispatches run in record order,
## a full memory barrier separates them, one vkQueueSubmit waits on one fence.
##
## Scalar ruling: the scalar image stays byte-identical to the packed Nim Scalars
## object that the MSL twins setBytes.
##
## - std430 storage buffers preserve that image, std140 uniform blocks would pad
##   int[4] arrays to 16-byte strides
## - each entry's scalar image uploads into its own device buffer at record time
## - an upload maps host-visible memory and never submits
##
## Tensor buffers leak (tensorcery frees no device memory yet), scalar buffers free after the fence wait.

import std/tables
import std/strutils
import pkg/iface

import
  workspace/crucible/src/runtime/backends/vulkan_common,
  workspace/crucible/src/runtime/backends/vulkan_compile {.all.},
  workspace/crucible/src/runtime/backends/vulkan_descriptor_runtime,
  workspace/crucible/src/abis/vulkan_abi as vk,
  workspace/data_structures/src/small_seqs,
  workspace/tensorcery/src/raw/dtypes,
  workspace/tensorcery/src/raw/bridge,
  workspace/tensorcery/src/raw/impl,
  workspace/tensorcery/src/ctx,
  workspace/tensorcery/src/tape,
  workspace/tensorcery/src/kernels/k_add,
  workspace/tensorcery/src/kernels/k_mul,
  workspace/tensorcery/src/kernels/k_mul2,
  workspace/tensorcery/src/kernels/k_fill,
  workspace/tensorcery/src/kernels/k_gather,
  workspace/tensorcery/src/kernels/k_layernorm,
  workspace/tensorcery/src/kernels/k_gelu,
  workspace/tensorcery/src/kernels/k_softmax,
  workspace/tensorcery/src/kernels/k_crossentropy,
  workspace/tensorcery/src/kernels/k_where,
  workspace/tensorcery/src/kernels/k_sgd_adam,
  workspace/tensorcery/src/kernels/k_vulkan

const
  vkStructureTypeMemoryBarrier = vk.VkStructureType(7)
    ## VkStructureType value of VkMemoryBarrier, absent from the vendored ABI enum.
  stageCompute = 1'u32 shl 11
    ## VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT.
  accessShaderRead = 1'u32 shl 5
    ## VK_ACCESS_SHADER_READ_BIT.
  accessShaderWrite = 1'u32 shl 6
    ## VK_ACCESS_SHADER_WRITE_BIT.
  workgroupSize* = 256
    ## Threads per workgroup of the elementwise dispatches, baked into the GLSL sources.
  maxVulkanBindings = 5
    ## Descriptor staging cap: today's largest binding count, one buffer slot set
    ## plus the scalar image. Raises at dispatch when a kernel exceeds it.

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

proc drainVulkan(ctx: Ctx) {.nimcall.}
  ## Forward: ensureVulkan wires the drain before its body.

const kernelNames: array[KernelId, string] =
  [opAdd: "k_add", opMul: "k_mul", opMul2: "k_mul2", opFill: "k_fill",
   opReshape: "", opPermute: "", opSlice: "",
   opGather: "k_gather", opLayerNorm: "k_layernorm", opGelu: "k_gelu",
   opSoftmax: "k_softmax", opCrossEntropy: "k_crossentropy", opWhere: "k_where",
   opSgd: "", opAdam: "",
   opSoftmaxBwd: "", opGeluBwd: "", opLayerNormBwdDx: "", opLayerNormBwdW: "",
   opLayerNormBwdB: "", opCrossEntropyBwd: "", opGatherBwd: "", opSliceBwd: ""]

const kernelGlsl: array[KernelId, string] =
  [opAdd: kAddGlsl, opMul: kMulGlsl, opMul2: kMul2Glsl, opFill: kFillGlsl,
   opReshape: "", opPermute: "", opSlice: "",
   opGather: kGatherGlsl, opLayerNorm: kLayerNormGlsl, opGelu: kGeluGlsl,
   opSoftmax: kSoftmaxGlsl, opCrossEntropy: kCrossEntropyGlsl, opWhere: kWhereGlsl,
   opSgd: "", opAdam: "",
   opSoftmaxBwd: "", opGeluBwd: "", opLayerNormBwdDx: "", opLayerNormBwdW: "",
   opLayerNormBwdB: "", opCrossEntropyBwd: "", opGatherBwd: "", opSliceBwd: ""]

var
  session: VulkanContext
  pipelines: Table[string, PipeBundle]
  setLayouts: Table[int, vk.VkDescriptorSetLayout]
  deviceBuffers: Table[uint64, VulkanBuffer]
    ## Every tensor buffer by its VkBuffer address, so readback reaches the memory handle
    ## behind the ABI's private fields.
  scalarBufs: Table[pointer, Table[int, VulkanBuffer]]
    ## Per context, one scalar-image buffer per recorded op id, uploaded at record time and freed after the drain's fence wait.
  submissionCount*: int
    ## Vulkan submissions since the last reset, one per submitted command buffer.
    ## Tests reset it and assert the exact count.

proc resetSubmissionCount*() =
  submissionCount = 0

# Session setup
# ------------------------------------------------------------

proc load(name: cstring): pointer =
  ## Loads one device-level Vulkan function through vkGetInstanceProcAddr.
  session.gpaAddr(session.instance, name)

proc addressSlots(kernel: KernelId): int =
  ## Address slots at the head of one recorded op's blob. The scalar image occupies the trailing bytes.
  case kernel
  of opAdd: addBufferSlots
  of opMul: mulBufferSlots
  of opMul2: mul2BufferSlots
  of opFill: fillBufferSlots
  of opGather: gatherBufferSlots
  of opLayerNorm: layerNormBufferSlots
  of opGelu: geluBufferSlots
  of opSoftmax: softmaxBufferSlots
  of opCrossEntropy: crossEntropyBufferSlots
  of opWhere: whereBufferSlots
  else: 0

proc cachePipeline(kernel: KernelId): PipeBundle =
  ## Compiles the kernel's GLSL source and builds the compute pipeline on first use.
  let name = kernelNames[kernel]
  # one source of truth for the workgroup size: it interpolates into the GLSL and the grid math
  let glsl = kernelGlsl[kernel] % $workgroupSize
  doAssert glsl.len > 0, "vulkan: kernel " & $kernel & " has no GLSL source"
  if not pipelines.hasKey(name):
    let spirv = compileGlslToSpirvViaGlslang(glsl, name)
    let bindings = uint32(addressSlots(kernel) + 1)
    if not setLayouts.hasKey(int(bindings)):
      setLayouts[int(bindings)] = session.createDescriptorSetLayout(bindings)
    let setLayout = setLayouts[int(bindings)]
    let shaderModule = session.createShaderModule(spirv)
    let vkCreatePipelineLayout = cast[
      proc(device: vk.VkDevice, pCI: ptr vk.VkPipelineLayoutCreateInfo, pAlloc: pointer, pLayout: ptr vk.VkPipelineLayout): vk.VkResult {.cdecl.}
    ](load("vkCreatePipelineLayout"))
    let vkCreateComputePipelines = cast[
      proc(device: vk.VkDevice, cache: vk.VkPipelineCache, count: uint32, pCIs: ptr vk.VkComputePipelineCreateInfo, pAlloc: pointer, pPipelines: ptr vk.VkPipeline): vk.VkResult {.cdecl.}
    ](load("vkCreateComputePipelines"))
    var plCI = vk.VkPipelineLayoutCreateInfo(
      sType: vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      setLayoutCount: 1)
    plCI.pSetLayouts = setLayout.unsafeAddr
    var playout: vk.VkPipelineLayout
    check vkCreatePipelineLayout(session.device, plCI.addr, nil, playout.addr)
    var stageInfo = vk.VkPipelineShaderStageCreateInfo(
      sType: vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      stage: vk.VK_SHADER_STAGE_COMPUTE_BIT,
      module: shaderModule,
      pName: name)
    var cpCI = vk.VkComputePipelineCreateInfo(
      sType: vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      stage: stageInfo,
      layout: playout)
    var phandle: vk.VkPipeline
    check vkCreateComputePipelines(session.device, session.pipelineCache, 1, cpCI.addr, nil, phandle.addr)
    session.destroyShaderModule(shaderModule)
    pipelines[name] = PipeBundle(handle: phandle, layout: playout, setLayout: setLayout)
  pipelines[name]

proc ensureVulkan*(ctx: Ctx) =
  ## One-time session setup plus ctx wiring, idempotent.
  if session.isNil:
    session = newVulkanContext()
  ctx.drains[bkVulkan] = drainVulkan

# Device storage
# ------------------------------------------------------------

proc allocDeviceMem*(nbytes: int): DeviceHandle =
  ## Allocates a Vulkan buffer, returned as an opaque handle and registered for readback.
  var buf = session.allocBuffer(nbytes)
  let key = cast[uint64](buf.handle)
  deviceBuffers[key] = buf
  DeviceHandle(key)

proc allocSharedMem*(nbytes, numel: int): SharedMemory =
  ## Allocates a Vulkan buffer as a SharedMemory handle, contents uninitialized.
  SharedMemory(handle: allocDeviceMem(nbytes), seqNo: 0, numel: numel)

proc uploadDeviceMem*(mem: DeviceHandle, data: pointer, nbytes: int) =
  ## Copies host bytes into a device buffer through its host-visible mapping.
  ## An upload, never a submission.
  var buf = deviceBuffers[uint64(mem)]
  session.writeBuffer(buf, data, nbytes)

# Drain: one command buffer, one submission
# ------------------------------------------------------------

proc readAddr(blob: openArray[byte], off: int): vk.VkBuffer =
  ## Reads one 8-byte device address slot out of a recorded blob.
  var v: uint64
  copyMem(v.addr, unsafeAddr blob[off], 8)
  cast[vk.VkBuffer](v)

proc hasPendingAfter(ctx: Ctx, backend: BackendKind, opId: int): bool =
  ## Returns true when a later pending entry of `backend` still dispatches.
  for entry in ctx.record:
    if entry.backend == backend and entry.id > opId and
        ctx.isPending(entry.id) and not isShapeOp(entry.kernel):
      return true

proc bindAndDispatch(cmdBuf: vk.VkCommandBuffer, entry: OpEntry, slots: int,
    scalarBuf: VulkanBuffer, pipe: PipeBundle, pools: var seq[vk.VkDescriptorPool]) =
  ## Binds one recorded op: `slots` buffer bindings from the blob's leading address slots
  ## and the scalar-image buffer as the last binding, then dispatches. The pool joins
  ## `pools` for destruction after the fence wait.
  let vkUpdateDescriptorSets = cast[
    proc(device: vk.VkDevice, count: uint32, pWrites: ptr vk.VkWriteDescriptorSet, copyCount: uint32, pCopies: ptr vk.VkWriteDescriptorSet) {.cdecl.}
  ](load("vkUpdateDescriptorSets"))
  let vkCmdBindPipeline = cast[
    proc(cb: vk.VkCommandBuffer, bindPoint: vk.VkPipelineBindPoint, pipeline: vk.VkPipeline) {.cdecl.}
  ](load("vkCmdBindPipeline"))
  let vkCmdBindDescriptorSets = cast[
    proc(cb: vk.VkCommandBuffer, bindPoint: vk.VkPipelineBindPoint, layout: vk.VkPipelineLayout, firstSet: uint32, setCount: uint32, pSets: ptr vk.VkDescriptorSet, dynOffCount: uint32, pDynOffs: ptr uint32) {.cdecl.}
  ](load("vkCmdBindDescriptorSets"))
  let vkCmdDispatch = cast[
    proc(cb: vk.VkCommandBuffer, gx: uint32, gy: uint32, gz: uint32) {.cdecl.}
  ](load("vkCmdDispatch"))

  let bindings = slots + 1
  let pool = session.createDescriptorPool(uint32(bindings))
  pools.add pool
  let ds = session.allocateDescriptorSet(pool, pipe.setLayout)
  var infos: array[maxVulkanBindings, vk.VkDescriptorBufferInfo]
  var writes: array[maxVulkanBindings, vk.VkWriteDescriptorSet]
  for i in 0 ..< slots:
    infos[i] = vk.VkDescriptorBufferInfo(
      buffer: readAddr(entry.blob, i * 8), offset: 0, range: vk.VkDeviceSize.high)
    writes[i] = vk.VkWriteDescriptorSet(
      sType: vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      dstSet: ds, dstBinding: uint32(i), dstArrayElement: 0,
      descriptorCount: 1, descriptorType: vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      pBufferInfo: infos[i].addr)
  infos[slots] = vk.VkDescriptorBufferInfo(
    buffer: scalarBuf.handle, offset: 0, range: vk.VkDeviceSize(scalarBuf.size))
  writes[slots] = vk.VkWriteDescriptorSet(
    sType: vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    dstSet: ds, dstBinding: uint32(slots), dstArrayElement: 0,
    descriptorCount: 1, descriptorType: vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    pBufferInfo: infos[slots].addr)
  vkUpdateDescriptorSets(session.device, uint32(bindings), writes[0].addr, 0, nil)

  vkCmdBindPipeline(cmdBuf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe.handle)
  vkCmdBindDescriptorSets(cmdBuf, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipe.layout, 0, 1, ds.addr, 0, nil)
  vkCmdDispatch(cmdBuf, uint32(entry.cfg.gridX), uint32(entry.cfg.gridY), uint32(entry.cfg.gridZ))

proc drainVulkan(ctx: Ctx) {.nimcall.} =
  ## One submission for the pending entries: one command buffer, one dispatch per
  ## entry in record order, a full memory barrier between dispatches, one submit,
  ## one fence wait. Scalar-image buffers free once the fence wait consumed them.
  let vkAllocateCommandBuffers = cast[
    proc(device: vk.VkDevice, pAllocateInfo: ptr vk.VkCommandBufferAllocateInfo, pCommandBuffers: ptr vk.VkCommandBuffer): vk.VkResult {.cdecl.}
  ](load("vkAllocateCommandBuffers"))
  let vkBeginCommandBuffer = cast[
    proc(cb: vk.VkCommandBuffer, pBeginInfo: ptr vk.VkCommandBufferBeginInfo): vk.VkResult {.cdecl.}
  ](load("vkBeginCommandBuffer"))
  let vkCmdPipelineBarrier = cast[
    proc(cb: vk.VkCommandBuffer, srcStage, dstStage: uint32, depFlags: uint32, memBarriers: uint32, pMemBarriers: ptr VkMemoryBarrier, bufBarriers: uint32, pBufBarriers: pointer, imgBarriers: uint32, pImgBarriers: pointer) {.cdecl.}
  ](load("vkCmdPipelineBarrier"))
  let vkEndCommandBuffer = cast[
    proc(cb: vk.VkCommandBuffer): vk.VkResult {.cdecl.}
  ](load("vkEndCommandBuffer"))
  let vkCreateFence = cast[
    proc(device: vk.VkDevice, pCI: ptr vk.VkFenceCreateInfo, pAlloc: pointer, fence: ptr vk.VkFence): vk.VkResult {.cdecl.}
  ](load("vkCreateFence"))
  let vkQueueSubmit = cast[
    proc(queue: vk.VkQueue, submitCount: uint32, pSubmit: ptr vk.VkSubmitInfo, fence: vk.VkFence): vk.VkResult {.cdecl.}
  ](load("vkQueueSubmit"))
  let vkWaitForFences = cast[
    proc(device: vk.VkDevice, fenceCount: uint32, pFences: ptr vk.VkFence, waitAll: vk.VkBool32, timeout: uint64): vk.VkResult {.cdecl.}
  ](load("vkWaitForFences"))
  let vkDestroyFence = cast[
    proc(device: vk.VkDevice, fence: vk.VkFence, pAlloc: pointer) {.cdecl.}
  ](load("vkDestroyFence"))
  let vkFreeCommandBuffers = cast[
    proc(device: vk.VkDevice, pool: vk.VkCommandPool, count: uint32, pCBs: ptr vk.VkCommandBuffer) {.cdecl.}
  ](load("vkFreeCommandBuffers"))

  var cmdBuf: vk.VkCommandBuffer
  var allocInfo = vk.VkCommandBufferAllocateInfo(
    sType: vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    commandPool: session.commandPool,
    level: 0,
    commandBufferCount: 1)
  check vkAllocateCommandBuffers(session.device, allocInfo.addr, cmdBuf.addr)
  var beginInfo = vk.VkCommandBufferBeginInfo(
    sType: vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    flags: vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
  check vkBeginCommandBuffer(cmdBuf, beginInfo.addr)

  var barrier = VkMemoryBarrier(
    sType: vkStructureTypeMemoryBarrier,
    srcAccessMask: accessShaderWrite,
    dstAccessMask: accessShaderRead or accessShaderWrite)

  var pools: seq[vk.VkDescriptorPool]
  for entry in ctx.record:
    if entry.backend != bkVulkan or not ctx.isPending(entry.id):
      continue
    if isShapeOp(entry.kernel):
      # pure metadata, never dispatches
      continue
    let pipe = cachePipeline(entry.kernel)
    let ctxPtr = cast[pointer](ctx)
    doAssert scalarBufs.hasKey(ctxPtr) and scalarBufs[ctxPtr].hasKey(entry.id),
        "vulkan drain: op " & $entry.id & " (" & $entry.kernel & ") has no scalar-image buffer"
    let scalarBuf = scalarBufs[ctxPtr][entry.id]
    case entry.kernel
    of opAdd:
      bindAndDispatch(cmdBuf, entry, addBufferSlots, scalarBuf, pipe, pools)
    of opMul:
      bindAndDispatch(cmdBuf, entry, mulBufferSlots, scalarBuf, pipe, pools)
    of opMul2:
      bindAndDispatch(cmdBuf, entry, mul2BufferSlots, scalarBuf, pipe, pools)
    of opFill:
      bindAndDispatch(cmdBuf, entry, fillBufferSlots, scalarBuf, pipe, pools)
    of opGather:
      bindAndDispatch(cmdBuf, entry, gatherBufferSlots, scalarBuf, pipe, pools)
    of opLayerNorm:
      bindAndDispatch(cmdBuf, entry, layerNormBufferSlots, scalarBuf, pipe, pools)
    of opGelu:
      bindAndDispatch(cmdBuf, entry, geluBufferSlots, scalarBuf, pipe, pools)
    of opSoftmax:
      bindAndDispatch(cmdBuf, entry, softmaxBufferSlots, scalarBuf, pipe, pools)
    of opCrossEntropy:
      bindAndDispatch(cmdBuf, entry, crossEntropyBufferSlots, scalarBuf, pipe, pools)
    of opSgd, opAdam, opSoftmaxBwd, opGeluBwd,
       opLayerNormBwdDx, opLayerNormBwdW, opLayerNormBwdB, opCrossEntropyBwd,
       opGatherBwd, opSliceBwd:
      doAssert false, "vulkan drain: " & $entry.kernel & " has no Vulkan kernel"
    of opWhere:
      bindAndDispatch(cmdBuf, entry, whereBufferSlots, scalarBuf, pipe, pools)
    of opReshape, opPermute, opSlice:
      # pure metadata, skipped above
      discard
    if hasPendingAfter(ctx, bkVulkan, entry.id):
      vkCmdPipelineBarrier(cmdBuf, stageCompute, stageCompute, 0,
                           1, barrier.addr, 0, nil, 0, nil)

  check vkEndCommandBuffer(cmdBuf)
  var fence: vk.VkFence
  var fci = vk.VkFenceCreateInfo(sType: vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
  check vkCreateFence(session.device, fci.addr, nil, fence.addr)
  var submitInfo = vk.VkSubmitInfo(
    sType: vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
    commandBufferCount: 1,
    pCommandBuffers: cmdBuf.addr)
  check vkQueueSubmit(session.queue, 1, submitInfo.addr, fence)
  submissionCount.inc
  check vkWaitForFences(session.device, 1, fence.addr, vk.VK_TRUE, uint64.high)
  vkDestroyFence(session.device, fence, nil)
  vkFreeCommandBuffers(session.device, session.commandPool, 1, cmdBuf.addr)
  for pool in pools:
    session.destroyDescriptorPool(pool)
  let ctxPtr = cast[pointer](ctx)
  if scalarBufs.hasKey(ctxPtr):
    var spent = scalarBufs[ctxPtr]
    scalarBufs.del(ctxPtr)
    for buf in spent.mvalues:
      var b = buf
      b.dealloc()

# Record helpers
# ------------------------------------------------------------

proc elementwiseCfg(n: int): LaunchCfg =
  LaunchCfg(
    gridX: int32((n + workgroupSize - 1) div workgroupSize),
    gridY: 1, gridZ: 1,
    blkX: workgroupSize, blkY: 1, blkZ: 1)

proc padTo4(v: SmallSeq[4, int32], pad: int32): array[4, int32] =
  ## Pads to 4 axes with `pad`: shapes pad with 1s (padded axes contribute nothing to the divmod chain), strides with 0s.
  for i in 0 ..< 4:
    result[i] = if i < int(v.len): v[i] else: pad

proc toVulkanTensorImpl*[T](ctx: Ctx, data: openArray[T], shape: openArray[int], requiresGrad = false): TensorImpl[T, bkVulkan] =
  ## Builds a Vulkan tensor from host data, contiguous, uploaded at construction.
  let meta = contiguousMeta(shape)
  result = TensorImpl[T, bkVulkan](
    ctx: ctx,
    tid: ctx.newTid(),
    requiresGrad: requiresGrad,
    shape: meta.shape,
    strides: meta.strides,
    mem: SharedMemory(handle: allocDeviceMem(data.len * sizeof(T)),
        seqNo: 0, numel: data.len))
  uploadDeviceMem(result.mem.handle, unsafeAddr data[0], data.len * sizeof(T))

proc record[T](s: TensorImpl[T, bkVulkan], kernel: KernelId, blob: sink seq[byte], cfg: LaunchCfg,
    outMem: SharedMemory, outShape: SmallSeq[4, int32], requiresGrad, isGrad: bool,
    inA, inB, inC: int, scalars: pointer, scalarBytes: int): TensorImpl[T, bkVulkan] =
  ## Enqueues one record entry, uploads the scalar image into its device buffer
  ## and wraps the freshly allocated output storage.
  ##
  ## The storage epoch advances to the entry, so reads of the output materialize it first.
  ## The output registers in the tape registry.
  ##
  ## - every blob address's owner must stay alive until the step boundary
  ## - grad-segment intermediates have no caller-side owner and depend on the registry
  let outTid = s.ctx.newTid()
  var scalarBuf = session.allocBuffer(scalarBytes)
  session.writeBuffer(scalarBuf, scalars, scalarBytes)
  let opId = s.ctx.enqueue(kernel, blob, cfg, bkVulkan, outTid, inA, inB, inC, 0, isGrad)
  scalarBufs.mgetOrPut(cast[pointer](s.ctx), initTable[int, VulkanBuffer]())[opId] = scalarBuf
  outMem.seqNo = opId
  result = TensorImpl[T, bkVulkan](
    ctx: s.ctx,
    tid: outTid,
    requiresGrad: requiresGrad,
    shape: outShape,
    strides: contiguousStrides(outShape),
    mem: outMem)
  tape.register(s.ctx, outTid, result.to(AnyTensorImpl))

proc toErased*[T](s: TensorImpl[T, bkVulkan]): AnyTensorImpl =
  ## Returns the erased face over `s`.
  ##
  ## Contract: the vtable's mixins resolve at the outermost instantiation scope.
  ## tensors.nim re-exports the typed conformant procs, so module users satisfy
  ## the interface conformance check.
  s.to(AnyTensorImpl)

# Microgpt forward op set, Vulkan backend
# ------------------------------------------------------------

proc add*[T](s: TensorImpl[T, bkVulkan], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased add: packs the op's args into one blob at enqueue, uploads the scalar
  ## image to its device buffer, appends one record entry, returns the erased
  ## output tensor. The output requires grad when either input does.
  let o = other.to(TensorImpl[T, bkVulkan])
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let sc = AddScalars(
    aOff: int32(s.offset),
    bOff: int32(o.offset),
    shape: padTo4(s.shape, 1),
    sa: padTo4(s.strides, 0),
    sb: padTo4(o.strides, 0),
    n: int32(n))
  let blob = packAddArgs(AddArgs(
    outAddr: cast[uint64](outMem.handle),
    aAddr: cast[uint64](s.mem.handle),
    bAddr: cast[uint64](o.mem.handle),
    scalars: sc))
  s.record(opAdd, blob, elementwiseCfg(n), outMem, s.shape,
      s.requiresGrad or o.requiresGrad, false, s.tid, o.tid, 0,
      sc.addr, sizeof(AddScalars)).to(AnyTensorImpl)

proc readbackBytes*[T](s: TensorImpl[T, bkVulkan]): seq[byte] =
  ## Returns:
  ## - the tensor as host bytes in logical order, last axis fastest
  ##
  ## A view gathers through its strides and offset. Reading a pending tensor
  ## evals the context first (the MLX rule).
  if s.ctx.isPending(s.mem.seqNo):
    s.ctx.eval()
  var buf = deviceBuffers[uint64(s.mem.handle)]
  let raw = readBuffer[T](session, buf)
  let base = raw[s.offset].unsafeAddr
  stridedBytes[T](base, s.shape, s.strides)

# Refusals: the Vulkan backend ships the forward op set only
# ------------------------------------------------------------

proc addNoGrad*[T](s: TensorImpl[T, bkVulkan], other: AnyTensorImpl): AnyTensorImpl =
  ## Refusal: the Vulkan backend ships the forward op set. Grad-chain composition is CPU and Metal only.
  doAssert false,
      "vulkan: addNoGrad(tensor " & $s.tid & ", tensor " & $other.getTid() & ") refused, the Vulkan backend ships the forward op set only"

proc mulNoGrad*[T](s: TensorImpl[T, bkVulkan], other: AnyTensorImpl): AnyTensorImpl =
  ## Refusal: the Vulkan backend ships the forward op set. Grad-chain composition is CPU and Metal only.
  doAssert false,
      "vulkan: mulNoGrad(tensor " & $s.tid & ", tensor " & $other.getTid() & ") refused, the Vulkan backend ships the forward op set only"

proc mulScalarNoGrad*[T](s: TensorImpl[T, bkVulkan], scalar: float64): AnyTensorImpl =
  ## Refusal: the Vulkan backend ships the forward op set. Grad-chain composition is CPU and Metal only.
  doAssert false,
      "vulkan: mulScalarNoGrad(tensor " & $s.tid & ", scalar " & $scalar & ") refused, the Vulkan backend ships the forward op set only"

proc mul*[T](s: TensorImpl[T, bkVulkan], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased tensor product: packs the op's args into one blob at enqueue,
  ## appends one record entry, returns the erased output tensor.
  ## The output requires grad when either input does.
  let o = other.to(TensorImpl[T, bkVulkan])
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let sc = Mul2Scalars(
    aOff: int32(s.offset),
    bOff: int32(o.offset),
    shape: padTo4(s.shape, 1),
    sa: padTo4(s.strides, 0),
    sb: padTo4(o.strides, 0),
    n: int32(n))
  let blob = packMul2Args(Mul2Args(
    outAddr: cast[uint64](outMem.handle),
    aAddr: cast[uint64](s.mem.handle),
    bAddr: cast[uint64](o.mem.handle),
    scalars: sc))
  s.record(opMul2, blob, elementwiseCfg(n), outMem, s.shape,
      s.requiresGrad or o.requiresGrad, false, s.tid, o.tid, 0,
      sc.addr, sizeof(Mul2Scalars)).to(AnyTensorImpl)

proc mulScalar*[T](s: TensorImpl[T, bkVulkan], scalar: float64): AnyTensorImpl =
  ## Erased scalar multiply: the scalar narrows to the storage element type at pack time.
  ## The output requires grad when the input does.
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let sc = MulScalars(
    aOff: int32(s.offset),
    shape: padTo4(s.shape, 1),
    sa: padTo4(s.strides, 0),
    scalar: float32(scalar),
    n: int32(n))
  let blob = packMulArgs(MulArgs(
    outAddr: cast[uint64](outMem.handle),
    aAddr: cast[uint64](s.mem.handle),
    scalars: sc))
  s.record(opMul, blob, elementwiseCfg(n), outMem, s.shape,
      s.requiresGrad, false, s.tid, 0, 0,
      sc.addr, sizeof(MulScalars)).to(AnyTensorImpl)

proc onesNoGrad*[T](s: TensorImpl[T, bkVulkan]): AnyTensorImpl =
  ## Erased ones fill: one record entry over a fresh output storage, grad flag off.
  ## backward() seeds the cotangent map from it.
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let sc = FillScalars(n: int32(n))
  let blob = packFillArgs(FillArgs(
    outAddr: cast[uint64](outMem.handle),
    scalars: sc))
  s.record(opFill, blob, elementwiseCfg(n), outMem, s.shape,
      false, true, 0, 0, 0,
      sc.addr, sizeof(FillScalars)).to(AnyTensorImpl)

proc gather*[T](s: TensorImpl[T, bkVulkan], indices: AnyTensorImpl): AnyTensorImpl =
  ## Erased embedding gather: one embedding vector per index over the rank-2 table.
  ## The output requires grad when the table does. The int32 indices never do.
  ##
  ## Raises AssertionDefect for a non rank-2 table or any non-contiguous operand.
  let idx = indices.to(TensorImpl[int32, bkVulkan])
  doAssert int(s.shape.len) == 2,
      "gather: the table must be rank 2 (vocabulary, embedding dim), got rank " & $s.shape.len
  doAssert isContiguous(s.shape, s.strides), "gather: non-contiguous table refused"
  doAssert isContiguous(idx.shape, idx.strides), "gather: non-contiguous index tensor refused"
  let e = int32(s.shape[int(s.shape.len) - 1])
  let n = idx.getNumel() * int(e)
  let outMem = allocSharedMem(n * sizeof(T), n)
  let sc = GatherScalars(
    wOff: int32(s.offset), idxOff: int32(idx.offset), e: e, n: int32(n))
  let blob = packGatherArgs(GatherArgs(
    outAddr: cast[uint64](outMem.handle),
    wAddr: cast[uint64](s.mem.handle),
    idxAddr: cast[uint64](idx.mem.handle),
    scalars: sc))
  s.record(opGather, blob, elementwiseCfg(n), outMem, appendDim(idx.shape, e),
      s.requiresGrad, false, s.tid, idx.tid, 0,
      sc.addr, sizeof(GatherScalars)).to(AnyTensorImpl)

proc layernorm*[T](s: TensorImpl[T, bkVulkan], w, b: AnyTensorImpl, eps: float64): AnyTensorImpl =
  ## Erased last-axis layernorm over `s` with the learned scale `w` and shift `b`.
  ## The output requires grad when any operand does.
  ##
  ## Raises AssertionDefect when the scale or shift does not hold one element per
  ## normalized axis or is non-contiguous.
  let wImpl = w.to(TensorImpl[T, bkVulkan])
  let bImpl = b.to(TensorImpl[T, bkVulkan])
  let d = int(s.shape[int(s.shape.len) - 1])
  let rows = s.getNumel() div d
  doAssert wImpl.getNumel() == d and bImpl.getNumel() == d,
      "layernorm: scale and shift must hold one element per normalized axis, " &
      $wImpl.getNumel() & " and " & $bImpl.getNumel() & " for axis size " & $d
  doAssert isContiguous(wImpl.shape, wImpl.strides) and isContiguous(bImpl.shape, bImpl.strides),
      "layernorm: non-contiguous scale or shift refused"
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let sc = LayerNormScalars(
    xOff: int32(s.offset), wOff: int32(wImpl.offset), bOff: int32(bImpl.offset),
    xShape: padTo4(s.shape, 1), xs: padTo4(s.strides, 0),
    d: int32(d), rows: int32(rows), xStep: int32(s.strides[int(s.shape.len) - 1]),
    eps: float32(eps))
  let blob = packLayerNormArgs(LayerNormArgs(
    outAddr: cast[uint64](outMem.handle),
    xAddr: cast[uint64](s.mem.handle),
    wAddr: cast[uint64](wImpl.mem.handle),
    bAddr: cast[uint64](bImpl.mem.handle),
    scalars: sc))
  s.record(opLayerNorm, blob, elementwiseCfg(rows), outMem, s.shape,
      s.requiresGrad or wImpl.requiresGrad or bImpl.requiresGrad, false,
      s.tid, wImpl.tid, bImpl.tid,
      sc.addr, sizeof(LayerNormScalars)).to(AnyTensorImpl)

proc gelu*[T](s: TensorImpl[T, bkVulkan]): AnyTensorImpl =
  ## Erased tanh-approximate gelu over `s`. The output requires grad when `s` does.
  ## The input stages through the BridgeTensor boundary into the kernel image.
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let staged = BridgeTensor[T](
    base: cast[uint64](s.mem.handle), offset: s.offset,
    shape: s.shape, strides: s.strides)
  let view = staged.toView()
  let sc = GeluScalars(aOff: view.off, n: int32(n))
  let blob = packGeluArgs(GeluArgs(
    outAddr: cast[uint64](outMem.handle),
    aAddr: view.base,
    scalars: sc))
  s.record(opGelu, blob, elementwiseCfg(n), outMem, s.shape,
      s.requiresGrad, false, s.tid, 0, 0,
      sc.addr, sizeof(GeluScalars)).to(AnyTensorImpl)

proc softmax*[T](s: TensorImpl[T, bkVulkan]): AnyTensorImpl =
  ## Erased last-axis softmax over `s`. The output requires grad when `s` does.
  let dAxis = int(s.shape[int(s.shape.len) - 1])
  let rows = s.getNumel() div dAxis
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let sc = SoftmaxScalars(
    xOff: int32(s.offset),
    xShape: padTo4(s.shape, 1), xs: padTo4(s.strides, 0),
    d: int32(dAxis), rows: int32(rows), xStep: int32(s.strides[int(s.shape.len) - 1]))
  let blob = packSoftmaxArgs(SoftmaxArgs(
    outAddr: cast[uint64](outMem.handle),
    xAddr: cast[uint64](s.mem.handle),
    scalars: sc))
  s.record(opSoftmax, blob, elementwiseCfg(rows), outMem, s.shape,
      s.requiresGrad, false, s.tid, 0, 0,
      sc.addr, sizeof(SoftmaxScalars)).to(AnyTensorImpl)

proc crossEntropy*[T](s: TensorImpl[T, bkVulkan], targets: AnyTensorImpl): AnyTensorImpl =
  ## Erased cross entropy: one loss per set of `v` logits over the last axis of `s`,
  ## the int32 `targets` name the class per set. The output requires grad when the logits do.
  ##
  ## Raises AssertionDefect for a non-contiguous target view.
  let tgt = targets.to(TensorImpl[int32, bkVulkan])
  doAssert isContiguous(tgt.shape, tgt.strides), "cross entropy: non-contiguous target view refused"
  let v = int(s.shape[int(s.shape.len) - 1])
  let rows = s.getNumel() div v
  let outMem = allocSharedMem(rows * sizeof(T), rows)
  let sc = CrossEntropyScalars(
    logitsOff: int32(s.offset), targetOff: int32(tgt.offset),
    logitsShape: padTo4(s.shape, 1), logitsStrides: padTo4(s.strides, 0),
    v: int32(v), rows: int32(rows), logitStep: int32(s.strides[int(s.shape.len) - 1]))
  let blob = packCrossEntropyArgs(CrossEntropyArgs(
    lossAddr: cast[uint64](outMem.handle),
    logitsAddr: cast[uint64](s.mem.handle),
    targetAddr: cast[uint64](tgt.mem.handle),
    scalars: sc))
  s.record(opCrossEntropy, blob, elementwiseCfg(rows), outMem, dropLastDim(s.shape),
      s.requiresGrad, false, s.tid, tgt.tid, 0,
      sc.addr, sizeof(CrossEntropyScalars)).to(AnyTensorImpl)

proc where*[T](s: TensorImpl[T, bkVulkan], cond, other: AnyTensorImpl): AnyTensorImpl =
  ## Erased elementwise select with `s` on the true branch and `other` on the false branch.
  ## The output requires grad when any operand does.
  ##
  ## Raises AssertionDefect on a shape mismatch, no broadcasting.
  let c = cond.to(TensorImpl[T, bkVulkan])
  let o = other.to(TensorImpl[T, bkVulkan])
  doAssert s.shape == c.shape and s.shape == o.shape,
      "where: shape mismatch " & $s.shape & " vs " & $c.shape & " vs " & $o.shape &
      " refused, no broadcasting"
  doAssert isContiguous(c.shape, c.strides) and isContiguous(s.shape, s.strides) and
      isContiguous(o.shape, o.strides),
      "where: non-contiguous operands refused, the kernel reads them linearly from their offsets"
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let sc = WhereScalars(
    condOff: int32(c.offset), aOff: int32(s.offset), bOff: int32(o.offset),
    n: int32(n))
  let blob = packWhereArgs(WhereArgs(
    outAddr: cast[uint64](outMem.handle),
    condAddr: cast[uint64](c.mem.handle),
    aAddr: cast[uint64](s.mem.handle),
    bAddr: cast[uint64](o.mem.handle),
    scalars: sc))
  s.record(opWhere, blob, elementwiseCfg(n), outMem, s.shape,
      s.requiresGrad or c.requiresGrad or o.requiresGrad, false,
      c.tid, s.tid, o.tid,
      sc.addr, sizeof(WhereScalars)).to(AnyTensorImpl)

proc sgdUpdate*[T](s: TensorImpl[T, bkVulkan], grad: AnyTensorImpl, lr: float64): void =
  ## Refusal: in-place optimizer updates stay on CPU and Metal.
  doAssert false,
      "vulkan: sgdUpdate(tensor " & $s.tid & ", tensor " & $grad.getTid() & ") refused, in-place optimizer updates have no Vulkan kernel"

proc adamUpdate*[T](s: TensorImpl[T, bkVulkan], grad, m, v: AnyTensorImpl, pack: AdamScalars): void =
  ## Refusal: in-place optimizer updates stay on CPU and Metal.
  doAssert false,
      "vulkan: adamUpdate(tensor " & $s.tid & ", tensor " & $grad.getTid() & ") refused, in-place optimizer updates have no Vulkan kernel"

proc softmaxBwdNoGrad*[T](s: TensorImpl[T, bkVulkan], g, p: AnyTensorImpl): AnyTensorImpl =
  ## Refusal: fused backward kernels stay on CPU and Metal.
  doAssert false,
      "vulkan: softmaxBwdNoGrad(tensor " & $s.tid & ") refused, fused backward kernels have no Vulkan kernel"

proc geluBwdNoGrad*[T](s: TensorImpl[T, bkVulkan], g: AnyTensorImpl): AnyTensorImpl =
  ## Refusal: fused backward kernels stay on CPU and Metal.
  doAssert false,
      "vulkan: geluBwdNoGrad(tensor " & $s.tid & ") refused, fused backward kernels have no Vulkan kernel"

proc layernormBwdDxNoGrad*[T](s: TensorImpl[T, bkVulkan], g, w: AnyTensorImpl, eps: float64): AnyTensorImpl =
  ## Refusal: fused backward kernels stay on CPU and Metal.
  doAssert false,
      "vulkan: layernormBwdDxNoGrad(tensor " & $s.tid & ") refused, fused backward kernels have no Vulkan kernel"

proc layernormBwdWNoGrad*[T](s: TensorImpl[T, bkVulkan], g, x: AnyTensorImpl, eps: float64): AnyTensorImpl =
  ## Refusal: fused backward kernels stay on CPU and Metal.
  doAssert false,
      "vulkan: layernormBwdWNoGrad(tensor " & $s.tid & ") refused, fused backward kernels have no Vulkan kernel"

proc layernormBwdBNoGrad*[T](s: TensorImpl[T, bkVulkan], g: AnyTensorImpl): AnyTensorImpl =
  ## Refusal: fused backward kernels stay on CPU and Metal.
  doAssert false,
      "vulkan: layernormBwdBNoGrad(tensor " & $s.tid & ") refused, fused backward kernels have no Vulkan kernel"

proc crossEntropyBwdNoGrad*[T](s: TensorImpl[T, bkVulkan], g, targets: AnyTensorImpl): AnyTensorImpl =
  ## Refusal: fused backward kernels stay on CPU and Metal.
  doAssert false,
      "vulkan: crossEntropyBwdNoGrad(tensor " & $s.tid & ") refused, fused backward kernels have no Vulkan kernel"

proc gatherBwdNoGrad*[T](s: TensorImpl[T, bkVulkan], g, idx: AnyTensorImpl): AnyTensorImpl =
  ## Refusal: fused backward kernels stay on CPU and Metal.
  doAssert false,
      "vulkan: gatherBwdNoGrad(tensor " & $s.tid & ") refused, fused backward kernels have no Vulkan kernel"

proc sliceBwdNoGrad*[T](s: TensorImpl[T, bkVulkan], g, view: AnyTensorImpl, start: SmallSeq[4, int32]): AnyTensorImpl =
  ## Refusal: fused backward kernels stay on CPU and Metal.
  doAssert false,
      "vulkan: sliceBwdNoGrad(tensor " & $s.tid & ") refused, fused backward kernels have no Vulkan kernel"
