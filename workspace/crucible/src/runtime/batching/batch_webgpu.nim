# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[strscans, strutils, tables]
import workspace/crucible/vendor/wgpu
import workspace/crucible/src/crucible_codegen
import workspace/crucible/src/runtime/backends/webgpu_runtime
import workspace/crucible/src/runtime/backends/runtime_utils
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/launch/launch_serialization

export engines_common.LaunchConfig, engines_common.dim3

# ############################################################
#
#                   WebGPU batch submission
#
# ############################################################

type
  SlotKind = enum
    skBuffer
    skWindow

  Slot = object
    case kind: SlotKind
    of skBuffer:
      buffer: int
    of skWindow:
      off, size: int

  BatchEntry = object
    module: WGPUShaderModule
    entryPoint: string
    pipelineState: WGPUComputePipeline
    fleet, squadron: Dim3
    borrowedBuffers: seq[WGPUBuffer]
    scalars: seq[byte]
    slots: seq[Slot]
    bindGroup: WGPUBindGroup

  WebgpuPipeline = object
    pipeline: WGPUComputePipeline
    pipelineLayout: WGPUPipelineLayout
    bindGroupLayout: WGPUBindGroupLayout

  WebgpuBatchObj = object
    device: WGPUDevice
    queue: WGPUQueue
    modules: Table[string, WGPUShaderModule]
    pipelines: Table[(WGPUShaderModule, string), WebgpuPipeline]
    entries: seq[BatchEntry]
    scalarBuffer: WGPUBuffer

  WebgpuBatch* = ref WebgpuBatchObj

const
  MinStorageBufferOffsetAlignment = 256
    ## WebGPU spec default: one storage binding offset is a multiple of minStorageBufferOffsetAlignment.
  MinStorageBindingSizeMultiple = 4
    ## a storage binding size is a multiple of 4 bytes

proc `=destroy`*(batch: var WebgpuBatchObj) =
  ## Releases the batch's own objects, never the borrowed handles.
  for m in batch.modules.values:
    wgpuShaderModuleRelease(m)
  for p in batch.pipelines.values:
    wgpuComputePipelineRelease(p.pipeline)
    wgpuPipelineLayoutRelease(p.pipelineLayout)
    wgpuBindGroupLayoutRelease(p.bindGroupLayout)
  for e in batch.entries:
    if e.bindGroup != nil:
      wgpuBindGroupRelease(e.bindGroup)
  if batch.scalarBuffer != nil:
    wgpuBufferRelease(batch.scalarBuffer)

proc parseShaderSquadron(wgsl: string): Dim3 =
  ## Extracts the shader-baked squadron from the WGSL preamble
  ## `@compute @workgroup_size(64, 8, 1)`, 1..3 literal args, missing default to 1.
  ## Returns (0, 0, 0) when absent or non-literal, run then fails squadron validation.
  const marker = "@workgroup_size("
  let i = wgsl.find(marker)
  if i < 0:
    return
  let tail = wgsl[i + marker.len .. ^1]
  var x, y, z: int
  if scanf(tail, "$s$i$s,$s$i$s,$s$i$s)", x, y, z) or
     scanf(tail, "$s$i$s,$s$i$s)", x, y) or
     scanf(tail, "$s$i)", x):
    if x <= 0:
      return
    result = Dim3(x: x, y: if y > 0: y else: 1,
                       z: if z > 0: z else: 1)

proc storeEntryArg(entry: var BatchEntry, v: auto, _: int) =
  when typeof(v) is WGPUBuffer:
    entry.borrowedBuffers.add v
    entry.slots.add Slot(kind: skBuffer, buffer: entry.borrowedBuffers.high)
  else:
    when typeof(v) is ptr|pointer:
      static:
        error "Pointer device function args are not supported, pass the caller's WGPUBuffer handle, got " & $typeof(v)
    const layout = computeDeviceLayout(typeof(v), cgWebGPU)
    let off = entry.scalars.len.round_multiple_up(MinStorageBufferOffsetAlignment)
    # wgpuQueueWriteBuffer requires a multiple-of-4 size, every scalar window is padded to it
    let size = layout.size.round_multiple_up(MinStorageBindingSizeMultiple)
    entry.scalars.setLen(off + size)
    var w = off
    entry.scalars.packFrom(w, v, cgWebGPU)
    entry.slots.add Slot(kind: skWindow, off: off, size: size)

proc compilePipeline(batch: WebgpuBatch, entry: BatchEntry): WebgpuPipeline =
  let n = entry.slots.len
  var bglEntries = newSeq[WGPUBindGroupLayoutEntry](n)
  for i in 0 ..< n:
    var layout = WGPUBufferBindingLayout(minBindingSize: 0)
    if entry.slots[i].kind == skWindow:
      layout.`type` = WGPUBufferBindingTypeReadOnlyStorage
      layout.minBindingSize = entry.slots[i].size.csize_t
    else:
      layout.`type` = WGPUBufferBindingTypeStorage
    bglEntries[i] = WGPUBindGroupLayoutEntry(
      binding: i.cuint,
      visibility: WGPUShaderStageCompute,
      buffer: layout)
  var bglDesc = WGPUBindGroupLayoutDescriptor(entryCount: n.csize_t, entries: bglEntries[0].addr)
  let bgl = wgpuDeviceCreateBindGroupLayout(batch.device, addr bglDesc)
  if bgl == nil:
    raiseAssert("WebGPU batch: failed to create bind group layout")
  var plDesc = WGPUPipelineLayoutDescriptor(bindGroupLayoutCount: 1, bindGroupLayouts: bgl.addr)
  let pl = wgpuDeviceCreatePipelineLayout(batch.device, addr plDesc)
  if pl == nil:
    raiseAssert("WebGPU batch: failed to create pipeline layout")
  var entryPtView = WGPUStringView(data: cstring(entry.entryPoint), length: entry.entryPoint.len.csize_t)
  var computeState = WGPUComputeState(module: entry.module, entryPoint: entryPtView)
  var cpDesc = WGPUComputePipelineDescriptor(layout: pl, compute: computeState)
  wgpuDevicePushErrorScope(batch.device, wgpuErrorFilterValidation)
  let pipeline = wgpuDeviceCreateComputePipeline(batch.device, addr cpDesc)
  var popScope = PopErrorScopeData(done: false)
  var popInfo = WGPUPopErrorScopeCallbackInfo(
    mode: wgpuCallbackModeAllowProcessEvents,
    callback: popErrorScopeCb,
    userdata1: popScope.addr,
    userdata2: nil)
  discard wgpuDevicePopErrorScope(batch.device, popInfo)
  while not popScope.done:
    if not wgpuDevicePoll(batch.device, true, nil):
      raiseAssert("WebGPU: wgpuDevicePoll failed")
  if popScope.errType != wgpuErrorTypeNoError:
    let uncaptured = takeUncapturedError()
    raiseAssert("WebGPU pipeline creation failed for " & entry.entryPoint & ": " & popScope.message &
      (if uncaptured.errType != wgpuErrorTypeNoError: " / " & uncaptured.message else: ""))
  if pipeline == nil:
    raiseAssert("WebGPU batch: failed to create compute pipeline")
  result = WebgpuPipeline(pipeline: pipeline, pipelineLayout: pl, bindGroupLayout: bgl)

proc newWebgpuBatch*(device: WGPUDevice, queue: WGPUQueue): WebgpuBatch =
  ## Borrowsdevice, queue and every bound buffer.
  WebgpuBatch(device: device, queue: queue)

proc recordGlobalFn*(batch: WebgpuBatch, wgsl: string, entryPoint: string, cfg: LaunchConfig, output: auto, args: tuple) =
  if batch.entries.len > 0 and batch.entries[0].bindGroup != nil:
    raiseAssert("recordGlobalFn: the batch is finalized, entries are frozen")
  if cfg.sharedMem != 0 or cfg.stream != 0:
    raiseAssert("WebGPU batch: launch-time sharedMem/stream are unsupported, they are Metal/CUDA features")
  if cfg.grid.x < 1 or cfg.grid.y < 1 or cfg.grid.z < 1:
    raiseAssert("WebGPU batch: cfg.grid must be >= 1 per axis, got " &
                $cfg.grid.x & "x" & $cfg.grid.y & "x" & $cfg.grid.z)
  let baked = parseShaderSquadron(wgsl)
  let squadron = if cfg.blk.x == 1 and cfg.blk.y == 1 and cfg.blk.z == 1:
              baked
            else:
              cfg.blk
  if baked.x == 0 or
     squadron.x != baked.x or squadron.y != baked.y or squadron.z != baked.z:
    raiseAssert("WebGPU batch: run squadron=" & $squadron.x & "x" & $squadron.y & "x" & $squadron.z &
         " != baked squadron " & $baked.x & "x" & $baked.y & "x" & $baked.z &
         " — launch config mismatch (squadron is shader-baked on WebGPU)")
  if wgsl notin batch.modules:
    var wgslView = WGPUStringView(data: cstring(wgsl), length: wgsl.len.csize_t)
    var shaderSource = WGPUShaderSourceWGSL(
      chain: WGPUChainedStruct(sType: WGPUSType_ShaderSourceWGSL, next: nil),
      code: wgslView
    )
    var shaderDesc = WGPUShaderModuleDescriptor(nextInChain: addr shaderSource)
    let module = wgpuDeviceCreateShaderModule(batch.device, addr shaderDesc)
    if module == nil:
      raiseAssert("WebGPU batch: failed to create shader module")
    batch.modules[wgsl] = module
  var entry = BatchEntry(module: batch.modules[wgsl], entryPoint: entryPoint,
                         fleet: cfg.grid, squadron: squadron)
  bindArgs(entry, output, args, storeEntryArg, WGPUBuffer, cgWebGPU)
  batch.entries.add entry

func getScalarSize*(batch: WebgpuBatch): int =
  for e in batch.entries:
    result = result + e.scalars.len.round_multiple_up(MinStorageBufferOffsetAlignment)

proc finalize*(batch: WebgpuBatch) =
  if batch.entries.len > 0 and batch.entries[0].bindGroup != nil:
    return
  if batch.entries.len == 0:
    raiseAssert("finalize: the batch has no entries")
  if batch.getScalarSize > 0:
    var desc = WGPUBufferDescriptor(
      usage: wgpuBufferUsageStorage or wgpuBufferUsageCopyDst,
      size: batch.getScalarSize.csize_t)
    batch.scalarBuffer = wgpuDeviceCreateBuffer(batch.device, addr desc)
    if batch.scalarBuffer == nil:
      raiseAssert("WebGPU batch: failed to create the scalar buffer")
  var windowBase = 0
  for e in batch.entries.mitems:
    # One entrypoint name exists in many modules, a pipeline is per (module, entrypoint) pair
    let pipelineKey = (e.module, e.entryPoint)
    if pipelineKey notin batch.pipelines:
      batch.pipelines[pipelineKey] = compilePipeline(batch, e)
    e.pipelineState = batch.pipelines[pipelineKey].pipeline
    if e.scalars.len > 0:
      wgpuQueueWriteBuffer(batch.queue, batch.scalarBuffer, windowBase.uint64,
                           e.scalars[0].addr, e.scalars.len.csize_t)
    var bgEntries = newSeq[WGPUBindGroupEntry](e.slots.len)
    for i in 0 ..< e.slots.len:
      case e.slots[i].kind
      of skBuffer:
        bgEntries[i] = WGPUBindGroupEntry(
          binding: i.cuint,
          buffer: e.borrowedBuffers[e.slots[i].buffer],
          size: wgpuBufferGetSize(e.borrowedBuffers[e.slots[i].buffer]).csize_t)
      of skWindow:
        bgEntries[i] = WGPUBindGroupEntry(
          binding: i.cuint,
          buffer: batch.scalarBuffer,
          offset: (windowBase + e.slots[i].off).csize_t,
          size: e.slots[i].size.csize_t)
    var bgDesc = WGPUBindGroupDescriptor(
      layout: batch.pipelines[pipelineKey].bindGroupLayout,
      entryCount: e.slots.len.csize_t,
      entries: bgEntries[0].addr)
    e.bindGroup = wgpuDeviceCreateBindGroup(batch.device, addr bgDesc)
    if e.bindGroup == nil:
      raiseAssert("WebGPU batch: failed to create bind group")
    windowBase = windowBase + e.scalars.len.round_multiple_up(MinStorageBufferOffsetAlignment)

proc replay*(batch: WebgpuBatch) =
  if batch.entries.len == 0 or batch.entries[0].bindGroup == nil:
    raiseAssert("replay: finalize the batch first")
  let encoder = wgpuDeviceCreateCommandEncoder(batch.device, nil)
  if encoder == nil:
    raiseAssert("WebGPU batch: failed to create command encoder")
  wgpuDevicePushErrorScope(batch.device, wgpuErrorFilterValidation)
  let pass = wgpuCommandEncoderBeginComputePass(encoder, nil)
  for e in batch.entries:
    wgpuComputePassEncoderSetPipeline(pass, e.pipelineState)
    wgpuComputePassEncoderSetBindGroup(pass, 0, e.bindGroup, 0, nil)
    wgpuComputePassEncoderDispatchWorkgroups(pass, cuint(e.fleet.x), cuint(e.fleet.y), cuint(e.fleet.z))
  wgpuComputePassEncoderEnd(pass)
  let cmdBuf = wgpuCommandEncoderFinish(encoder, nil)
  wgpuQueueSubmit(batch.queue, 1, cmdBuf.addr)
  wgpuCommandBufferRelease(cmdBuf)
  wgpuCommandEncoderRelease(encoder)
  var popScope = PopErrorScopeData(done: false)
  var popInfo = WGPUPopErrorScopeCallbackInfo(
    mode: wgpuCallbackModeAllowProcessEvents,
    callback: popErrorScopeCb,
    userdata1: popScope.addr,
    userdata2: nil)
  discard wgpuDevicePopErrorScope(batch.device, popInfo)
  while not popScope.done:
    if not wgpuDevicePoll(batch.device, true, nil):
      raiseAssert("WebGPU: wgpuDevicePoll failed")
  if popScope.errType != wgpuErrorTypeNoError:
    raiseAssert("WebGPU batch replay failed: " & popScope.message)
  let uncaptured = takeUncapturedError()
  if uncaptured.errType != wgpuErrorTypeNoError:
    raiseAssert("WebGPU batch uncaptured error: " & uncaptured.message)
