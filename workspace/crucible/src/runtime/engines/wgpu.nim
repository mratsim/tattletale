# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import std/[macros, monotimes, os, strutils, tables, times]

import workspace/crucible/vendor/wgpu
import ../../codegen/ir/gpu_types
import ../exec/wgpu_runtime
import ./engines_common
import ../errors
import ../launch/launch_serialization

export wgpu_runtime

# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════

type
  WgpuPipelineCache = object
    pipeline: WGPUComputePipeline
    pipelineLayout: WGPUPipelineLayout
    bindGroupLayout: WGPUBindGroupLayout

  WgpuCache = object
    module: WGPUShaderModule
    pipelines: Table[(string, seq[int]), WgpuPipelineCache]

  WgpuEngine* = ref object
    source: string
    ctx: WgpuContext
    cache: WgpuCache
    shaderSquadron: Dim3

proc backendKind*(_: typedesc[WgpuEngine]): static BackendKind = bkWGSL

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════

proc `=destroy`(cache: var WgpuCache) =
  if cache.module != nil:
    wgpuShaderModuleRelease(cache.module)
  for c in cache.pipelines.mvalues:
    wgpuComputePipelineRelease(c.pipeline)
    wgpuPipelineLayoutRelease(c.pipelineLayout)
    wgpuBindGroupLayoutRelease(c.bindGroupLayout)

proc newWgpuEngine*(): WgpuEngine =
  WgpuEngine(
    ctx: initWgpu(),
    cache: WgpuCache(pipelines: initTable[(string, seq[int]), WgpuPipelineCache]())
  )

# ═════════════════════════════════════════════════════════════════════════
# ▸ Launch serialization
# ═════════════════════════════════════════════════════════════════════════

type SerializationCtx = object
  handles: array[64, WGPUBuffer]
  chunkOff, chunkLen: array[64, int]
  blob: seq[byte]
  pn: int

proc store(ctx: var SerializationCtx, v: WGPUBuffer, argIdx: int) =
  ctx.handles[argIdx] = v

proc store[T](ctx: var SerializationCtx, v: T, argIdx: int) =
  when T is ptr|pointer:
    static:
      error "Pointer kernel args are not supported, pass the DeviceMem handle from allocDevice, got " & $T
  ctx.pn = roundStepUp(ctx.pn, 256)
  ctx.chunkOff[argIdx] = ctx.pn
  const size = deviceLayout(T, bkWGSL).size
  let padded = roundStepUp(size, 4)
  ctx.chunkLen[argIdx] = padded
  ctx.blob.setLen(ctx.pn + padded)
  if size > 0:
    var blob: array[size, byte]
    var off = 0
    blob.packFrom(off, v, bkWGSL)
    copyMem(ctx.blob[ctx.pn].addr, blob[0].addr, size)
  inc ctx.pn, padded

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

template check*(status: WGPUBufferMapAsyncStatus) =
  let code = status
  if code != wgpuBufferMapAsyncStatusSuccess:
    failLoud("WGPUBufferMapAsyncStatus " & $code)

proc ingest*(engine: WgpuEngine, source: string) =
  if engine.cache.module != nil:
    wgpuShaderModuleRelease(engine.cache.module)
    engine.cache.module = nil
  for c in engine.cache.pipelines.mvalues:
    wgpuComputePipelineRelease(c.pipeline)
    wgpuPipelineLayoutRelease(c.pipelineLayout)
    wgpuBindGroupLayoutRelease(c.bindGroupLayout)
  engine.cache.pipelines.clear()
  var wgslView = WGPUStringView(data: cstring(source), length: source.len.csize_t)
  var wgslSource = WGPUShaderSourceWGSL(
    chain: WGPUChainedStruct(
      sType: WGPUSType_ShaderSourceWGSL,
      next: nil
    ),
    code: wgslView
  )
  var shaderDesc = WGPUShaderModuleDescriptor(
    nextInChain: addr wgslSource
  )
  engine.cache.module = wgpuDeviceCreateShaderModule(engine.ctx.device, addr shaderDesc)
  if engine.cache.module == nil:
    failLoud("WebGPU: failed to create shader module")
  engine.source = source
  engine.shaderSquadron = parseShaderSquadron(source)

proc getArtifact*(engine: WgpuEngine): string =
  engine.source

proc deviceName*(engine: WgpuEngine): string {.inline.} =
  engine.ctx.deviceName()

proc allocDevice*[T](engine: WgpuEngine, _: typedesc[T], len: int): DeviceMem[WGPUBuffer, T] =
  ## Allocates len elements of T on the engine
  DeviceMem[WGPUBuffer, T](mem: engine.allocDeviceImpl(len * sizeof(T)))

proc copyToDeviceFromHost*[T](engine: WgpuEngine, dst: DeviceMem[WGPUBuffer, T], src: ptr T or ptr UncheckedArray[T], n: int) =
  ## Copies n elements from host to device
  engine.copyToDeviceFromHostImpl(dst.mem, src, n * sizeof(T))

proc copyToHostFromDevice*[T](engine: WgpuEngine, dst: ptr T or ptr UncheckedArray[T], src: DeviceMem[WGPUBuffer, T], n: int) =
  ## Copies n elements from device to host
  engine.copyToHostFromDeviceImpl(dst, src.mem, n * sizeof(T))
# ─────────────────────────────────────────────────────────────────────────
# ▸ PRIVATE
# ─────────────────────────────────────────────────────────────────────────

proc parseShaderSquadron(wgsl: string): Dim3 =
  ## Extracts the shader-baked squadron from the WGSL preamble
  ## `@compute @workgroup_size(64, 8, 1)`, 1..3 literal args, missing default to 1.
  ## Returns (0, 0, 0) when absent or non-literal, run then fails squadron validation.
  const marker = "@workgroup_size("
  let i = wgsl.find(marker)
  if i < 0:
    return Dim3(x: 0, y: 0, z: 0)
  var j = i + marker.len
  var dim = 0
  while j < wgsl.len and wgsl[j] != ')':
    if wgsl[j] in {'0' .. '9'}:
      var n = 0
      while j < wgsl.len and wgsl[j] in {'0' .. '9'}:
        n = n * 10 + (ord(wgsl[j]) - ord('0'))
        inc j
      case dim
      of 0: result.x = n
      of 1: result.y = n
      else: result.z = n
      inc dim
    else:
      inc j
  if result.x == 0:
    return Dim3(x: 0, y: 0, z: 0)   # absent or non-literal
  if result.y == 0: result.y = 1
  if result.z == 0: result.z = 1

proc launchKernelImpl*(engine: WgpuEngine, kernel: string, output: auto, args: var tuple, cfg: LaunchConfig) =
  if cfg.sharedMem != 0 or cfg.stream != 0:
    failLoud("WebGPU run: launch-time sharedMem/stream are unsupported, they are Metal/CUDA features")
  let device = engine.ctx.device
  let queue  = engine.ctx.queue

  let squadron = if cfg.blk.x == 1 and cfg.blk.y == 1 and cfg.blk.z == 1:
              engine.shaderSquadron
            else:
              cfg.blk
  if engine.shaderSquadron.x == 0 or
     squadron.x != engine.shaderSquadron.x or squadron.y != engine.shaderSquadron.y or
     squadron.z != engine.shaderSquadron.z:
    failLoud("WebGPU run squadron=" & $squadron.x & "x" & $squadron.y & "x" & $squadron.z &
         " != baked squadron " & $engine.shaderSquadron.x & "x" & $engine.shaderSquadron.y &
         "x" & $engine.shaderSquadron.z &
         " — launch config mismatch (squadron is shader-baked on WebGPU)")

  # Storage-binding offsets must be multiples of minStorageBufferOffsetAlignment,
  # the device keeps the spec-default limits, the spec-default alignment is 256.
  const n = launchArgCount(typeof(output), typeof(args), WGPUBuffer, bkWGSL)

  var dctx = SerializationCtx()
  bindArgs(dctx, output, args, store, WGPUBuffer, bkWGSL)
  let scalarBytes = dctx.blob

  var scratch = WGPUBuffer(nil)
  if scalarBytes.len > 0:
    var scratchDesc = WGPUBufferDescriptor(
      usage: wgpuBufferUsageStorage or wgpuBufferUsageCopyDst,
      size: scalarBytes.len.csize_t,
      mappedAtCreation: false)
    scratch = wgpuDeviceCreateBuffer(device, addr scratchDesc)
    if scratch == nil:
      failLoud("WebGPU: failed to create the scalar scratch buffer")

  defer:
    if scratch != nil:
      wgpuBufferRelease(scratch)

  var argShapeKey = newSeq[int](n)
  for i in 0 ..< n:
    argShapeKey[i] = -dctx.chunkLen[i]
  let cacheKey = (kernel, argShapeKey)
  var cache: WgpuPipelineCache
  if engine.cache.pipelines.hasKey(cacheKey):
    cache = engine.cache.pipelines[cacheKey]
  else:
    let totalBindings = n
    var entries = newSeq[WGPUBindGroupLayoutEntry](totalBindings)
    entries[0] = WGPUBindGroupLayoutEntry(
      binding: 0.cuint,
      visibility: WGPUShaderStageCompute,
      buffer: WGPUBufferBindingLayout(
        `type`: WGPUBufferBindingTypeStorage,
        minBindingSize: 0
      )
    )
    for i in 1 ..< n:
      entries[i] = WGPUBindGroupLayoutEntry(
        binding: i.cuint,
        visibility: WGPUShaderStageCompute,
        buffer: WGPUBufferBindingLayout(
          `type`: if dctx.chunkLen[i] > 0: WGPUBufferBindingTypeReadOnlyStorage
                  else: WGPUBufferBindingTypeStorage,
          minBindingSize: dctx.chunkLen[i].csize_t
        )
      )
    var bglDesc = WGPUBindGroupLayoutDescriptor(
      entryCount: totalBindings.csize_t,
      entries: entries[0].addr
    )
    let bgl = wgpuDeviceCreateBindGroupLayout(device, addr bglDesc)
    if bgl == nil:
      failLoud("WebGPU: failed to create bind group layout")
    var plDesc = WGPUPipelineLayoutDescriptor(
      bindGroupLayoutCount: 1,
      bindGroupLayouts: bgl.addr
    )
    let pl = wgpuDeviceCreatePipelineLayout(device, addr plDesc)
    if pl == nil:
      failLoud("WebGPU: failed to create pipeline layout")
    var entryPtView = WGPUStringView(data: cstring(kernel), length: kernel.len.csize_t)
    var computeState = WGPUComputeState(
      module: engine.cache.module,
      entryPoint: entryPtView
    )
    var cpDesc = WGPUComputePipelineDescriptor(
      layout: pl,
      compute: computeState
    )
    wgpuDevicePushErrorScope(device, wgpuErrorFilterValidation)
    let pipeline = wgpuDeviceCreateComputePipeline(device, addr cpDesc)
    var pipeScope = PopErrorScopeData(done: false)
    var pipePopInfo = WGPUPopErrorScopeCallbackInfo(
      mode: wgpuCallbackModeAllowProcessEvents,
      callback: popErrorScopeCb,
      userdata1: pipeScope.addr,
      userdata2: nil
    )
    discard wgpuDevicePopErrorScope(device, pipePopInfo)
    while not pipeScope.done:
      if not wgpuDevicePoll(device, true, nil):
        failLoud("WebGPU: wgpuDevicePoll failed")
    if pipeScope.errType != wgpuErrorTypeNoError:
      let uncaptured = takeUncapturedError()
      failLoud("WebGPU pipeline creation failed: " & pipeScope.message &
        (if uncaptured.errType != wgpuErrorTypeNoError: " / " & uncaptured.message else: ""))
    if pipeline == nil:
      failLoud("WebGPU: failed to create compute pipeline")
    cache = WgpuPipelineCache(
      pipeline: pipeline,
      pipelineLayout: pl,
      bindGroupLayout: bgl
    )
    engine.cache.pipelines[cacheKey] = cache

  let totalBindings = n
  var bgEntries = newSeq[WGPUBindGroupEntry](totalBindings)

  bgEntries[0] = WGPUBindGroupEntry(
    binding: 0.cuint,
    buffer: dctx.handles[0],
    size: wgpuBufferGetSize(dctx.handles[0]).csize_t
  )
  for i in 1 ..< n:
    bgEntries[i] = WGPUBindGroupEntry(
      binding: i.cuint,
      buffer: if dctx.chunkLen[i] > 0: scratch else: dctx.handles[i],
      offset: dctx.chunkOff[i].cuint,
      size: dctx.chunkLen[i].csize_t
    )
  for i in 1 ..< n:
    if dctx.chunkLen[i] == 0:
      bgEntries[i].size = wgpuBufferGetSize(dctx.handles[i]).csize_t
  var bgDesc = WGPUBindGroupDescriptor(
    layout: cache.bindGroupLayout,
    entryCount: totalBindings.csize_t,
    entries: bgEntries[0].addr
  )
  let bg = wgpuDeviceCreateBindGroup(device, addr bgDesc)
  if bg == nil:
    failLoud("WebGPU: failed to create bind group")
  defer:
    wgpuBindGroupRelease(bg)

  if scalarBytes.len > 0:
    wgpuQueueWriteBuffer(queue, scratch, 0, scalarBytes[0].addr, scalarBytes.len.csize_t)

  let encoder = wgpuDeviceCreateCommandEncoder(device, nil)
  if encoder == nil:
    failLoud("WebGPU: failed to create command encoder")
  defer:
    wgpuCommandEncoderRelease(encoder)

  let pass = wgpuCommandEncoderBeginComputePass(encoder, nil)
  wgpuComputePassEncoderSetPipeline(pass, cache.pipeline)
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nil)
  wgpuDevicePushErrorScope(device, wgpuErrorFilterValidation)
  wgpuComputePassEncoderDispatchWorkgroups(pass, uint32(cfg.grid.x), uint32(cfg.grid.y), uint32(cfg.grid.z))
  wgpuComputePassEncoderEnd(pass)
  let cmdBuf = wgpuCommandEncoderFinish(encoder, nil)

  var scopeData = PopErrorScopeData(done: false)
  var popCbInfo = WGPUPopErrorScopeCallbackInfo(
    mode: wgpuCallbackModeAllowProcessEvents,
    callback: popErrorScopeCb,
    userdata1: scopeData.addr,
    userdata2: nil
  )
  discard wgpuDevicePopErrorScope(device, popCbInfo)
  wgpuQueueSubmit(queue, 1, cmdBuf.addr)
  defer:
    wgpuCommandBufferRelease(cmdBuf)
  while not scopeData.done:
    if not wgpuDevicePoll(device, true, nil):
      failLoud("WebGPU: wgpuDevicePoll failed")
  if scopeData.errType != wgpuErrorTypeNoError:
    failLoud("WebGPU dispatch failed: " & scopeData.message)

  let uncaptured = takeUncapturedError()
  if uncaptured.errType != wgpuErrorTypeNoError:
    failLoud("WebGPU uncaptured error: " & uncaptured.message)

proc allocDeviceImpl(engine: WgpuEngine, n: int): WGPUBuffer =
  let device = engine.ctx.device
  var desc = WGPUBufferDescriptor(
    usage: wgpuBufferUsageStorage or wgpuBufferUsageCopySrc or
           wgpuBufferUsageCopyDst,
    size: n.csize_t,
    mappedAtCreation: false)
  result = wgpuDeviceCreateBuffer(device, addr desc)
  if result == nil:
    failLoud("WebGPU: failed to create a " & $n & "-byte device buffer")

proc copyToDeviceFromHostImpl(engine: WgpuEngine, dst: WGPUBuffer, src: pointer, n: int) =
  wgpuQueueWriteBuffer(engine.ctx.queue, dst, 0, src, n.csize_t)

proc copyToHostFromDeviceImpl(engine: WgpuEngine, dst: pointer, src: WGPUBuffer, n: int) =
  let device = engine.ctx.device
  let queue  = engine.ctx.queue
  var stagingDesc = WGPUBufferDescriptor(
    usage: wgpuBufferUsageMapRead or wgpuBufferUsageCopyDst,
    size: n.csize_t,
    mappedAtCreation: false)
  let stagingBuf = wgpuDeviceCreateBuffer(device, addr stagingDesc)
  if stagingBuf == nil:
    failLoud("WebGPU: failed to create the staging buffer")
  defer:
    wgpuBufferRelease(stagingBuf)
  let encoder = wgpuDeviceCreateCommandEncoder(device, nil)
  if encoder == nil:
    failLoud("WebGPU: failed to create command encoder")
  defer:
    wgpuCommandEncoderRelease(encoder)
  wgpuCommandEncoderCopyBufferToBuffer(encoder, src, 0, stagingBuf, 0, n.csize_t)
  let cmdBuf = wgpuCommandEncoderFinish(encoder, nil)
  wgpuQueueSubmit(queue, 1, cmdBuf.addr)
  defer:
    wgpuCommandBufferRelease(cmdBuf)
  let mapData = create(MapDoneData)
  mapData[] = MapDoneData(done: false, resultBytes: n,
                          status: wgpuBufferMapAsyncStatusUnknown)
  defer:
    dealloc(mapData)
  var mapCbInfo = WGPUBufferMapCallbackInfo(
    mode: wgpuCallbackModeAllowProcessEvents,
    callback: bufferMapCb,
    userdata1: mapData,
    userdata2: nil
  )
  discard wgpuBufferMapAsync(stagingBuf, wgpuMapModeRead, 0, n.csize_t, mapCbInfo)
  if not wgpuDevicePoll(device, true, nil):
    failLoud("WebGPU: wgpuDevicePoll failed")
  let deadline = getMonoTime() + initDuration(seconds = 5)
  while not mapData.done:
    if not wgpuDevicePoll(device, false, nil):
      failLoud("WebGPU: wgpuDevicePoll failed")
    if deadline <= getMonoTime():
      failLoud("WebGPU: timed out waiting for wgpuBufferMapAsync callback")
    sleep(2)
  check mapData.status
  let mappedPtr = wgpuBufferGetMappedRange(stagingBuf, 0, mapData.resultBytes.csize_t)
  if mappedPtr != nil and n > 0:
    copyMem(dst, mappedPtr, n)
  wgpuBufferUnmap(stagingBuf)
