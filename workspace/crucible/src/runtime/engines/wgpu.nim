# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import std/[monotimes, os, strutils, tables, times]

import workspace/crucible/vendor/wgpu
import ../../codegen/ir/gpu_types
import ../exec/wgpu_runtime
import ../engines_common
import ../errors
import ../launch/launch_serialization

export wgpu_runtime
# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  WgpuCtx = object
    ## RAII value wrapper, `=destroy` releases the device resources when the engine ref dies.
    ctx: WgpuContext

  WgpuPipelineCache = object
    ## Per-(kernel, arg-shape) pipeline artifacts: bind group layout, pipeline
    ## layout and compute pipeline. Created together on a cache miss, released
    ## together at re-ingest or engine destruction.
    pipeline: WGPUComputePipeline
    pipelineLayout: WGPUPipelineLayout
    bindGroupLayout: WGPUBindGroupLayout

  WgpuCache = object
    ## RAII value wrapper, `=destroy` fires when the engine ref dies.
    module: WGPUShaderModule
    pipelines: Table[(string, seq[int]), WgpuPipelineCache]

  WgpuEngine* = ref object
    ## A WebGPU engine, build with `bkWGSL.init()`: WGSL compute kernels on one device.
    ## All GPU resources release when the engine dies.
    source: string
    ctx: WgpuCtx
    cache: WgpuCache   # after ctx: released before the context destroy
    bakedBlk: Dim3   # workgroup size baked into the shader at ingest

proc backendKind*(_: typedesc[WgpuEngine]): static BackendKind = bkWGSL

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════
proc `=destroy`(c: var WgpuCtx) =
  ## Releases the wgpu resources, nil-guarded against double teardown.
  if c.ctx.instance != nil:
    wgpuDeviceRelease(c.ctx.device)
    wgpuAdapterRelease(c.ctx.adapter)
    wgpuInstanceRelease(c.ctx.instance)
    c.ctx.device = nil
    c.ctx.adapter = nil
    c.ctx.instance = nil

proc `=destroy`(cache: var WgpuCache) =
  ## Releases the shader module and cached pipelines while the device context is still alive.
  if cache.module != nil:
    wgpuShaderModuleRelease(cache.module)
  for c in cache.pipelines.mvalues:
    wgpuComputePipelineRelease(c.pipeline)
    wgpuPipelineLayoutRelease(c.pipelineLayout)
    wgpuBindGroupLayoutRelease(c.bindGroupLayout)

proc newWgpuEngine(): WgpuEngine =
  WgpuEngine(
    ctx: WgpuCtx(ctx: initWgpu()),
    cache: WgpuCache(pipelines: initTable[(string, seq[int]), WgpuPipelineCache]())
  )

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

template check*(status: WGPUBufferMapAsyncStatus) =
  ## Error policy: stacktrace, stderr, exit 1. A template so instantiationInfo()
  ## reports the caller's location.
  let code = status
  if code != wgpuBufferMapAsyncStatusSuccess:
    failLoud("WGPUBufferMapAsyncStatus " & $code)

proc ingest*(engine: WgpuEngine, source: string) =
  ## Stores the WGSL source and builds the shader module, re-entrant (a new artifact replaces the old, cached pipelines are invalidated).
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
  engine.cache.module = wgpuDeviceCreateShaderModule(engine.ctx.ctx.device, addr shaderDesc)
  if engine.cache.module == nil:
    failLoud("WebGPU: failed to create shader module")
  engine.source = source
  engine.bakedBlk = parseBakedBlk(source)

proc getArtifact*(engine: WgpuEngine): string =
  ## Returns the WGSL kernel source.
  engine.source

proc deviceName*(engine: WgpuEngine): string {.inline.} =
  ## Returns the WebGPU adapter device name (e.g. "NVIDIA RTX PRO 6000 ...").
  engine.ctx.ctx.deviceName()

# ─────────────────────────────────────────────────────────────────────────
# ▸ PRIVATE
# ─────────────────────────────────────────────────────────────────────────

proc parseBakedBlk(wgsl: string): Dim3 =
  ## Extracts the shader-baked workgroup size from the WGSL preamble
  ## `@compute @workgroup_size(64, 8, 1)`, 1..3 literal args, missing default to 1.
  ## Returns (0, 0, 0) when absent or non-literal, run then fails blk validation.
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


proc launchKernelImpl*(engine: WgpuEngine, kernel: string, output: WGPUBuffer, args: tuple, grid, blk = Dim3()) =
  ## Runs `kernel` on the engine's device, building the pipeline on first use:
  ## - `output` receives the kernel's result, `args` bind positionally
  ## - `grid`/`blk` carry the launch geometry, `blk` must match the shader's baked workgroup size
  ## - a plain run dispatches (1, 1)
  ## - example: engine.launchKernel<<(1024, 256)>>("saxpy", outDev, (xDev, yDev, 2.0'f32))
  let device = engine.ctx.ctx.device
  let queue  = engine.ctx.ctx.queue

  # blk is shader-baked (@workgroup_size), the default blk (1, 1, 1) dispatches the baked size,
  # an explicit blk must match it exactly.
  let blk = if blk.x == 1 and blk.y == 1 and blk.z == 1:
              engine.bakedBlk
            else:
              blk
  if engine.bakedBlk.x == 0 or
     blk.x != engine.bakedBlk.x or blk.y != engine.bakedBlk.y or
     blk.z != engine.bakedBlk.z:
    failLoud("WebGPU run blk=" & $blk.x & "x" & $blk.y & "x" & $blk.z &
         " != baked workgroup size " & $engine.bakedBlk.x & "x" & $engine.bakedBlk.y &
         "x" & $engine.bakedBlk.z &
         " — launch config mismatch (blk is shader-baked on WebGPU)")

  # Storage-binding offsets must be multiples of minStorageBufferOffsetAlignment,
  # the device keeps the spec-default limits, the spec-default alignment is 256.
  # byte chunks pack into a scratch storage buffer at 256-byte alignment
  const n = launchArgCount(typeof(output), typeof(args), WGPUBuffer, bkWGSL)
  type SinkCtx = object
    handles: array[n, WGPUBuffer]
    chunkOff, chunkLen: array[n, int]
    blob: seq[byte]
    pn: int
  proc sinkArg(ctx: var SinkCtx, isBuffer: bool, h: WGPUBuffer, idx: int, p: pointer, len: int) {.nimcall.} =
    if isBuffer:
      ctx.handles[idx] = h
    else:
      ctx.pn = (ctx.pn + 255) div 256 * 256
      ctx.chunkOff[idx] = ctx.pn
      # storage-buffer binding sizes are multiples of 4 bytes
      let padded = (len + 3) div 4 * 4
      ctx.chunkLen[idx] = padded
      ctx.blob.setLen(ctx.pn + padded)
      copyMem(ctx.blob[ctx.pn].addr, p, len)
      inc ctx.pn, padded
  var dctx = SinkCtx()
  bindArgs(dctx, bkWGSL, output, args, false, sinkArg)
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
  # Released with the launch, not with the allocation block, the bind group
  # and the queue upload below still use it.
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
  # wgpu-native rejects a zero binding size, whole-buffer bindings carry
  # the buffer's queried size.
  bgEntries[0] = WGPUBindGroupEntry(
    binding: 0.cuint,
    buffer: output,
    size: wgpuBufferGetSize(output).csize_t
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

  # Record commands: compute pass with the launch validation scope
  let encoder = wgpuDeviceCreateCommandEncoder(device, nil)
  if encoder == nil:
    failLoud("WebGPU: failed to create command encoder")
  defer:
    wgpuCommandEncoderRelease(encoder)

  let pass = wgpuCommandEncoderBeginComputePass(encoder, nil)
  wgpuComputePassEncoderSetPipeline(pass, cache.pipeline)
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nil)
  # Dispatch-level validation failures surface only through the error scope,
  # push a validation scope around the launch and check the popped result below.
  wgpuDevicePushErrorScope(device, wgpuErrorFilterValidation)
  wgpuComputePassEncoderDispatchWorkgroups(pass, uint32(grid.x),
                                           uint32(grid.y), uint32(grid.z))
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

  # errors outside the dispatch error scope land in the device's
  # uncaptured-error callback, check it before declaring the run done
  let uncaptured = takeUncapturedError()
  if uncaptured.errType != wgpuErrorTypeNoError:
    failLoud("WebGPU uncaptured error: " & uncaptured.message)

proc allocDevice*(engine: WgpuEngine, n: int): WGPUBuffer =
  ## Storage buffer of n bytes, readable and writable by copy, release via freeDevice.
  let device = engine.ctx.ctx.device
  var desc = WGPUBufferDescriptor(
    usage: wgpuBufferUsageStorage or wgpuBufferUsageCopySrc or
           wgpuBufferUsageCopyDst,
    size: n.csize_t,
    mappedAtCreation: false)
  result = wgpuDeviceCreateBuffer(device, addr desc)
  if result == nil:
    failLoud("WebGPU: failed to create a " & $n & "-byte device buffer")

proc freeDevice*(engine: WgpuEngine, h: WGPUBuffer) =
  wgpuBufferRelease(h)

proc copyToDeviceFromHost*(engine: WgpuEngine, dst: WGPUBuffer, src: pointer, n: int) =
  wgpuQueueWriteBuffer(engine.ctx.ctx.queue, dst, 0, src, n.csize_t)

proc copyToHostFromDevice*(engine: WgpuEngine, dst: pointer, src: WGPUBuffer, n: int) =
  ## Staging copy + map: WebGPU storage buffers are not host-mappable.
  let device = engine.ctx.ctx.device
  let queue  = engine.ctx.ctx.queue
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
