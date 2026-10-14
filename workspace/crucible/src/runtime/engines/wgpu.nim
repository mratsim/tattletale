# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import std/[monotimes, os, strutils, tables, times]

import workspace/crucible/vendor/wgpu
import ../exec/wgpu_runtime
import ../chevrons

export wgpu_runtime
# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  WgpuCtx = object
    ## RAII value wrapper, `=destroy` fires when the engine ref dies.
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
    cache: WgpuCache   # after ctx: released before ctx shutdown
    bakedBlk: Dim3   # workgroup size baked into the shader at ingest

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════
proc `=destroy`(c: var WgpuCtx) =
  if c.ctx.instance != nil:
    c.ctx.shutdown()

proc `=destroy`(cache: var WgpuCache) =
  ## Releases the shader module and cached pipelines while the device context is still alive.
  if cache.module != nil:
    wgpuShaderModuleRelease(cache.module)
  for c in cache.pipelines.mvalues:
    wgpuComputePipelineRelease(c.pipeline)
    wgpuPipelineLayoutRelease(c.pipelineLayout)
    wgpuBindGroupLayoutRelease(c.bindGroupLayout)

proc newWgpuEngine(): WgpuEngine =
  ## Private factory, engines.nim reaches it via `import {.all.}`.
  WgpuEngine(
    ctx: WgpuCtx(ctx: initWgpu()),
    cache: WgpuCache(pipelines: initTable[(string, seq[int]), WgpuPipelineCache]())
  )

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

template check*(status: WGPUBufferMapAsyncStatus, quitOnFailure = true) =
  ## Unified error policy: stacktrace + stderr + quit(1) unless `quitOnFailure = false`.
  ## A template so instantiationInfo() reports the caller's location.
  let code = status
  if code != wgpuBufferMapAsyncStatusSuccess:
    writeStackTrace()
    stderr.write($instantiationInfo() & " exited with error: WGPUBufferMapAsyncStatus " & $code & '\n')
    if quitOnFailure:
      quit 1

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
    quit("WebGPU: failed to create shader module")
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

template failLoud(msg: string) =
  ## Unified error policy: stacktrace + stderr + quit(1) with the caller's location.
  ## A template so instantiationInfo() reports the call site.
  writeStackTrace()
  stderr.write($instantiationInfo() & " exited with error: " & msg & '\n')
  quit 1

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


proc run*(engine: WgpuEngine, kernel: string, output: WGPUBuffer, args: tuple, cfg = LaunchConfig()) =
  ## Runs `kernel` on the engine's device, building the pipeline on first use.
  ##
  ## Usage:
  ##   engine.run<<1024, 256>>("saxpy", outDev, (xDev, yDev, 2.0'f32))
  ##
  ## `output` is the kernel's first argument, then the args in declaration order.
  ## Device buffers pass as-is, scalars copy by value into a scratch buffer
  ## created per launch (WebGPU has no push constants).
  ##
  ## Host seqs and arrays are not launch args:
  ## allocDevice + copyToDeviceFromHost upload them, copyToHostFromDevice reads results back.
  ##
  ## Plain `run` dispatches with the shader's workgroup size, chevron
  ## `run<<grid, blk>>` sets the grid, blk must match the shader's baked workgroup size.
  let device = engine.ctx.ctx.device
  let queue  = engine.ctx.ctx.queue

  # blk is shader-baked (@workgroup_size), a default cfg dispatches the baked size,
  # an explicit blk must match it exactly.
  let blk = if cfg.blk.x == 1 and cfg.blk.y == 1 and cfg.blk.z == 1:
              engine.bakedBlk
            else:
              cfg.blk
  if engine.bakedBlk.x == 0 or
     blk.x != engine.bakedBlk.x or blk.y != engine.bakedBlk.y or
     blk.z != engine.bakedBlk.z:
    quit("WebGPU run blk=" & $blk.x & "x" & $blk.y & "x" & $blk.z &
         " != baked workgroup size " & $engine.bakedBlk.x & "x" & $engine.bakedBlk.y &
         "x" & $engine.bakedBlk.z &
         " — launch config mismatch (blk is shader-baked on WebGPU)")

  var scalarBytes = newSeq[byte]()
  type ArgShape = tuple[isScalar: bool, width: int, offset: int]
  var argShapes: seq[ArgShape]
  var argsCopy = args
  for f in fields(argsCopy):
    when typeof(f) is WGPUBuffer:
      argShapes.add (false, 0, 0)
    elif typeof(f) is bool:
      let v = int32(f)
      argShapes.add (true, sizeof(int32), scalarBytes.len)
      for j in 0 ..< sizeof(int32):
        scalarBytes.add cast[ptr UncheckedArray[byte]](v.addr)[j]
    else:
      argShapes.add (true, sizeof(f), scalarBytes.len)
      for j in 0 ..< sizeof(f):
        scalarBytes.add cast[ptr UncheckedArray[byte]](f.addr)[j]

  var scratch = WGPUBuffer(nil)
  if scalarBytes.len > 0:
    var scratchDesc = WGPUBufferDescriptor(
      usage: wgpuBufferUsageStorage or wgpuBufferUsageCopyDst,
      size: scalarBytes.len.csize_t,
      mappedAtCreation: false)
    scratch = wgpuDeviceCreateBuffer(device, addr scratchDesc)
    if scratch == nil:
      quit("WebGPU: failed to create the scalar scratch buffer")
    defer:
      wgpuBufferRelease(scratch)

  var argShapeKey = newSeq[int](argShapes.len)
  for i, a in argShapes:
    argShapeKey[i] = if a.isScalar: -a.width else: 0
  let cacheKey = (kernel, argShapeKey)
  var cache: WgpuPipelineCache
  if engine.cache.pipelines.hasKey(cacheKey):
    cache = engine.cache.pipelines[cacheKey]
  else:
    let totalBindings = argShapes.len + 1
    var entries = newSeq[WGPUBindGroupLayoutEntry](totalBindings)
    entries[0] = WGPUBindGroupLayoutEntry(
      binding: 0.cuint,
      visibility: WGPUShaderStageCompute,
      buffer: WGPUBufferBindingLayout(
        `type`: WGPUBufferBindingTypeStorage,
        minBindingSize: 0
      )
    )
    for i, a in argShapes:
      entries[i + 1] = WGPUBindGroupLayoutEntry(
        binding: (i + 1).cuint,
        visibility: WGPUShaderStageCompute,
        buffer: WGPUBufferBindingLayout(
          `type`: if a.isScalar: WGPUBufferBindingTypeReadOnlyStorage
                  else: WGPUBufferBindingTypeStorage,
          minBindingSize: a.width.csize_t
        )
      )
    var bglDesc = WGPUBindGroupLayoutDescriptor(
      entryCount: totalBindings.csize_t,
      entries: entries[0].addr
    )
    let bgl = wgpuDeviceCreateBindGroupLayout(device, addr bglDesc)
    if bgl == nil:
      quit("WebGPU: failed to create bind group layout")
    var plDesc = WGPUPipelineLayoutDescriptor(
      bindGroupLayoutCount: 1,
      bindGroupLayouts: bgl.addr
    )
    let pl = wgpuDeviceCreatePipelineLayout(device, addr plDesc)
    if pl == nil:
      quit("WebGPU: failed to create pipeline layout")
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
        quit("WebGPU: wgpuDevicePoll failed")
    if pipeScope.errType != wgpuErrorTypeNoError:
      let uncaptured = takeUncapturedError()
      failLoud("WebGPU pipeline creation failed: " & pipeScope.message &
        (if uncaptured.errType != wgpuErrorTypeNoError: " / " & uncaptured.message else: ""))
    if pipeline == nil:
      quit("WebGPU: failed to create compute pipeline")
    cache = WgpuPipelineCache(
      pipeline: pipeline,
      pipelineLayout: pl,
      bindGroupLayout: bgl
    )
    engine.cache.pipelines[cacheKey] = cache

  let totalBindings = argShapes.len + 1
  var bgEntries = newSeq[WGPUBindGroupEntry](totalBindings)
  bgEntries[0] = WGPUBindGroupEntry(
    binding: 0.cuint,
    buffer: output,
    size: 0
  )
  for i, a in argShapes:
    bgEntries[i + 1] = WGPUBindGroupEntry(
      binding: (i + 1).cuint,
      buffer: if a.isScalar: scratch else: WGPUBuffer(nil),
      offset: a.offset.cuint,
      size: (if a.isScalar: a.width else: 0).csize_t
    )
  var bi = 0
  for f in fields(argsCopy):
    when typeof(f) is WGPUBuffer:
      bgEntries[bi + 1].buffer = f
      inc bi
  var bgDesc = WGPUBindGroupDescriptor(
    layout: cache.bindGroupLayout,
    entryCount: totalBindings.csize_t,
    entries: bgEntries[0].addr
  )
  let bg = wgpuDeviceCreateBindGroup(device, addr bgDesc)
  if bg == nil:
    quit("WebGPU: failed to create bind group")
  defer:
    wgpuBindGroupRelease(bg)

  if scalarBytes.len > 0:
    wgpuQueueWriteBuffer(queue, scratch, 0, scalarBytes[0].addr, scalarBytes.len.csize_t)

  # Record commands: compute pass with the launch validation scope
  let encoder = wgpuDeviceCreateCommandEncoder(device, nil)
  if encoder == nil:
    quit("WebGPU: failed to create command encoder")
  defer:
    wgpuCommandEncoderRelease(encoder)

  let pass = wgpuCommandEncoderBeginComputePass(encoder, nil)
  wgpuComputePassEncoderSetPipeline(pass, cache.pipeline)
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nil)
  # Dispatch-level validation failures surface only through the error scope,
  # push a validation scope around the launch and check the popped result below.
  wgpuDevicePushErrorScope(device, wgpuErrorFilterValidation)
  wgpuComputePassEncoderDispatchWorkgroups(pass, uint32(cfg.grid.x),
                                           uint32(cfg.grid.y), uint32(cfg.grid.z))
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
      quit("WebGPU: wgpuDevicePoll failed")
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
      quit("WebGPU: failed to create a " & $n & "-byte device buffer")

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
      quit("WebGPU: failed to create the staging buffer")
    defer:
      wgpuBufferRelease(stagingBuf)
    let encoder = wgpuDeviceCreateCommandEncoder(device, nil)
    if encoder == nil:
      quit("WebGPU: failed to create command encoder")
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
      quit("WebGPU: wgpuDevicePoll failed")
    let deadline = getMonoTime() + initDuration(seconds = 5)
    while not mapData.done:
      if not wgpuDevicePoll(device, false, nil):
        quit("WebGPU: wgpuDevicePoll failed")
      if deadline <= getMonoTime():
        quit("WebGPU: timed out waiting for wgpuBufferMapAsync callback")
      sleep(2)
    check mapData.status
    let mappedPtr = wgpuBufferGetMappedRange(stagingBuf, 0, mapData.resultBytes.csize_t)
    if mappedPtr != nil and n > 0:
      copyMem(dst, mappedPtr, n)
    wgpuBufferUnmap(stagingBuf)
