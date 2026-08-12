# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## WgpuEngine — WebGPU (wgpu-native) execution (moved from codegen/wgpu.nim,
## decoupled from the compile-time DSL: no gpu_compiler import).
##
## ingest = build shader module from WGSL; getArtifact = WGSL; run = pipeline +
## dispatchWorkgroups where grid = dispatchWorkgroups count. blk is
## shader-baked (@workgroup_size): run's blk is validated loudly. The
## staging-buffer map pattern captures the map status in
## `check`. Scalars (ArgBlob size < 0) use a storage-buffer fallback
## (future: uniform).
##
## The device context (initWgpu) and its callbacks live in `exec/wgpu_runtime`
## (imported below); this module owns the engine and its RAII `=destroy` hook.
##
## Structure: PUBLIC API block first (exported `*`); PRIVATE machinery below
## (no `*`). `{.experimental: "codeReordering".}` lifts Nim's
## declaration-before-use rule so the private types/helpers may follow the
## public surface that calls them.
{.experimental: "codeReordering".}

import std/[monotimes, os, strutils, tables, times]

import workspace/crucible/vendor/wgpu
import ../exec/wgpu_runtime
import ./arg_blobs
import ../chevrons

export wgpu_runtime
# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  WgpuCtx = object
    ## RAII value wrapper — `=destroy` fires when the engine ref dies.
    ctx: WgpuContext

  WgpuPipelineCache = object
    ## Per-(kernel, arg-shape) pipeline artifacts: bind group layout, pipeline
    ## layout and compute pipeline. Created together on a cache miss, released
    ## together at re-ingest or engine destruction.
    pipeline: WGPUComputePipeline
    pipelineLayout: WGPUPipelineLayout
    bindGroupLayout: WGPUBindGroupLayout

  WgpuCache = object
    ## RAII value wrapper — `=destroy` fires when the engine ref dies.
    module: WGPUShaderModule
    pipelines: Table[(string, seq[int]), WgpuPipelineCache]

  WgpuEngine* = ref object
    ## Fields directly (no Obj indirection); resources in the RAII value fields.
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
  ## Release the shader module and cached pipelines while the device context
  ## is still alive: this field is declared after `ctx`, so reverse-order
  ## field destruction runs it before `ctx`'s `=destroy` shuts the device.
  if cache.module != nil:
    wgpuShaderModuleRelease(cache.module)
  for c in cache.pipelines.mvalues:
    wgpuComputePipelineRelease(c.pipeline)
    wgpuPipelineLayoutRelease(c.pipelineLayout)
    wgpuBindGroupLayoutRelease(c.bindGroupLayout)

proc newWgpuEngine(): WgpuEngine =
  ## Private factory — engines.nim reaches it via `import {.all.}`.
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
  ## Store the WGSL source and build the shader module. Re-entrant: replaces
  ## the previous artifact (the device context persists — only the artifact
  ## is replaced) and invalidates cached pipelines.
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
  ## The WGSL kernel source.
  engine.source

proc deviceName*(engine: WgpuEngine): string {.inline.} =
  ## The WebGPU adapter device name (e.g. "NVIDIA RTX PRO 6000 ...").
  engine.ctx.ctx.deviceName()

# ─────────────────────────────────────────────────────────────────────────
# ▸ PRIVATE
# ─────────────────────────────────────────────────────────────────────────

template failLoud(msg: string) =
  ## Unified error policy: stacktrace + stderr + quit(1) with the caller's
  ## location. A template so instantiationInfo() reports the call site.
  writeStackTrace()
  stderr.write($instantiationInfo() & " exited with error: " & msg & '\n')
  quit 1

proc parseBakedBlk(wgsl: string): Dim3 =
  ## Extract the shader-baked workgroup size from the WGSL preamble:
  ## `@compute @workgroup_size(64, 8, 1)` — 1..3 literal args, missing
  ## default to 1. Returns (0, 0, 0) when absent or non-literal, so run
  ## fails blk validation loudly.
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

proc runImpl(engine: WgpuEngine, kernel: string, output: ArgBlob,
             blobs: seq[ArgBlob], cfg: LaunchConfig) =
  ## Pipeline + dispatchWorkgroups(cfg.grid). The output is binding 0
  ## (output first, per CONVENTIONS.md), then the input args in order
  ## (all as storage buffers — scalars via the buffer fallback).
  let device = engine.ctx.ctx.device
  let queue  = engine.ctx.ctx.queue
  let outSize = abs(output.size)

  # blk is shader-baked (@workgroup_size). A default cfg (plain run) dispatches
  # with the baked size; an explicit blk must match it exactly.
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

  # 1. Shader module: created once at ingest (engine.module) and reused by
  # every run, per the ingest-once / run-many contract.

  # 2. Create buffers via the staging pattern (no Map usage on storage buffers)
  let numInputs = blobs.len
  let totalBindings = numInputs + 1
  var inputBuffers = newSeq[WGPUBuffer](numInputs)
  var inputSizes = newSeq[int](numInputs)
  for i in 0 ..< numInputs:
    let sz = abs(blobs[i].size)   # scalars → small storage buffer fallback
    inputSizes[i] = sz
    let desc = WGPUBufferDescriptor(
      usage: wgpuBufferUsageStorage or wgpuBufferUsageCopyDst,
      size: sz.csize_t,
      mappedAtCreation: false
    )
    inputBuffers[i] = wgpuDeviceCreateBuffer(device, addr desc)
    if inputBuffers[i] == nil:
      quit("WebGPU: failed to create input buffer")
  defer:
    for buf in inputBuffers:
      wgpuBufferRelease(buf)

  # Output buffer: shader writes here, then we copy to staging.
  # COPY_DST is required for the in-place output upload (β·C).
  var outBufDesc = WGPUBufferDescriptor(
    usage: wgpuBufferUsageStorage or wgpuBufferUsageCopySrc or wgpuBufferUsageCopyDst,
    size: outSize.csize_t,
    mappedAtCreation: false)
  let outBuf = wgpuDeviceCreateBuffer(device, addr outBufDesc)
  if outBuf == nil:
    quit("WebGPU: failed to create output buffer")
  defer:
    wgpuBufferRelease(outBuf)

  # Staging buffer: copy output here, then map for CPU reading
  var stagingDesc = WGPUBufferDescriptor(
    usage: wgpuBufferUsageMapRead or wgpuBufferUsageCopyDst,
    size: outSize.csize_t,
    mappedAtCreation: false)
  let stagingBuf = wgpuDeviceCreateBuffer(device, addr stagingDesc)
  if stagingBuf == nil:
    quit("WebGPU: failed to create staging buffer")
  defer:
    wgpuBufferRelease(stagingBuf)

  # 3. Bind group layout, pipeline layout and compute pipeline: cached per
  # (kernel, arg shape), per the ingest-once / run-many contract. Rebuilt
  # only on a cache miss; the bind group (section 4) and the buffers stay
  # per-run.
  var argSizes = newSeq[int](numInputs)
  for i in 0 ..< numInputs:
    argSizes[i] = blobs[i].size   # signed: negative = scalar (read-only binding)
  let cacheKey = (kernel, argSizes)
  var cache: WgpuPipelineCache
  if engine.cache.pipelines.hasKey(cacheKey):
    cache = engine.cache.pipelines[cacheKey]
  else:
    # Bind group layout: output at 0, then inputs
    var entries = newSeq[WGPUBindGroupLayoutEntry](totalBindings)
    entries[0] = WGPUBindGroupLayoutEntry(
      binding: 0.cuint,
      visibility: WGPUShaderStageCompute,
      buffer: WGPUBufferBindingLayout(
        `type`: WGPUBufferBindingTypeStorage,
        minBindingSize: outSize.csize_t
      )
    )
    for i in 0 ..< numInputs:
      # Scalar (by-value) inputs map to `var<storage, read>` in the WGSL the
      # backend emits, so they bind read-only; ptr inputs map to
      # `var<storage, read_write>` and keep the read-write Storage binding.
      let isScalar = blobs[i].size < 0
      entries[i + 1] = WGPUBindGroupLayoutEntry(
        binding: (i + 1).cuint,
        visibility: WGPUShaderStageCompute,
        buffer: WGPUBufferBindingLayout(
          `type`: if isScalar: WGPUBufferBindingTypeReadOnlyStorage
                  else: WGPUBufferBindingTypeStorage,
          minBindingSize: inputSizes[i].csize_t
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
    # Compute pipeline: entry point is the kernel name, module from ingest
    var entryPtView = WGPUStringView(data: cstring(kernel), length: kernel.len.csize_t)
    var computeState = WGPUComputeState(
      module: engine.cache.module,
      entryPoint: entryPtView
    )
    var cpDesc = WGPUComputePipelineDescriptor(
      layout: pl,
      compute: computeState
    )
    let pipeline = wgpuDeviceCreateComputePipeline(device, addr cpDesc)
    if pipeline == nil:
      quit("WebGPU: failed to create compute pipeline")
    cache = WgpuPipelineCache(
      pipeline: pipeline,
      pipelineLayout: pl,
      bindGroupLayout: bgl
    )
    engine.cache.pipelines[cacheKey] = cache
  let bgl = cache.bindGroupLayout
  let pl = cache.pipelineLayout
  let pipeline = cache.pipeline

  # 4. Bind group entries (output at 0, then inputs)
  var bgEntries = newSeq[WGPUBindGroupEntry](totalBindings)
  bgEntries[0] = WGPUBindGroupEntry(
    binding: 0.cuint,
    buffer: outBuf,
    size: outSize.csize_t
  )
  for i in 0 ..< numInputs:
    bgEntries[i + 1] = WGPUBindGroupEntry(
      binding: (i + 1).cuint,
      buffer: inputBuffers[i],
      size: inputSizes[i].csize_t
    )
  var bgDesc = WGPUBindGroupDescriptor(
    layout: bgl,
    entryCount: totalBindings.csize_t,
    entries: bgEntries[0].addr
  )
  let bg = wgpuDeviceCreateBindGroup(device, addr bgDesc)
  if bg == nil:
    quit("WebGPU: failed to create bind group")
  defer:
    wgpuBindGroupRelease(bg)

  # 5. Compute pipeline: cached in section 3, released at re-ingest / destroy.

  # 6. Write input data + the output's current contents (in-place β·C)
  for i in 0 ..< numInputs:
    if inputSizes[i] > 0:
      wgpuQueueWriteBuffer(queue, inputBuffers[i], 0, blobs[i].data, inputSizes[i].csize_t)
  if outSize > 0:
    wgpuQueueWriteBuffer(queue, outBuf, 0, output.data, outSize.csize_t)

  # 7. Record commands: compute pass + copy output → staging
  let encoder = wgpuDeviceCreateCommandEncoder(device, nil)
  if encoder == nil:
    quit("WebGPU: failed to create command encoder")
  defer:
    wgpuCommandEncoderRelease(encoder)

  let pass = wgpuCommandEncoderBeginComputePass(encoder, nil)
  wgpuComputePassEncoderSetPipeline(pass, pipeline)
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nil)
  # grid = dispatchWorkgroups count (per axis)
  # Dispatch-level validation failures (e.g. grid.x > 65535, the per-axis max
  # in webgpu.h) are only reported through the error scope: the staging copy
  # and readback would otherwise succeed and return stale output. Push a
  # validation scope around the launch and check the popped result below.
  wgpuDevicePushErrorScope(device, wgpuErrorFilterValidation)
  wgpuComputePassEncoderDispatchWorkgroups(pass, uint32(cfg.grid.x),
                                           uint32(cfg.grid.y), uint32(cfg.grid.z))
  wgpuComputePassEncoderEnd(pass)
  wgpuCommandEncoderCopyBufferToBuffer(encoder, outBuf, 0, stagingBuf, 0, outSize.csize_t)
  let cmdBuf = wgpuCommandEncoderFinish(encoder, nil)

  # Pop the error scope before submitting: wgpu-native reports dispatch
  # validation failures at finish, and an invalid command buffer would
  # otherwise abort at submit. Fail loudly per the unified error policy
  # instead of copying back stale output.
  var scopeData = PopErrorScopeData(done: false)
  var popCbInfo = WGPUPopErrorScopeCallbackInfo(
    mode: wgpuCallbackModeAllowProcessEvents,
    callback: popErrorScopeCb,
    userdata1: scopeData.addr,
    userdata2: nil
  )
  discard wgpuDevicePopErrorScope(device, popCbInfo)
  while not scopeData.done:
    if not wgpuDevicePoll(device, true, nil):
      quit("WebGPU: wgpuDevicePoll failed")
  if scopeData.errType != wgpuErrorTypeNoError:
    failLoud("WebGPU dispatch failed: " & scopeData.message)

  # 8. Submit
  wgpuQueueSubmit(queue, 1, cmdBuf.addr)
  defer:
    wgpuCommandBufferRelease(cmdBuf)

  # 9. Request map + poll to process callback, capture the map status.
  # mapData is heap-allocated: wgpu-native may defer the callback past the
  # poll, so it must never write into dead stack memory. The caller owns the
  # block: it polls, reads, then frees. The callback never frees it.
  let mapData = create(MapDoneData)
  mapData[] = MapDoneData(done: false, resultBytes: outSize,
                          status: wgpuBufferMapAsyncStatusUnknown)
  defer:
    dealloc(mapData)
  var mapCbInfo = WGPUBufferMapCallbackInfo(
    mode: wgpuCallbackModeAllowProcessEvents,
    callback: bufferMapCb,
    userdata1: mapData,
    userdata2: nil
  )
  discard wgpuBufferMapAsync(stagingBuf, wgpuMapModeRead, 0, outSize.csize_t, mapCbInfo)
  # Bounded wait with a deadline (mirrors waitForRequest): the callback may
  # be deferred past the blocking poll, so keep processing events until it
  # fires and quit loudly if it never does.
  if not wgpuDevicePoll(device, true, nil):
    dealloc(mapData)
    quit("WebGPU: wgpuDevicePoll failed")
  let deadline = getMonoTime() + initDuration(seconds = 5)
  while not mapData.done:
    if not wgpuDevicePoll(device, false, nil):
      dealloc(mapData)
      quit("WebGPU: wgpuDevicePoll failed")
    if deadline <= getMonoTime():
      dealloc(mapData)
      quit("WebGPU: timed out waiting for wgpuBufferMapAsync callback")
    sleep(2)
  check mapData.status

  # 10. Read mapped data
  let mappedPtr = wgpuBufferGetMappedRange(stagingBuf, 0, mapData.resultBytes.csize_t)
  if mappedPtr != nil and outSize > 0:
    copyMem(output.data, mappedPtr, outSize)
  wgpuBufferUnmap(stagingBuf)

  # 11. Safety net: errors outside the dispatch scope (buffer creation, queue
  # writes, submit) surface via the uncaptured-error callback installed at
  # device creation. Surface them with the same loud policy.
  let uncaptured = takeUncapturedError()
  if uncaptured.errType != wgpuErrorTypeNoError:
    failLoud("WebGPU uncaptured error: " & uncaptured.message)
