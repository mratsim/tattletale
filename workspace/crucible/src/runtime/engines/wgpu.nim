# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## WgpuEngine — WebGPU (wgpu-native) execution (moved from codegen/wgpu.nim,
## decoupled from the compile-time DSL: no gpu_compiler import).
##
## ingest = store WGSL; getArtifact = WGSL; run = pipeline +
## dispatchWorkgroups where grid = dispatchWorkgroups count. blk is
## shader-baked (@workgroup_size): run's blk is validated loudly. The
## staging-buffer map pattern is kept, with the map status captured in
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

import std/strutils

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

  WgpuEngine* = ref object
    ## Fields directly (no Obj indirection); resources in the RAII value field.
    source: string
    ctx: WgpuCtx
    bakedBlk: Dim3   # workgroup size baked into the shader at ingest

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════
proc `=destroy`(c: var WgpuCtx) =
  if c.ctx.instance != nil:
    c.ctx.shutdown()

proc newWgpuEngine(): WgpuEngine =
  ## Private factory — engines.nim reaches it via `import {.all.}`.
  WgpuEngine(ctx: WgpuCtx(ctx: initWgpu()))

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

template check*(status: WGPUBufferMapAsyncStatus, quitOnFailure = true) =
  let code = status
  if code != wgpuBufferMapAsyncStatusSuccess:
    writeStackTrace()
    stderr.write($instantiationInfo() & " exited with error: WGPUBufferMapAsyncStatus " & $code & '\n')
    if quitOnFailure:
      quit 1

proc ingest*(engine: WgpuEngine, source: string) =
  ## Store the WGSL source. Re-entrant: replaces the previous artifact
  ## (the device context persists — only the artifact is replaced).
  engine.source = source
  engine.bakedBlk = parseBakedBlk(source)

proc getArtifact*(engine: WgpuEngine): string =
  ## The WGSL kernel source.
  engine.source

template run*[T](engine: WgpuEngine, kernel: string, output: var T, args: untyped,
              cfg: LaunchConfig): untyped =
  var blobStorage: seq[byte]   # backing store for by-value scalars; lives until scope exit
  runImpl(engine, kernel, outBlob(output), flattenBlobs(args, blobStorage), cfg)

template run*[T](engine: WgpuEngine, kernel: string, output: var T, args: untyped): untyped =
  run(engine, kernel, output, args,
      LaunchConfig(blk: engine.bakedBlk))

# ─────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────
# ▸ PRIVATE
# ─────────────────────────────────────────────────────────────────────────

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

  # blk is shader-baked (@workgroup_size): validate loudly (relax later)
  if engine.bakedBlk.x == 0 or
     cfg.blk.x != engine.bakedBlk.x or cfg.blk.y != engine.bakedBlk.y or
     cfg.blk.z != engine.bakedBlk.z:
    quit("WebGPU run blk=" & $cfg.blk.x & "x" & $cfg.blk.y & "x" & $cfg.blk.z &
         " != baked workgroup size " & $engine.bakedBlk.x & "x" & $engine.bakedBlk.y &
         "x" & $engine.bakedBlk.z &
         " — launch config mismatch (blk is shader-baked on WebGPU)")

  # 1. Create shader module (chained WGPUShaderSourceWGSL)
  var wgslView = WGPUStringView(data: cstring(engine.source), length: engine.source.len.csize_t)
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
  let shader = wgpuDeviceCreateShaderModule(device, addr shaderDesc)
  if shader == nil:
    quit("WebGPU: failed to create shader module")
  defer:
    wgpuShaderModuleRelease(shader)

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

  # 3. Bind group layout: output at 0, then inputs
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
    entries[i + 1] = WGPUBindGroupLayoutEntry(
      binding: (i + 1).cuint,
      visibility: WGPUShaderStageCompute,
      buffer: WGPUBufferBindingLayout(
        `type`: WGPUBufferBindingTypeStorage,
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
  defer:
    wgpuBindGroupLayoutRelease(bgl)

  var plDesc = WGPUPipelineLayoutDescriptor(
    bindGroupLayoutCount: 1,
    bindGroupLayouts: bgl.addr
  )
  let pl = wgpuDeviceCreatePipelineLayout(device, addr plDesc)
  if pl == nil:
    quit("WebGPU: failed to create pipeline layout")
  defer:
    wgpuPipelineLayoutRelease(pl)

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

  # 5. Compute pipeline
  var entryPtView = WGPUStringView(data: cstring(kernel), length: kernel.len.csize_t)
  var computeState = WGPUComputeState(
    module: shader,
    entryPoint: entryPtView
  )
  var cpDesc = WGPUComputePipelineDescriptor(
    layout: pl,
    compute: computeState
  )
  let pipeline = wgpuDeviceCreateComputePipeline(device, addr cpDesc)
  if pipeline == nil:
    quit("WebGPU: failed to create compute pipeline")
  defer:
    wgpuComputePipelineRelease(pipeline)

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
  wgpuComputePassEncoderDispatchWorkgroups(pass, uint32(cfg.grid.x),
                                           uint32(cfg.grid.y), uint32(cfg.grid.z))
  wgpuComputePassEncoderEnd(pass)
  wgpuCommandEncoderCopyBufferToBuffer(encoder, outBuf, 0, stagingBuf, 0, outSize.csize_t)
  let cmdBuf = wgpuCommandEncoderFinish(encoder, nil)

  # 8. Submit
  wgpuQueueSubmit(queue, 1, cmdBuf.addr)
  defer:
    wgpuCommandBufferRelease(cmdBuf)

  # 9. Request map + poll to process callback — capture the map status
  var mapData = MapDoneData(done: false, resultBytes: outSize,
                            status: wgpuBufferMapAsyncStatusUnknown)
  var mapCbInfo = WGPUBufferMapCallbackInfo(
    mode: wgpuCallbackModeAllowProcessEvents,
    callback: bufferMapCb,
    userdata1: mapData.addr,
    userdata2: nil
  )
  discard wgpuBufferMapAsync(stagingBuf, wgpuMapModeRead, 0, outSize.csize_t, mapCbInfo)
  if not wgpuDevicePoll(device, true, nil):
    quit("WebGPU: wgpuDevicePoll failed")
  if not mapData.done:
    quit("WebGPU: wgpuBufferMapAsync callback never fired")
  check mapData.status

  # 10. Read mapped data
  let mappedPtr = wgpuBufferGetMappedRange(stagingBuf, 0, mapData.resultBytes.csize_t)
  if mappedPtr != nil and outSize > 0:
    copyMem(output.data, mappedPtr, outSize)
  wgpuBufferUnmap(stagingBuf)
