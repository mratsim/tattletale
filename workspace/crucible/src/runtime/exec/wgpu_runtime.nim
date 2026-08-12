## Constantine
## Copyright (c) 2018-2019    Status Research & Development GmbH
## Copyright (c) 2020-Present Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
## WebGPU (wgpu-native) execution DSL.
##
## Provides a Nim wrapper around `webgpu.h` (wgpu-native C API) for
## compiling and executing WGSL compute shaders on CPU (via SwiftShader/
## wgpu CPU backend) or GPU — no GPU required for CI.
##
## Library:  libwgpu_native.so  (https://github.com/gfx-rs/wgpu-native)
##
## Types and procs follow the standard webgpu.h conventions.
import std/[macros, os]
from std/strutils import normalize
import workspace/crucible/vendor/wgpu

# ###########################################################
#
#    High-level: WgpuContext
#
# ###########################################################
type
  AdapterCallbackData* = object
    adapter: WGPUAdapter
    done: bool
  DeviceCallbackData* = object
    device: WGPUDevice
    done: bool
  WorkDoneData* = object
    done: bool
  MapDoneData* = object
    done*: bool
    resultBytes*: int
    status*: WGPUBufferMapAsyncStatus  ## captured — the old execWgpu dropped it
  WgpuContext* = object
    instance*: WGPUInstance
    adapter*: WGPUAdapter
    device*: WGPUDevice
    queue*: WGPUQueue

{.push stackTrace: off.}

proc adapterCb(status: WGPURequestAdapterStatus,
               adapter: WGPUAdapter,
               message: WGPUStringView,
               userdata1: pointer,
               userdata2: pointer) {.cdecl.} =
  if status == wgpuRequestAdapterStatusSuccess:
    cast[ptr AdapterCallbackData](userdata1).adapter = adapter
  cast[ptr AdapterCallbackData](userdata1).done = true
proc deviceCb(status: WGPURequestDeviceStatus,
              device: WGPUDevice,
              message: WGPUStringView,
              userdata1: pointer,
              userdata2: pointer) {.cdecl.} =
  if status == wgpuRequestDeviceStatusSuccess:
    cast[ptr DeviceCallbackData](userdata1).device = device
  cast[ptr DeviceCallbackData](userdata1).done = true
proc workDoneCb(status: WGPUQueueWorkDoneStatus,
                 message: WGPUStringView,
                 userdata1: pointer,
                 userdata2: pointer) {.cdecl.} =
  cast[ptr WorkDoneData](userdata1).done = true
proc bufferMapCb*(status: WGPUBufferMapAsyncStatus,
                   message: WGPUStringView,
                   userdata1: pointer,
                   userdata2: pointer) {.cdecl.} =
  cast[ptr MapDoneData](userdata1).status = status
  cast[ptr MapDoneData](userdata1).done = true

{.pop.}

proc initWgpu*(): WgpuContext =
  ## Initializes wgpu-native: creates instance, picks adapter, opens device.
  let instance = wgpuCreateInstance(nil)
  doAssert instance != nil, "wgpuCreateInstance failed"
  var ad = AdapterCallbackData(done: false)
  var cbInfo = WGPURequestAdapterCallbackInfo(
    mode: wgpuCallbackModeAllowProcessEvents,
    callback: adapterCb,
    userdata1: ad.addr,
    userdata2: nil
  )
  discard wgpuInstanceRequestAdapter(instance, nil, cbInfo)
  # Poll until adapter request completes (wgpu-native stubs WGPUFuture/WaitAnyOnly)
  while not ad.done:
    wgpuInstanceProcessEvents(instance)
  doAssert ad.adapter != nil, "No suitable WebGPU adapter found"
  var dd = DeviceCallbackData(done: false)
  var devCbInfo = WGPURequestDeviceCallbackInfo(
    mode: wgpuCallbackModeAllowProcessEvents,
    callback: deviceCb,
    userdata1: dd.addr,
    userdata2: nil
  )
  discard wgpuAdapterRequestDevice(ad.adapter, nil, devCbInfo)
  # Poll until device request completes
  while not dd.done:
    wgpuInstanceProcessEvents(instance)
  doAssert dd.device != nil, "Failed to get WebGPU device"
  let queue = wgpuDeviceGetQueue(dd.device)
  doAssert queue != nil, "wgpuDeviceGetQueue failed"
  result = WgpuContext(
    instance: instance,
    adapter: ad.adapter,
    device: dd.device,
    queue: queue
  )
proc shutdown*(ctx: var WgpuContext) =
  ## Releases all wgpu resources. Idempotent (safe with =destroy hooks).
  if ctx.instance != nil:
    wgpuDeviceRelease(ctx.device)
    wgpuAdapterRelease(ctx.adapter)
    wgpuInstanceRelease(ctx.instance)
    ctx.device = nil
    ctx.adapter = nil
    ctx.instance = nil
# ###########################################################
#
#    execWgpu — run a WGSL compute shader with input/output buffers
#
# ###########################################################
proc hasStorageBufferUsage(usage: uint64): bool {.inline.} =
  ## Checks if the given usage flags include Storage.
  (usage and wgpuBufferUsageStorage) != 0
proc execWgpu*(ctx: var WgpuContext,
               wgsl: string,
               entryPoint: string,
               outputBytes: int,
               inputs: openArray[tuple[data: pointer, size: int]]): seq[byte] =
  ## Compiles and executes a WGSL compute shader.
  ##
  ## - `wgsl`:       the WGSL source code
  ## - `entryPoint`: name of the compute entry point
  ## - `outputBytes`: number of bytes to read back as result
  ## - `inputs`:     sequence of (pointer, size) tuples for input buffers
  ##
  ## Returns the output buffer contents as a seq[byte].
  ##
  ## Bindings follow WGSL parameter order:
  ##   binding 0..N-1 = inputs (in order), binding N = output.
  let device = ctx.device
  let queue  = ctx.queue
  let outBytes = outputBytes
  # 1. Create shader module (chained WGPUShaderSourceWGSL)
  var wgslView = WGPUStringView(data: cstring(wgsl), length: wgsl.len.csize_t)
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
  doAssert shader != nil, "Failed to create shader module"
  defer:
    wgpuShaderModuleRelease(shader)

  # 2. Create buffers via staging pattern (no Map usage on storage buffers)
  let numInputs = inputs.len
  let totalBindings = numInputs + 1
  var inputBuffers = newSeq[WGPUBuffer](numInputs)
  for i in 0 ..< numInputs:
    let desc = WGPUBufferDescriptor(
      usage: wgpuBufferUsageStorage or wgpuBufferUsageCopyDst,
      size: inputs[i].size.csize_t,
      mappedAtCreation: false
    )
    inputBuffers[i] = wgpuDeviceCreateBuffer(device, addr desc)
    doAssert inputBuffers[i] != nil, "Failed to create input buffer"
  defer:
    for buf in inputBuffers:
      wgpuBufferRelease(buf)

  # Output buffer: shader writes here, then we copy to staging
  var outBufDesc = WGPUBufferDescriptor(
    usage: wgpuBufferUsageStorage or wgpuBufferUsageCopySrc,
    size: outBytes.csize_t,
    mappedAtCreation: false)
  let outBuf = wgpuDeviceCreateBuffer(device, addr outBufDesc)
  doAssert outBuf != nil, "Failed to create output buffer"
  defer:
    wgpuBufferRelease(outBuf)

  # Staging buffer: copy output here, then map for CPU reading
  var stagingDesc = WGPUBufferDescriptor(
    usage: wgpuBufferUsageMapRead or wgpuBufferUsageCopyDst,
    size: outBytes.csize_t,
    mappedAtCreation: false)
  let stagingBuf = wgpuDeviceCreateBuffer(device, addr stagingDesc)
  doAssert stagingBuf != nil, "Failed to create staging buffer"
  defer:
    wgpuBufferRelease(stagingBuf)
  # 4. Create bind group layout: inputs 0..N-1, output N
  var entries = newSeq[WGPUBindGroupLayoutEntry](totalBindings)
  for i in 0 ..< numInputs:
    entries[i] = WGPUBindGroupLayoutEntry(
      binding: i.cuint,
      visibility: 4,  # WGPUShaderStage::Compute
      buffer: WGPUBufferBindingLayout(
        `type`: 3,  # WGPUBufferBindingType_Storage
        minBindingSize: inputs[i].size.csize_t
      )
    )
  entries[numInputs] = WGPUBindGroupLayoutEntry(
    binding: numInputs.cuint,
    visibility: 4,
    buffer: WGPUBufferBindingLayout(
      `type`: 3,  # Storage
      minBindingSize: outBytes.csize_t
    )
  )
  var bglDesc = WGPUBindGroupLayoutDescriptor(
    entryCount: totalBindings.csize_t,
    entries: entries[0].addr
  )
  let bgl = wgpuDeviceCreateBindGroupLayout(device, addr bglDesc)
  doAssert bgl != nil, "Failed to create bind group layout"
  defer:
    wgpuBindGroupLayoutRelease(bgl)

  var plDesc = WGPUPipelineLayoutDescriptor(
    bindGroupLayoutCount: 1,
    bindGroupLayouts: bgl.addr
  )
  let pl = wgpuDeviceCreatePipelineLayout(device, addr plDesc)
  doAssert pl != nil, "Failed to create pipeline layout"
  defer:
    wgpuPipelineLayoutRelease(pl)

  # 5. Create bind group entries (same order: inputs then output)
  var bgEntries = newSeq[WGPUBindGroupEntry](totalBindings)
  for i in 0 ..< numInputs:
    bgEntries[i] = WGPUBindGroupEntry(
      binding: i.cuint,
      buffer: inputBuffers[i],
      size: inputs[i].size.csize_t
    )
  bgEntries[numInputs] = WGPUBindGroupEntry(
    binding: numInputs.cuint,
    buffer: outBuf,
    size: outBytes.csize_t
  )
  var bgDesc = WGPUBindGroupDescriptor(
    layout: bgl,
    entryCount: totalBindings.csize_t,
    entries: bgEntries[0].addr
  )
  let bg = wgpuDeviceCreateBindGroup(device, addr bgDesc)
  doAssert bg != nil, "Failed to create bind group"
  defer:
    wgpuBindGroupRelease(bg)

  # 6. Create compute pipeline
  var entryPtView = WGPUStringView(data: cstring(entryPoint), length: entryPoint.len.csize_t)
  var computeState = WGPUComputeState(
    module: shader,
    entryPoint: entryPtView
  )
  var cpDesc = WGPUComputePipelineDescriptor(
    layout: pl,
    compute: computeState
  )
  let pipeline = wgpuDeviceCreateComputePipeline(device, addr cpDesc)
  doAssert pipeline != nil, "Failed to create compute pipeline"
  defer:
    wgpuComputePipelineRelease(pipeline)

  # 7. Write input data before recording (uses queue, not encoder)
  for i in 0 ..< numInputs:
    wgpuQueueWriteBuffer(queue, inputBuffers[i], 0, inputs[i].data, inputs[i].size.csize_t)
  # 8. Record commands: compute pass + copy output → staging
  let encoder = wgpuDeviceCreateCommandEncoder(device, nil)
  doAssert encoder != nil, "Failed to create command encoder"
  defer:
    wgpuCommandEncoderRelease(encoder)

  let pass = wgpuCommandEncoderBeginComputePass(encoder, nil)
  wgpuComputePassEncoderSetPipeline(pass, pipeline)
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nil)
  # Compute workgroup count from output size (1 workgroup = 256 threads)
  let wgs = 256'u32
  let totalThreads = ((outBytes.uint32 + 3'u32) div 4'u32).max(wgs)
  let numWorkgroups = (totalThreads + wgs - 1'u32) div wgs
  wgpuComputePassEncoderDispatchWorkgroups(pass, numWorkgroups, 1'u32, 1'u32)
  wgpuComputePassEncoderEnd(pass)
  wgpuCommandEncoderCopyBufferToBuffer(encoder, outBuf, 0, stagingBuf, 0, outBytes.csize_t)
  let cmdBuf = wgpuCommandEncoderFinish(encoder, nil)
  # 9. Submit
  wgpuQueueSubmit(queue, 1, cmdBuf.addr)
  defer:
    wgpuCommandBufferRelease(cmdBuf)

  # 10. Request map + poll to process callback
  var mapData = MapDoneData(done: false, resultBytes: outBytes)
  var mapCbInfo = WGPUBufferMapCallbackInfo(
    mode: wgpuCallbackModeAllowProcessEvents,
    callback: bufferMapCb,
    userdata1: mapData.addr,
    userdata2: nil
  )
  discard wgpuBufferMapAsync(stagingBuf, wgpuMapModeRead, 0, outBytes.csize_t, mapCbInfo)
  doAssert wgpuDevicePoll(device, true, nil), "wgpuDevicePoll failed"
  doAssert mapData.done, "wgpuBufferMapAsync callback never fired"
  # 11. Read mapped data
  let mappedPtr = wgpuBufferGetMappedRange(stagingBuf, 0, mapData.resultBytes.csize_t)
  result = newSeq[byte](mapData.resultBytes)
  if mappedPtr != nil:
    copyMem(result[0].addr, mappedPtr, mapData.resultBytes)
  wgpuBufferUnmap(stagingBuf)
