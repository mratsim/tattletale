## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
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
## Example (engine API):
##   import workspace/crucible
##   const code = wgsl:
##     proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
##       output[0] = a[0] + b[0]
##   var engine = bkWGSL.init()
##   engine.ingest(code)
##   var out: array[1, uint32]
##   engine.run("addKernel", out, ([1'u32], [2'u32]))

## Types and procs follow the standard webgpu.h conventions.
import workspace/crucible/vendor/wgpu

# ###########################################################
#
#    High-level: WgpuContext
#
# ###########################################################
type
  AdapterCallbackData = object
    adapter: WGPUAdapter
    done: bool
  DeviceCallbackData = object
    device: WGPUDevice
    done: bool
  MapDoneData* = object
    ## Map-callback state — captured by `bufferMapCb` for the engine's poll loop.
    done*: bool
    resultBytes*: int
    status*: WGPUBufferMapAsyncStatus
  WgpuContext* = object
    ## The live wgpu handles (instance → adapter → device → queue).
    instance*: WGPUInstance
    adapter*: WGPUAdapter
    device*: WGPUDevice
    queue*: WGPUQueue

proc deviceName*(ctx: WgpuContext): string =
  ## The WebGPU adapter device name (e.g. "NVIDIA RTX PRO 6000 ...").
  var info: WGPUAdapterInfo
  wgpuAdapterGetInfo(ctx.adapter, addr info)
  if info.device.data != nil and info.device.length > 0:
    result = newString(info.device.length.int)
    copyMem(result[0].addr, info.device.data, info.device.length)

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
proc bufferMapCb*(status: WGPUBufferMapAsyncStatus,
                  message: WGPUStringView,
                  userdata1: pointer,
                  userdata2: pointer) {.cdecl.} =
  ## Records the map result into the caller's MapDoneData (userdata1).
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
  ## Releases all wgpu resources. Idempotent; the engine's `=destroy` calls it.
  if ctx.instance != nil:
    wgpuDeviceRelease(ctx.device)
    wgpuAdapterRelease(ctx.adapter)
    wgpuInstanceRelease(ctx.instance)
    ctx.device = nil
    ctx.adapter = nil
    ctx.instance = nil
