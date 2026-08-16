## Tattletale
## Copyright (c) 2026 Mamy Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Vendored wgpu-native path config + raw WebGPU C API bindings.
##
## Provides:
##   - Path configuration for the wgpu-native shared library
##   - Low-level WGPU types, enums, descriptor structs
##   - Function imports from `libwgpu_native.so`
##
## Public re-exports are consumed by `engines/wgpu.nim`.
##
## Vendor directory layout:
##   vendor/wgpu/
##     lib/libwgpu_native.so
##     include/webgpu/webgpu.h

import std/os

# ═══════════════════════════════════════════════════════════════════════
# Path configuration
# ═══════════════════════════════════════════════════════════════════════

{.used.}

const WgpuVendorDir* = currentSourcePath.parentDir()
  ## Absolute path to the vendored wgpu-native directory.

const WgpuLibPath* = WgpuVendorDir / "wgpu" / "lib"
  ## Absolute path to the directory containing `libwgpu_native.so`

const WgpuIncludePath* = WgpuVendorDir / "wgpu" / "include"
  ## Absolute path to the directory containing `webgpu/webgpu.h`

{.passC: "-I" & WgpuIncludePath.}

# ═══════════════════════════════════════════════════════════════════════
# Opaque handle types
# ═══════════════════════════════════════════════════════════════════════

type
  WGPUInstance*            = pointer
  WGPUAdapter*             = pointer
  WGPUDevice*              = pointer
  WGPUQueue*               = pointer
  WGPUShaderModule*        = pointer
  WGPUComputePipeline*     = pointer
  WGPUBuffer*              = pointer
  WGPUBindGroup*           = pointer
  WGPUBindGroupLayout*     = pointer
  WGPUSampler*             = pointer
  WGPUTextureView*         = pointer
  WGPUPipelineLayout*      = pointer
  WGPUCommandEncoder*      = pointer
  WGPUComputePassEncoder*  = pointer
  WGPUCommandBuffer*       = pointer
  WGPUFuture*              = distinct uint64

  WGPUChainedStruct* {.bycopy.} = object
    next*: ptr WGPUChainedStruct
    sType*: cuint

# ═══════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════

const
  WGPUSType_ShaderSourceWGSL* = 0x00000002'u32

const  # WGPUShaderStage (WGPUFlags = uint64)
  WGPUShaderStageCompute* = 4'u64

const  # WGPUBufferBindingType
  WGPUBufferBindingTypeStorage* = 3'u32
  WGPUBufferBindingTypeReadOnlyStorage* = 4'u32

const  # WGPUMapMode (WGPUFlags = uint64)
  wgpuMapModeRead*: uint64 = 0x0000000000000001'u64

  # WGPUBufferUsage (WGPUFlags = uint64)
  wgpuBufferUsageMapRead*: uint64 = 0x0000000000000001'u64
  wgpuBufferUsageCopySrc*: uint64 = 0x0000000000000004'u64
  wgpuBufferUsageCopyDst*: uint64 = 0x0000000000000008'u64
  wgpuBufferUsageStorage*: uint64 = 0x0000000000000080'u64

# ═══════════════════════════════════════════════════════════════════════
# Descriptor structs
# ═══════════════════════════════════════════════════════════════════════

type
  WGPUInstanceDescriptor* {.bycopy.} = object
    nextInChain*: pointer
  WGPURequestAdapterOptions* {.bycopy.} = object
    nextInChain*: pointer
    compatibleSurface*: pointer
  WGPUDeviceDescriptor* {.bycopy.} = object
    nextInChain*: pointer
    label*: WGPUStringView
    requiredFeatureCount*: csize_t
    requiredFeatures*: pointer
    requiredLimits*: pointer
    defaultQueue*: WGPUQueueDescriptor
    deviceLostCallbackInfo*: WGPUDeviceLostCallbackInfo
    uncapturedErrorCallbackInfo*: WGPUUncapturedErrorCallbackInfo

  WGPUQueueDescriptor* {.bycopy.} = object
    nextInChain*: pointer
    label*: WGPUStringView

  WGPUStringView* {.bycopy.} = object
    data*: cstring
    length*: csize_t

  WGPUAdapterInfo* {.bycopy.} = object
    nextInChain*: ptr WGPUChainedStruct
    vendor*: WGPUStringView
    architecture*: WGPUStringView
    device*: WGPUStringView
    description*: WGPUStringView
    backendType*: cuint
    adapterType*: cuint
    vendorID*: uint32
    deviceID*: uint32
    subgroupMinSize*: uint32
    subgroupMaxSize*: uint32

  WGPUShaderModuleDescriptor* {.bycopy.} = object
    nextInChain*: pointer
    label*: WGPUStringView

  WGPUShaderSourceWGSL* {.bycopy.} = object
    chain*: WGPUChainedStruct
    code*: WGPUStringView

  WGPUComputePipelineDescriptor* {.bycopy.} = object
    nextInChain*: pointer
    label*: WGPUStringView
    layout*: WGPUPipelineLayout
    compute*: WGPUComputeState

  WGPUComputeState* {.bycopy.} = object
    nextInChain*: pointer
    module*: WGPUShaderModule
    entryPoint*: WGPUStringView
    constantCount*: csize_t
    constants*: ptr WGPUConstantEntry

  WGPUConstantEntry* {.bycopy.} = object
    nextInChain*: pointer
    key*: WGPUStringView
    value*: cdouble

  WGPUBufferDescriptor* {.bycopy.} = object
    nextInChain*: pointer
    label*: WGPUStringView
    usage*: uint64
    size*: csize_t
    mappedAtCreation*: bool

  WGPUBindGroupLayoutEntry* {.bycopy.} = object
    nextInChain*: pointer
    binding*: cuint
    visibility*: uint64
    bindingArraySize*: cuint
    buffer*: WGPUBufferBindingLayout
    sampler*: WGPUSamplerBindingLayout
    texture*: WGPUTextureBindingLayout
    storageTexture*: WGPUStorageTextureBindingLayout

  WGPUBufferBindingLayout* {.bycopy.} = object
    nextInChain*: pointer
    `type`*: cuint
    hasDynamicOffset*: bool
    minBindingSize*: csize_t

  WGPUSamplerBindingLayout* {.bycopy.} = object
    nextInChain*: pointer
    `type`*: cuint

  WGPUTextureBindingLayout* {.bycopy.} = object
    nextInChain*: pointer
    sampleType*: cuint
    viewDimension*: cuint
    multisampled*: bool

  WGPUStorageTextureBindingLayout* {.bycopy.} = object
    nextInChain*: pointer
    access*: cuint
    format*: cuint
    viewDimension*: cuint

  WGPUBindGroupLayoutDescriptor* {.bycopy.} = object
    nextInChain*: pointer
    label*: WGPUStringView
    entryCount*: csize_t
    entries*: ptr WGPUBindGroupLayoutEntry

  WGPUPipelineLayoutDescriptor* {.bycopy.} = object
    nextInChain*: pointer
    label*: WGPUStringView
    bindGroupLayoutCount*: csize_t
    bindGroupLayouts*: ptr WGPUBindGroupLayout

  WGPUBindGroupEntry* {.bycopy.} = object
    nextInChain*: pointer
    binding*: cuint
    buffer*: WGPUBuffer
    offset*: csize_t
    size*: csize_t
    sampler*: WGPUSampler
    textureView*: WGPUTextureView

  WGPUBindGroupDescriptor* {.bycopy.} = object
    nextInChain*: pointer
    label*: WGPUStringView
    layout*: WGPUBindGroupLayout
    entryCount*: csize_t
    entries*: ptr WGPUBindGroupEntry

  WGPUCommandEncoderDescriptor* {.bycopy.} = object
    nextInChain*: pointer
    label*: WGPUStringView

  WGPUComputePassDescriptor* {.bycopy.} = object
    nextInChain*: pointer
    label*: WGPUStringView

  WGPUCommandBufferDescriptor* {.bycopy.} = object
    nextInChain*: pointer
    label*: WGPUStringView

  WGPUCallbackMode* {.size: sizeof(cuint).} = enum
    wgpuCallbackModeInvalid            = 0
    wgpuCallbackModeWaitAnyOnly        = 1
    wgpuCallbackModeAllowProcessEvents = 2
    wgpuCallbackModeAllowSpontaneous   = 3


  WGPURequestAdapterStatus* {.size: sizeof(cuint).} = enum
    wgpuRequestAdapterStatusSuccess          = 1
    wgpuRequestAdapterStatusCallbackCancelled = 2
    wgpuRequestAdapterStatusUnavailable       = 3
    wgpuRequestAdapterStatusError             = 4

  WGPURequestDeviceStatus* {.size: sizeof(cuint).} = enum
    wgpuRequestDeviceStatusSuccess          = 1
    wgpuRequestDeviceStatusCallbackCancelled = 2
    wgpuRequestDeviceStatusError             = 3


  WGPUBufferMapAsyncStatus* {.size: sizeof(cuint).} = enum
    wgpuBufferMapAsyncStatusSuccess                   = 1
    wgpuBufferMapAsyncStatusCallbackCancelled          = 2
    wgpuBufferMapAsyncStatusError                      = 3
    wgpuBufferMapAsyncStatusMappingAlreadyPending      = 4
    wgpuBufferMapAsyncStatusOffsetOutOfRange           = 5
    wgpuBufferMapAsyncStatusSizeOutOfRange              = 6
    wgpuBufferMapAsyncStatusInvalidOperation            = 7
    wgpuBufferMapAsyncStatusDeviceLost                  = 8
    wgpuBufferMapAsyncStatusDestroyedBeforeCallback     = 9
    wgpuBufferMapAsyncStatusUnknown                     = 10

  WGPUErrorFilter* {.size: sizeof(cuint).} = enum
    wgpuErrorFilterValidation  = 1
    wgpuErrorFilterOutOfMemory = 2
    wgpuErrorFilterInternal    = 3

  WGPUErrorType* {.size: sizeof(cuint).} = enum
    wgpuErrorTypeNoError       = 1
    wgpuErrorTypeValidation    = 2
    wgpuErrorTypeOutOfMemory   = 3
    wgpuErrorTypeInternal      = 4
    wgpuErrorTypeUnknown       = 5

  WGPUPopErrorScopeStatus* {.size: sizeof(cuint).} = enum
    wgpuPopErrorScopeStatusSuccess           = 1
    wgpuPopErrorScopeStatusCallbackCancelled = 2
    wgpuPopErrorScopeStatusError             = 3

  WGPURequestAdapterCallbackInfo* {.bycopy.} = object
    nextInChain*: pointer
    mode*: WGPUCallbackMode
    callback*: pointer
    userdata1*: pointer
    userdata2*: pointer

  WGPURequestDeviceCallbackInfo* {.bycopy.} = object
    nextInChain*: pointer
    mode*: WGPUCallbackMode
    callback*: pointer
    userdata1*: pointer
    userdata2*: pointer

  WGPUBufferMapCallbackInfo* {.bycopy.} = object
    nextInChain*: pointer
    mode*: WGPUCallbackMode
    callback*: pointer
    userdata1*: pointer
    userdata2*: pointer

  WGPUDeviceLostCallbackInfo* {.bycopy.} = object
    nextInChain*: pointer
    mode*: WGPUCallbackMode
    callback*: pointer
    userdata1*: pointer
    userdata2*: pointer

  WGPUUncapturedErrorCallbackInfo* {.bycopy.} = object
    nextInChain*: pointer
    callback*: pointer
    userdata1*: pointer
    userdata2*: pointer

  WGPUPopErrorScopeCallbackInfo* {.bycopy.} = object
    nextInChain*: pointer
    mode*: WGPUCallbackMode
    callback*: pointer
    userdata1*: pointer
    userdata2*: pointer

# ═══════════════════════════════════════════════════════════════════════
# Function imports from libwgpu_native.so
# ═══════════════════════════════════════════════════════════════════════

const libWgpu = (
  when defined(windows): WgpuLibPath / "wgpu_native.dll"
  elif defined(macosx):  WgpuLibPath / "libwgpu_native.dylib"
  else:                  WgpuLibPath / "libwgpu_native.so"
)

# --- Instance / Adapter / Device ---
proc wgpuCreateInstance*(desc: ptr WGPUInstanceDescriptor): WGPUInstance
  {.importc: "wgpuCreateInstance", dynlib: libWgpu.}

proc wgpuInstanceRequestAdapter*(
    instance: WGPUInstance,
    options: ptr WGPURequestAdapterOptions,
    callbackInfo: WGPURequestAdapterCallbackInfo): WGPUFuture
  {.importc: "wgpuInstanceRequestAdapter", dynlib: libWgpu.}

proc wgpuAdapterRequestDevice*(
    adapter: WGPUAdapter,
    descriptor: ptr WGPUDeviceDescriptor,
    callbackInfo: WGPURequestDeviceCallbackInfo): WGPUFuture
  {.importc: "wgpuAdapterRequestDevice", dynlib: libWgpu.}

proc wgpuAdapterGetInfo*(adapter: WGPUAdapter, info: ptr WGPUAdapterInfo)
  {.importc: "wgpuAdapterGetInfo", dynlib: libWgpu.}

proc wgpuInstanceProcessEvents*(instance: WGPUInstance)
  {.importc: "wgpuInstanceProcessEvents", dynlib: libWgpu.}

proc wgpuDeviceGetQueue*(device: WGPUDevice): WGPUQueue
  {.importc: "wgpuDeviceGetQueue", dynlib: libWgpu.}

# --- Error scopes ---
proc wgpuDevicePushErrorScope*(device: WGPUDevice, filter: WGPUErrorFilter)
  {.importc: "wgpuDevicePushErrorScope", dynlib: libWgpu.}

proc wgpuDevicePopErrorScope*(device: WGPUDevice,
                              callbackInfo: WGPUPopErrorScopeCallbackInfo): WGPUFuture
  {.importc: "wgpuDevicePopErrorScope", dynlib: libWgpu.}

# --- Shader & Pipeline ---
proc wgpuDeviceCreateShaderModule*(device: WGPUDevice,
                                    desc: ptr WGPUShaderModuleDescriptor): WGPUShaderModule
  {.importc: "wgpuDeviceCreateShaderModule", dynlib: libWgpu.}

proc wgpuDeviceCreateComputePipeline*(device: WGPUDevice,
                                       desc: ptr WGPUComputePipelineDescriptor): WGPUComputePipeline
  {.importc: "wgpuDeviceCreateComputePipeline", dynlib: libWgpu.}

# --- Buffers ---
proc wgpuDeviceCreateBuffer*(device: WGPUDevice,
                              desc: ptr WGPUBufferDescriptor): WGPUBuffer
  {.importc: "wgpuDeviceCreateBuffer", dynlib: libWgpu.}


# --- Bind groups ---
proc wgpuDeviceCreateBindGroupLayout*(device: WGPUDevice,
                                       desc: ptr WGPUBindGroupLayoutDescriptor): WGPUBindGroupLayout
  {.importc: "wgpuDeviceCreateBindGroupLayout", dynlib: libWgpu.}

proc wgpuDeviceCreatePipelineLayout*(device: WGPUDevice,
                                      desc: ptr WGPUPipelineLayoutDescriptor): WGPUPipelineLayout
  {.importc: "wgpuDeviceCreatePipelineLayout", dynlib: libWgpu.}

proc wgpuDeviceCreateBindGroup*(device: WGPUDevice,
                                 desc: ptr WGPUBindGroupDescriptor): WGPUBindGroup
  {.importc: "wgpuDeviceCreateBindGroup", dynlib: libWgpu.}

# --- Command recording ---
proc wgpuDeviceCreateCommandEncoder*(device: WGPUDevice,
                                      desc: ptr WGPUCommandEncoderDescriptor): WGPUCommandEncoder
  {.importc: "wgpuDeviceCreateCommandEncoder", dynlib: libWgpu.}

proc wgpuCommandEncoderBeginComputePass*(encoder: WGPUCommandEncoder,
                                          desc: ptr WGPUComputePassDescriptor): WGPUComputePassEncoder
  {.importc: "wgpuCommandEncoderBeginComputePass", dynlib: libWgpu.}

proc wgpuComputePassEncoderSetPipeline*(encoder: WGPUComputePassEncoder,
                                         pipeline: WGPUComputePipeline)
  {.importc: "wgpuComputePassEncoderSetPipeline", dynlib: libWgpu.}

proc wgpuComputePassEncoderSetBindGroup*(encoder: WGPUComputePassEncoder,
                                          groupIndex: cuint,
                                          group: WGPUBindGroup,
                                          dynamicOffsetCount: csize_t,
                                          dynamicOffsets: pointer)
  {.importc: "wgpuComputePassEncoderSetBindGroup", dynlib: libWgpu.}

proc wgpuComputePassEncoderDispatchWorkgroups*(encoder: WGPUComputePassEncoder,
                                                x: cuint, y: cuint, z: cuint)
  {.importc: "wgpuComputePassEncoderDispatchWorkgroups", dynlib: libWgpu.}

proc wgpuComputePassEncoderEnd*(encoder: WGPUComputePassEncoder)
  {.importc: "wgpuComputePassEncoderEnd", dynlib: libWgpu.}

proc wgpuCommandEncoderFinish*(encoder: WGPUCommandEncoder,
                                desc: ptr WGPUCommandBufferDescriptor): WGPUCommandBuffer
  {.importc: "wgpuCommandEncoderFinish", dynlib: libWgpu.}

# --- Queue submission ---

proc wgpuQueueSubmit*(queue: WGPUQueue,
                       commandCount: csize_t,
                       commands: ptr WGPUCommandBuffer)
  {.importc: "wgpuQueueSubmit", dynlib: libWgpu.}

# --- Buffer mapping ---
proc wgpuQueueWriteBuffer*(
    queue: WGPUQueue,
    buffer: WGPUBuffer,
    offset: uint64,
    data: pointer,
    size: csize_t)
  {.importc: "wgpuQueueWriteBuffer", dynlib: libWgpu.}

proc wgpuBufferMapAsync*(
    buffer: WGPUBuffer,
    mode: uint64,
    offset: csize_t,
    size: csize_t,
    callbackInfo: WGPUBufferMapCallbackInfo): WGPUFuture
  {.importc: "wgpuBufferMapAsync", dynlib: libWgpu.}

proc wgpuBufferGetMappedRange*(buffer: WGPUBuffer,
                                      offset: csize_t,
                                      size: csize_t): pointer
  {.importc: "wgpuBufferGetMappedRange", dynlib: libWgpu.}

proc wgpuBufferUnmap*(buffer: WGPUBuffer)
  {.importc: "wgpuBufferUnmap", dynlib: libWgpu.}

# --- Polling ---
proc wgpuDevicePoll*(device: WGPUDevice,
                      wait: bool,
                      submissionIndex: ptr uint64): bool
  {.importc: "wgpuDevicePoll", dynlib: libWgpu.}

proc wgpuCommandEncoderCopyBufferToBuffer*(
    encoder: WGPUCommandEncoder,
    source: WGPUBuffer,
    sourceOffset: csize_t,
    destination: WGPUBuffer,
    destinationOffset: csize_t,
    size: csize_t)
  {.importc: "wgpuCommandEncoderCopyBufferToBuffer", dynlib: libWgpu.}

# --- Destructors ---
proc wgpuInstanceRelease*(instance: WGPUInstance)
  {.importc: "wgpuInstanceRelease", dynlib: libWgpu.}

proc wgpuAdapterRelease*(adapter: WGPUAdapter)
  {.importc: "wgpuAdapterRelease", dynlib: libWgpu.}

proc wgpuDeviceRelease*(device: WGPUDevice)
  {.importc: "wgpuDeviceRelease", dynlib: libWgpu.}

proc wgpuShaderModuleRelease*(module: WGPUShaderModule)
  {.importc: "wgpuShaderModuleRelease", dynlib: libWgpu.}

proc wgpuComputePipelineRelease*(pipeline: WGPUComputePipeline)
  {.importc: "wgpuComputePipelineRelease", dynlib: libWgpu.}

proc wgpuBindGroupLayoutRelease*(layout: WGPUBindGroupLayout)
  {.importc: "wgpuBindGroupLayoutRelease", dynlib: libWgpu.}

proc wgpuPipelineLayoutRelease*(layout: WGPUPipelineLayout)
  {.importc: "wgpuPipelineLayoutRelease", dynlib: libWgpu.}

proc wgpuBindGroupRelease*(group: WGPUBindGroup)
  {.importc: "wgpuBindGroupRelease", dynlib: libWgpu.}

proc wgpuCommandEncoderRelease*(encoder: WGPUCommandEncoder)
  {.importc: "wgpuCommandEncoderRelease", dynlib: libWgpu.}

proc wgpuCommandBufferRelease*(cmdBuf: WGPUCommandBuffer)
  {.importc: "wgpuCommandBufferRelease", dynlib: libWgpu.}

proc wgpuBufferRelease*(buf: WGPUBuffer)
  {.importc: "wgpuBufferRelease", dynlib: libWgpu.}
