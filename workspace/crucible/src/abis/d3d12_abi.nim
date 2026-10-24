# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Minimal Nim bindings for the D3D12 API subset needed for compute shaders,
## dynamically linked via d3d12.dll through std/dynlib.

import std/dynlib
import ../runtime/errors

# ═══════════════════════════════════════════════════════════════════════
# Scalar types
# ═══════════════════════════════════════════════════════════════════════

type
  HRESULT* = int32
  UINT* = uint32
  UINT64* = uint64
  SIZE_T* = csize_t
  HANDLE* = pointer
  # NUL-terminated UTF-16 string, not a cstring.
  # Nim's cstring maps to char*, a UTF-8 byte string on the C side.
  LPCWSTR* = pointer
  DXGI_FORMAT* = int32
  D3D_FEATURE_LEVEL* = uint32
  D3D12_GPU_VIRTUAL_ADDRESS* = uint64

# ═══════════════════════════════════════════════════════════════════════
# GUIDs
# ═══════════════════════════════════════════════════════════════════════

type
  GUID* = object
    data1*: uint32
    data2*: uint16
    data3*: uint16
    data4*: array[8, uint8]

proc guid*(data1: uint32, data2: uint16, data3: uint16, data4: array[8, uint8]): GUID {.inline.} =
  GUID(data1: data1, data2: data2, data3: data3, data4: data4)

const
  IID_ID3D12Device* = guid(0x189819F1'u32, 0x1DB6'u16, 0x4B57'u16,
      [0xBE'u8, 0x54, 0x18, 0x21, 0x33, 0x9B, 0x85, 0xF7])
  IID_ID3D12RootSignature* = guid(0xC54A6B66'u32, 0x72DF'u16, 0x4EE8'u16,
      [0x8B'u8, 0xE5, 0xA9, 0x46, 0xA1, 0x42, 0x92, 0x14])
  IID_ID3D12Resource* = guid(0x696442BE'u32, 0xA72E'u16, 0x4059'u16,
      [0xBC'u8, 0x79, 0x5B, 0x5C, 0x98, 0x04, 0x0F, 0xAD])
  IID_ID3D12CommandAllocator* = guid(0x6102DEE4'u32, 0xAF59'u16, 0x4B09'u16,
      [0xB9'u8, 0x99, 0xB4, 0x4D, 0x73, 0xF0, 0x9B, 0x24])
  IID_ID3D12GraphicsCommandList* = guid(0x5B160D0F'u32, 0xAC1B'u16, 0x4185'u16,
      [0x8B'u8, 0xA8, 0xB3, 0xAE, 0x42, 0xA5, 0xA4, 0x55])
  IID_ID3D12CommandQueue* = guid(0x0EC870A6'u32, 0x5D7E'u16, 0x4C22'u16,
      [0x8C'u8, 0xFC, 0x5B, 0xAA, 0xE0, 0x76, 0x16, 0xED])
  IID_ID3D12Fence* = guid(0x0A753DCF'u32, 0xC4D8'u16, 0x4B91'u16,
      [0xAD'u8, 0xF6, 0xBE, 0x5A, 0x60, 0xD9, 0x5A, 0x76])
  IID_ID3D12PipelineState* = guid(0x765A30F3'u32, 0xF624'u16, 0x4C6F'u16,
      [0xA8'u8, 0x28, 0xAC, 0xE9, 0x48, 0x62, 0x24, 0x45])

# ═══════════════════════════════════════════════════════════════════════
# Constants (values cross-checked against mini_d3d12.h)
# ═══════════════════════════════════════════════════════════════════════

const
  D3D12_COMMAND_LIST_TYPE_DIRECT* = 0'u32
  D3D12_COMMAND_QUEUE_FLAG_NONE* = 0'u32

  D3D12_HEAP_TYPE_DEFAULT* = 1'u32
  D3D12_HEAP_TYPE_UPLOAD* = 2'u32
  D3D12_HEAP_TYPE_READBACK* = 3'u32
  D3D12_HEAP_TYPE_CUSTOM* = 4'u32
  D3D12_HEAP_FLAG_NONE* = 0'u32

  D3D12_RESOURCE_DIMENSION_BUFFER* = 1'u32
  D3D12_TEXTURE_LAYOUT_ROW_MAJOR* = 1'u32
  DXGI_FORMAT_UNKNOWN* = DXGI_FORMAT(0)
  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS* = 0x4'u32

  D3D12_RESOURCE_STATE_COMMON* = 0'u32
  D3D12_RESOURCE_STATE_UNORDERED_ACCESS* = 0x8'u32
  D3D12_RESOURCE_STATE_COPY_DEST* = 0x400'u32
  D3D12_RESOURCE_STATE_COPY_SOURCE* = 0x800'u32
  # 0x1 | 0x2 | 0x40 | 0x80 | 0x200 | 0x800: vertex/constant + index +
  # non-pixel + pixel shader resource + indirect argument + copy source.
  D3D12_RESOURCE_STATE_GENERIC_READ* = 0xAC3'u32

  D3D12_ROOT_PARAMETER_TYPE_CBV* = 2'u32
  D3D12_ROOT_PARAMETER_TYPE_SRV* = 3'u32
  D3D12_ROOT_PARAMETER_TYPE_UAV* = 4'u32
  D3D12_SHADER_VISIBILITY_ALL* = 0'u32
  D3D12_ROOT_SIGNATURE_FLAG_NONE* = 0'u32
  D3D_ROOT_SIGNATURE_VERSION_1_0* = 0x1'u32

  D3D_FEATURE_LEVEL_11_0* = 0xB000'u32
  D3D12_FENCE_FLAG_NONE* = 0'u32

  # Cross-checked against the vendored real d3d12.h.
  # D3D12_FEATURE_SHADER_MODEL is enum member 7.
  # D3D_SHADER_MODEL_6_4 is 0x64 in d3dcommon's D3D_SHADER_MODEL.
  D3D12_FEATURE_SHADER_MODEL* = 7'u32
  D3D_SHADER_MODEL_6_4* = 0x64'u32

# ═══════════════════════════════════════════════════════════════════════
# Structures
# ═══════════════════════════════════════════════════════════════════════

type
  D3D12_COMMAND_QUEUE_DESC* = object
    `type`*: uint32        # D3D12_COMMAND_LIST_TYPE
    priority*: int32
    flags*: uint32         # D3D12_COMMAND_QUEUE_FLAGS
    nodeMask*: uint32

  D3D12_HEAP_PROPERTIES* = object
    `type`*: uint32        # D3D12_HEAP_TYPE
    cpuPageProperty*: uint32
    memoryPoolPreference*: uint32
    creationNodeMask*: uint32
    visibleNodeMask*: uint32

  D3D12_SAMPLE_DESC* = object
    count*: uint32
    quality*: uint32

  D3D12_RESOURCE_DESC* = object
    dimension*: uint32     # D3D12_RESOURCE_DIMENSION
    alignment*: uint64
    width*: uint64
    height*: uint32
    depthOrArraySize*: uint16
    mipLevels*: uint16
    format*: DXGI_FORMAT
    sampleDesc*: D3D12_SAMPLE_DESC
    layout*: uint32        # D3D12_TEXTURE_LAYOUT
    flags*: uint32

  D3D12_SHADER_BYTECODE* = object
    pShaderBytecode*: pointer
    bytecodeLength*: SIZE_T

  D3D12_CACHED_PIPELINE_STATE* = object
    pCachedBlob*: pointer
    cachedBlobSizeInBytes*: SIZE_T

  D3D12_COMPUTE_PIPELINE_STATE_DESC* = object
    pRootSignature*: pointer
    cs*: D3D12_SHADER_BYTECODE
    nodeMask*: uint32
    cachedPSO*: D3D12_CACHED_PIPELINE_STATE
    flags*: uint32

  D3D12_RANGE* = object
    `begin`*: SIZE_T
    `end`*: SIZE_T

  # In/out struct of the checkFeatureSupport D3D12_FEATURE_SHADER_MODEL query:
  # set the requested model, d3d12.dll writes back the highest model the driver accepts.
  D3D12_FEATURE_DATA_SHADER_MODEL* = object
    highestShaderModel*: uint32

  D3D12_ROOT_DESCRIPTOR* = object
    shaderRegister*: uint32
    registerSpace*: uint32

  D3D12_ROOT_PARAMETER_UNION* = object
    descriptor*: D3D12_ROOT_DESCRIPTOR
    slack: array[8, uint8]

  D3D12_ROOT_PARAMETER* = object
    parameterType*: uint32
    union* {.align: 8.}: D3D12_ROOT_PARAMETER_UNION
    shaderVisibility*: uint32

  D3D12_ROOT_SIGNATURE_DESC* = object
    numParameters*: uint32
    pParameters*: ptr D3D12_ROOT_PARAMETER
    numStaticSamplers*: uint32
    pStaticSamplers*: pointer
    flags*: uint32                     # D3D12_ROOT_SIGNATURE_FLAGS

  D3D12_RESOURCE_ALLOCATION_INFO* = object
    sizeInBytes*: uint64
    alignment*: uint64

# ═══════════════════════════════════════════════════════════════════════
# COM interfaces and vtables
# ═══════════════════════════════════════════════════════════════════════
# Each vtable lists the full flat COM slot sequence:
# - IUnknown at slots 0-2
# - the inherited slots next, in declaration order
# - the interface's own methods last

type
  ID3D12Device* = ptr object
    lpVtbl*: ptr ID3D12DeviceVtbl
  ID3D12DeviceVtbl* = object
    # slot 0
    queryInterface*: proc (self: ID3D12Device, riid: ptr GUID, ppvObject: ptr pointer): HRESULT {.stdcall.}
    # slot 1
    addRef*: proc (self: ID3D12Device): uint32 {.stdcall.}
    # slot 2
    release*: proc (self: ID3D12Device): uint32 {.stdcall.}
    # slot 3
    getPrivateData*: proc (self: ID3D12Device, refguid: pointer, pDataSize: ptr uint32, pData: pointer): HRESULT {.stdcall.}
    # slot 4
    setPrivateData*: proc (self: ID3D12Device, refguid: pointer, dataSize: uint32, data: pointer): HRESULT {.stdcall.}
    # slot 5
    setPrivateDataInterface*: proc (self: ID3D12Device, refguid: pointer, pData: pointer): HRESULT {.stdcall.}
    # slot 6
    setName*: proc (self: ID3D12Device, name: LPCWSTR): HRESULT {.stdcall.}
    # slot 7
    getNodeCount*: proc (self: ID3D12Device): uint32 {.stdcall.}
    # slot 8
    createCommandQueue*: proc (self: ID3D12Device, pDesc: ptr D3D12_COMMAND_QUEUE_DESC, riid: ptr GUID, ppCommandQueue: ptr pointer): HRESULT {.stdcall.}
    # slot 9
    createCommandAllocator*: proc (self: ID3D12Device, `type`: uint32, riid: ptr GUID, ppCommandAllocator: ptr pointer): HRESULT {.stdcall.}
    # slot 10
    createGraphicsPipelineState*: proc (self: ID3D12Device, pDesc: pointer, riid: ptr GUID, ppPipelineState: ptr pointer): HRESULT {.stdcall.}
    # slot 11
    createComputePipelineState*: proc (self: ID3D12Device, pDesc: ptr D3D12_COMPUTE_PIPELINE_STATE_DESC, riid: ptr GUID, ppPipelineState: ptr pointer): HRESULT {.stdcall.}
    # slot 12
    createCommandList*: proc (self: ID3D12Device, nodeMask: uint32, `type`: uint32, pCommandAllocator: ID3D12CommandAllocator, pInitialState: ID3D12PipelineState, riid: ptr GUID, ppCommandList: ptr pointer): HRESULT {.stdcall.}
    # slot 13
    checkFeatureSupport*: proc (self: ID3D12Device, feature: uint32, pFeatureSupportData: pointer, featureSupportDataSize: uint32): HRESULT {.stdcall.}
    # slot 14
    createDescriptorHeap*: proc (self: ID3D12Device, pDesc: pointer, riid: ptr GUID, ppHeap: ptr pointer): HRESULT {.stdcall.}
    # slot 15
    getDescriptorHandleIncrementSize*: proc (self: ID3D12Device, `type`: uint32): uint32 {.stdcall.}
    # slot 16
    createRootSignature*: proc (self: ID3D12Device, nodeMask: uint32, pBlobWithRootSignature: pointer, blobLengthInBytes: SIZE_T, riid: ptr GUID, ppvRootSignature: ptr pointer): HRESULT {.stdcall.}
    # slot 17
    createConstantBufferView*: proc (self: ID3D12Device, pDesc: pointer, destDescriptor: uint64): void {.stdcall.}
    # slot 18
    createShaderResourceView*: proc (self: ID3D12Device, pResource: pointer, pDesc: pointer, destDescriptor: uint64): void {.stdcall.}
    # slot 19
    createUnorderedAccessView*: proc (self: ID3D12Device, pResource: pointer, pCounterResource: pointer, pDesc: pointer, destDescriptor: uint64): void {.stdcall.}
    # slot 20
    createRenderTargetView*: proc (self: ID3D12Device, pResource: pointer, pDesc: pointer, destDescriptor: uint64): void {.stdcall.}
    # slot 21
    createDepthStencilView*: proc (self: ID3D12Device, pResource: pointer, pDesc: pointer, destDescriptor: uint64): void {.stdcall.}
    # slot 22
    createSampler*: proc (self: ID3D12Device, pDesc: pointer, destDescriptor: uint64): void {.stdcall.}
    # slot 23
    copyDescriptors*: proc (self: ID3D12Device, numDestDescriptorRanges: uint32, pDestDescriptorRangeStarts: pointer, pDestDescriptorRangeSizes: pointer, numSrcDescriptorRanges: uint32, pSrcDescriptorRangeStarts: pointer, pSrcDescriptorRangeSizes: pointer, descriptorHeapsType: uint32): void {.stdcall.}
    # slot 24
    copyDescriptorsSimple*: proc (self: ID3D12Device, numDescriptors: uint32, destDescriptorRangeStart: uint64, srcDescriptorRangeStart: uint64, descriptorHeapsType: uint32): void {.stdcall.}
    # slot 25
    getResourceAllocationInfo*: proc (self: ID3D12Device, visibleMask: uint32, numResourceDescs: uint32, pResourceDescs: ptr D3D12_RESOURCE_DESC): D3D12_RESOURCE_ALLOCATION_INFO {.stdcall.}
    # slot 26
    getCustomHeapProperties*: proc (self: ID3D12Device, nodeMask: uint32, heapType: uint32): D3D12_HEAP_PROPERTIES {.stdcall.}
    # slot 27
    createCommittedResource*: proc (self: ID3D12Device, pHeapProperties: ptr D3D12_HEAP_PROPERTIES, heapFlags: uint32, pDesc: ptr D3D12_RESOURCE_DESC, initialResourceState: uint32, pOptimizedClearValue: pointer, riidResource: ptr GUID, ppvResource: ptr pointer): HRESULT {.stdcall.}
    # slot 28
    createHeap*: proc (self: ID3D12Device, pDesc: pointer, riid: ptr GUID, ppHeap: ptr pointer): HRESULT {.stdcall.}
    # slot 29
    createPlacedResource*: proc (self: ID3D12Device, pHeap: pointer, heapOffset: uint64, pDesc: ptr D3D12_RESOURCE_DESC, initialResourceState: uint32, pOptimizedClearValue: pointer, riidResource: ptr GUID, ppvResource: ptr pointer): HRESULT {.stdcall.}
    # slot 30
    createReservedResource*: proc (self: ID3D12Device, pDesc: ptr D3D12_RESOURCE_DESC, initialResourceState: uint32, pOptimizedClearValue: pointer, riidResource: ptr GUID, ppvResource: ptr pointer): HRESULT {.stdcall.}
    # slot 31
    createSharedHandle*: proc (self: ID3D12Device, pObject: pointer, access: uint32, name: LPCWSTR, pHandle: ptr HANDLE): HRESULT {.stdcall.}
    # slot 32
    openSharedHandle*: proc (self: ID3D12Device, ntHandle: HANDLE, riid: ptr GUID, ppvObj: ptr pointer): HRESULT {.stdcall.}
    # slot 33
    openSharedHandleByName*: proc (self: ID3D12Device, name: LPCWSTR, access: uint32, pHandle: ptr HANDLE): HRESULT {.stdcall.}
    # slot 34
    makeResident*: proc (self: ID3D12Device, numObjects: uint32, ppObjects: ptr pointer): HRESULT {.stdcall.}
    # slot 35
    evict*: proc (self: ID3D12Device, numObjects: uint32, ppObjects: ptr pointer): HRESULT {.stdcall.}
    # slot 36
    createFence*: proc (self: ID3D12Device, initialValue: uint64, flags: uint32, riid: ptr GUID, ppFence: ptr pointer): HRESULT {.stdcall.}
    # slot 37
    getDeviceRemovedReason*: proc (self: ID3D12Device): HRESULT {.stdcall.}

  ID3D12CommandQueue* = ptr object
    lpVtbl*: ptr ID3D12CommandQueueVtbl
  ID3D12CommandQueueVtbl* = object
    # slot 0
    queryInterface*: proc (self: ID3D12CommandQueue, riid: ptr GUID, ppvObject: ptr pointer): HRESULT {.stdcall.}
    # slot 1
    addRef*: proc (self: ID3D12CommandQueue): uint32 {.stdcall.}
    # slot 2
    release*: proc (self: ID3D12CommandQueue): uint32 {.stdcall.}
    # slot 3
    getPrivateData*: proc (self: ID3D12CommandQueue, refguid: pointer, pDataSize: ptr uint32, pData: pointer): HRESULT {.stdcall.}
    # slot 4
    setPrivateData*: proc (self: ID3D12CommandQueue, refguid: pointer, dataSize: uint32, data: pointer): HRESULT {.stdcall.}
    # slot 5
    setPrivateDataInterface*: proc (self: ID3D12CommandQueue, refguid: pointer, pData: pointer): HRESULT {.stdcall.}
    # slot 6
    setName*: proc (self: ID3D12CommandQueue, name: LPCWSTR): HRESULT {.stdcall.}
    # slot 7
    getDevice*: proc (self: ID3D12CommandQueue, riid: ptr GUID, ppvDevice: ptr pointer): HRESULT {.stdcall.}
    # slot 8
    updateTileMappings*: proc (self: ID3D12CommandQueue, pResource: pointer, numResourceRegions: uint32, pResourceRegionStartCoordinates: pointer, pResourceRegionSizes: pointer, pHeap: pointer, numRanges: uint32, pRangeFlags: pointer, pHeapRangeStartOffsets: pointer, pRangeTileCounts: pointer, tileMappingFlags: uint32): void {.stdcall.}
    # slot 9
    copyTileMappings*: proc (self: ID3D12CommandQueue, pDstResource: pointer, pDstRegionStartCoordinate: pointer, pSrcResource: pointer, pSrcRegionStartCoordinate: pointer, pRegionSize: pointer, tileMappingFlags: uint32): void {.stdcall.}
    # slot 10
    executeCommandLists*: proc (self: ID3D12CommandQueue, numCommandLists: uint32, ppCommandLists: ptr ID3D12CommandList): void {.stdcall.}
    # slot 11
    setMarker*: proc (self: ID3D12CommandQueue, metadata: uint32, pData: pointer, size: uint32): void {.stdcall.}
    # slot 12
    beginEvent*: proc (self: ID3D12CommandQueue, metadata: uint32, pData: pointer, size: uint32): void {.stdcall.}
    # slot 13
    endEvent*: proc (self: ID3D12CommandQueue): void {.stdcall.}
    # slot 14
    signal*: proc (self: ID3D12CommandQueue, pFence: ID3D12Fence, value: uint64): HRESULT {.stdcall.}

  ID3D12CommandAllocator* = ptr object
    lpVtbl*: ptr ID3D12CommandAllocatorVtbl
  ID3D12CommandAllocatorVtbl* = object
    # slot 0
    queryInterface*: proc (self: ID3D12CommandAllocator, riid: ptr GUID, ppvObject: ptr pointer): HRESULT {.stdcall.}
    # slot 1
    addRef*: proc (self: ID3D12CommandAllocator): uint32 {.stdcall.}
    # slot 2
    release*: proc (self: ID3D12CommandAllocator): uint32 {.stdcall.}
    # slot 3
    getPrivateData*: proc (self: ID3D12CommandAllocator, refguid: pointer, pDataSize: ptr uint32, pData: pointer): HRESULT {.stdcall.}
    # slot 4
    setPrivateData*: proc (self: ID3D12CommandAllocator, refguid: pointer, dataSize: uint32, data: pointer): HRESULT {.stdcall.}
    # slot 5
    setPrivateDataInterface*: proc (self: ID3D12CommandAllocator, refguid: pointer, pData: pointer): HRESULT {.stdcall.}
    # slot 6
    setName*: proc (self: ID3D12CommandAllocator, name: LPCWSTR): HRESULT {.stdcall.}
    # slot 7
    getDevice*: proc (self: ID3D12CommandAllocator, riid: ptr GUID, ppvDevice: ptr pointer): HRESULT {.stdcall.}
    # slot 8
    reset*: proc (self: ID3D12CommandAllocator): HRESULT {.stdcall.}

  # Opaque handles, the smoke test passes them to the runtime without
  # dispatching through their vtables.
  ID3D12CommandList* = ptr object
    lpVtbl*: pointer

  ID3D12GraphicsCommandList* = ptr object
    lpVtbl*: ptr ID3D12GraphicsCommandListVtbl

  ID3D12GraphicsCommandListVtbl* = object
    # slot 0
    queryInterface*: proc (self: ID3D12GraphicsCommandList, riid: ptr GUID, ppvObject: ptr pointer): HRESULT {.stdcall.}
    # slot 1
    addRef*: proc (self: ID3D12GraphicsCommandList): uint32 {.stdcall.}
    # slot 2
    release*: proc (self: ID3D12GraphicsCommandList): uint32 {.stdcall.}
    # slot 3
    getPrivateData*: proc (self: ID3D12GraphicsCommandList, refguid: pointer, pDataSize: ptr uint32, pData: pointer): HRESULT {.stdcall.}
    # slot 4
    setPrivateData*: proc (self: ID3D12GraphicsCommandList, refguid: pointer, dataSize: uint32, data: pointer): HRESULT {.stdcall.}
    # slot 5
    setPrivateDataInterface*: proc (self: ID3D12GraphicsCommandList, refguid: pointer, pData: pointer): HRESULT {.stdcall.}
    # slot 6
    setName*: proc (self: ID3D12GraphicsCommandList, name: LPCWSTR): HRESULT {.stdcall.}
    # slot 7
    getDevice*: proc (self: ID3D12GraphicsCommandList, riid: ptr GUID, ppvDevice: ptr pointer): HRESULT {.stdcall.}
    # slot 8
    getType*: proc (self: ID3D12GraphicsCommandList): uint32 {.stdcall.}
    # slot 9
    close*: proc (self: ID3D12GraphicsCommandList): HRESULT {.stdcall.}
    # slot 10
    reset*: proc (self: ID3D12GraphicsCommandList, pAllocator: ID3D12CommandAllocator, pInitialState: ID3D12PipelineState): HRESULT {.stdcall.}
    # slot 11
    clearState*: proc (self: ID3D12GraphicsCommandList, pipelineState: ID3D12PipelineState): void {.stdcall.}
    # slot 12
    drawInstanced*: proc (self: ID3D12GraphicsCommandList, vertexCountPerInstance: uint32, instanceCount: uint32, startVertexLocation: uint32, startInstanceLocation: uint32): void {.stdcall.}
    # slot 13
    drawIndexedInstanced*: proc (self: ID3D12GraphicsCommandList, indexCountPerInstance: uint32, instanceCount: uint32, startIndexLocation: uint32, baseVertexLocation: int32, startInstanceLocation: uint32): void {.stdcall.}
    # slot 14
    dispatch*: proc (self: ID3D12GraphicsCommandList, threadGroupCountX: uint32, threadGroupCountY: uint32, threadGroupCountZ: uint32): void {.stdcall.}
    # slot 15
    copyBufferRegion*: proc (self: ID3D12GraphicsCommandList, pDstBuffer: ID3D12Resource, dstOffset: uint64, pSrcBuffer: ID3D12Resource, srcOffset: uint64, numBytes: uint64): void {.stdcall.}
    # slot 16
    copyTextureRegion*: proc (self: ID3D12GraphicsCommandList, pDst: pointer, dstSubresource: uint32, pDstOffset: pointer, pSrc: pointer, srcSubresource: uint32, pSrcOffset: pointer): void {.stdcall.}
    # slot 17
    copyResource*: proc (self: ID3D12GraphicsCommandList, pDstResource: ID3D12Resource, pSrcResource: ID3D12Resource): void {.stdcall.}
    # slot 18
    copyTiles*: proc (self: ID3D12GraphicsCommandList, pTiledResource: pointer, pTileRegionStartCoordinate: pointer, pTileRegionSize: pointer, pBuffer: pointer, bufferOffset: uint64, copyFlags: uint32): void {.stdcall.}
    # slot 19
    resolveSubresource*: proc (self: ID3D12GraphicsCommandList, pDstResource: pointer, dstSubresource: uint32, pSrcResource: pointer, srcSubresource: uint32, format: DXGI_FORMAT): void {.stdcall.}
    # slot 20
    iaSetPrimitiveTopology*: proc (self: ID3D12GraphicsCommandList, primitiveTopology: uint32): void {.stdcall.}
    # slot 21
    rsSetViewports*: proc (self: ID3D12GraphicsCommandList, numViewports: uint32, pViewports: pointer): void {.stdcall.}
    # slot 22
    rsSetScissorRects*: proc (self: ID3D12GraphicsCommandList, numRects: uint32, pRects: pointer): void {.stdcall.}
    # slot 23
    omSetBlendFactor*: proc (self: ID3D12GraphicsCommandList, blendFactor: pointer): void {.stdcall.}
    # slot 24
    omSetStencilRef*: proc (self: ID3D12GraphicsCommandList, stencilRef: uint32): void {.stdcall.}
    # slot 25
    setPipelineState*: proc (self: ID3D12GraphicsCommandList, pipelineState: ID3D12PipelineState): void {.stdcall.}
    # slot 26
    resourceBarrier*: proc (self: ID3D12GraphicsCommandList, numBarriers: uint32, pBarriers: pointer): void {.stdcall.}
    # slot 27
    executeBundle*: proc (self: ID3D12GraphicsCommandList, pCommandList: ID3D12CommandList): void {.stdcall.}
    # slot 28
    setDescriptorHeaps*: proc (self: ID3D12GraphicsCommandList, numDescriptorHeaps: uint32, ppDescriptorHeaps: pointer): void {.stdcall.}
    # slot 29
    setComputeRootSignature*: proc (self: ID3D12GraphicsCommandList, pRootSignature: ID3D12RootSignature): void {.stdcall.}
    # slot 30
    setGraphicsRootSignature*: proc (self: ID3D12GraphicsCommandList, pRootSignature: ID3D12RootSignature): void {.stdcall.}
    # slot 31
    setComputeRootDescriptorTable*: proc (self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, baseDescriptor: uint64): void {.stdcall.}
    # slot 32
    setGraphicsRootDescriptorTable*: proc (self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, baseDescriptor: uint64): void {.stdcall.}
    # slot 33
    setComputeRoot32BitConstant*: proc (self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, src32BitValue: uint32, destOffsetIn32BitValues: uint32): void {.stdcall.}
    # slot 34
    setGraphicsRoot32BitConstant*: proc (self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, src32BitValue: uint32, destOffsetIn32BitValues: uint32): void {.stdcall.}
    # slot 35
    setComputeRoot32BitConstants*: proc (self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, num32BitValuesToSet: uint32, pSrcData: pointer, destOffsetIn32BitValues: uint32): void {.stdcall.}
    # slot 36
    setGraphicsRoot32BitConstants*: proc (self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, num32BitValuesToSet: uint32, pSrcData: pointer, destOffsetIn32BitValues: uint32): void {.stdcall.}
    # slot 37
    setComputeRootConstantBufferView*: proc (self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, bufferLocation: D3D12_GPU_VIRTUAL_ADDRESS): void {.stdcall.}
    # slot 38
    setGraphicsRootConstantBufferView*: proc (self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, bufferLocation: D3D12_GPU_VIRTUAL_ADDRESS): void {.stdcall.}
    # slot 39
    setComputeRootShaderResourceView*: proc (self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, bufferLocation: D3D12_GPU_VIRTUAL_ADDRESS): void {.stdcall.}
    # slot 40
    setGraphicsRootShaderResourceView*: proc (self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, bufferLocation: D3D12_GPU_VIRTUAL_ADDRESS): void {.stdcall.}
    # slot 41
    setComputeRootUnorderedAccessView*: proc (self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, bufferLocation: D3D12_GPU_VIRTUAL_ADDRESS): void {.stdcall.}

  ID3D12Resource* = ptr object
    lpVtbl*: ptr ID3D12ResourceVtbl
  ID3D12ResourceVtbl* = object
    # slot 0
    queryInterface*: proc (self: ID3D12Resource, riid: ptr GUID, ppvObject: ptr pointer): HRESULT {.stdcall.}
    # slot 1
    addRef*: proc (self: ID3D12Resource): uint32 {.stdcall.}
    # slot 2
    release*: proc (self: ID3D12Resource): uint32 {.stdcall.}
    # slot 3
    getPrivateData*: proc (self: ID3D12Resource, refguid: pointer, pDataSize: ptr uint32, pData: pointer): HRESULT {.stdcall.}
    # slot 4
    setPrivateData*: proc (self: ID3D12Resource, refguid: pointer, dataSize: uint32, data: pointer): HRESULT {.stdcall.}
    # slot 5
    setPrivateDataInterface*: proc (self: ID3D12Resource, refguid: pointer, pData: pointer): HRESULT {.stdcall.}
    # slot 6
    setName*: proc (self: ID3D12Resource, name: LPCWSTR): HRESULT {.stdcall.}
    # slot 7
    getDevice*: proc (self: ID3D12Resource, riid: ptr GUID, ppvDevice: ptr pointer): HRESULT {.stdcall.}
    # slot 8
    map*: proc (self: ID3D12Resource, subresource: uint32, pReadRange: ptr D3D12_RANGE, ppData: ptr pointer): HRESULT {.stdcall.}
    # slot 9
    unmap*: proc (self: ID3D12Resource, subresource: uint32, pWrittenRange: ptr D3D12_RANGE): void {.stdcall.}
    # slot 10
    getDesc*: proc (self: ID3D12Resource): D3D12_RESOURCE_DESC {.stdcall.}
    # slot 11
    getGPUVirtualAddress*: proc (self: ID3D12Resource): D3D12_GPU_VIRTUAL_ADDRESS {.stdcall.}

  ID3D12Fence* = ptr object
    lpVtbl*: ptr ID3D12FenceVtbl
  ID3D12FenceVtbl* = object
    # slot 0
    queryInterface*: proc (self: ID3D12Fence, riid: ptr GUID, ppvObject: ptr pointer): HRESULT {.stdcall.}
    # slot 1
    addRef*: proc (self: ID3D12Fence): uint32 {.stdcall.}
    # slot 2
    release*: proc (self: ID3D12Fence): uint32 {.stdcall.}
    # slot 3
    getPrivateData*: proc (self: ID3D12Fence, refguid: pointer, pDataSize: ptr uint32, pData: pointer): HRESULT {.stdcall.}
    # slot 4
    setPrivateData*: proc (self: ID3D12Fence, refguid: pointer, dataSize: uint32, data: pointer): HRESULT {.stdcall.}
    # slot 5
    setPrivateDataInterface*: proc (self: ID3D12Fence, refguid: pointer, pData: pointer): HRESULT {.stdcall.}
    # slot 6
    setName*: proc (self: ID3D12Fence, name: LPCWSTR): HRESULT {.stdcall.}
    # slot 7
    getDevice*: proc (self: ID3D12Fence, riid: ptr GUID, ppvDevice: ptr pointer): HRESULT {.stdcall.}
    # slot 8
    getCompletedValue*: proc (self: ID3D12Fence): uint64 {.stdcall.}

  ID3D12RootSignature* = ptr object
    lpVtbl*: pointer

  ID3D12PipelineState* = ptr object
    lpVtbl*: pointer

  ID3DBlob* = ptr object
    lpVtbl*: ptr ID3DBlobVtbl
  ID3DBlobVtbl* = object
    # slot 0
    queryInterface*: proc (self: ID3DBlob, riid: ptr GUID, ppvObject: ptr pointer): HRESULT {.stdcall.}
    # slot 1
    addRef*: proc (self: ID3DBlob): uint32 {.stdcall.}
    # slot 2
    release*: proc (self: ID3DBlob): uint32 {.stdcall.}
    # slot 3
    getBufferPointer*: proc (self: ID3DBlob): pointer {.stdcall.}
    # slot 4
    getBufferSize*: proc (self: ID3DBlob): SIZE_T {.stdcall.}

# ═══════════════════════════════════════════════════════════════════════
# Method wrappers (vtable dispatch by named slot field)
# ═══════════════════════════════════════════════════════════════════════

proc createCommandQueue*(self: ID3D12Device, pDesc: ptr D3D12_COMMAND_QUEUE_DESC, riid: ptr GUID, ppCommandQueue: ptr pointer): HRESULT {.inline.} =
  self.lpVtbl.createCommandQueue(self, pDesc, riid, ppCommandQueue)

proc createCommandAllocator*(self: ID3D12Device, `type`: uint32, riid: ptr GUID, ppCommandAllocator: ptr pointer): HRESULT {.inline.} =
  self.lpVtbl.createCommandAllocator(self, `type`, riid, ppCommandAllocator)

proc createComputePipelineState*(self: ID3D12Device, pDesc: ptr D3D12_COMPUTE_PIPELINE_STATE_DESC, riid: ptr GUID, ppPipelineState: ptr pointer): HRESULT {.inline.} =
  self.lpVtbl.createComputePipelineState(self, pDesc, riid, ppPipelineState)

proc createCommandList*(self: ID3D12Device, nodeMask: uint32, `type`: uint32, pCommandAllocator: ID3D12CommandAllocator, pInitialState: ID3D12PipelineState, riid: ptr GUID, ppCommandList: ptr pointer): HRESULT {.inline.} =
  self.lpVtbl.createCommandList(self, nodeMask, `type`, pCommandAllocator, pInitialState, riid, ppCommandList)

proc createRootSignature*(self: ID3D12Device, nodeMask: uint32, pBlobWithRootSignature: pointer, blobLengthInBytes: SIZE_T, riid: ptr GUID, ppvRootSignature: ptr pointer): HRESULT {.inline.} =
  self.lpVtbl.createRootSignature(self, nodeMask, pBlobWithRootSignature, blobLengthInBytes, riid, ppvRootSignature)

proc createCommittedResource*(self: ID3D12Device, pHeapProperties: ptr D3D12_HEAP_PROPERTIES, heapFlags: uint32, pDesc: ptr D3D12_RESOURCE_DESC, initialResourceState: uint32, pOptimizedClearValue: pointer, riidResource: ptr GUID, ppvResource: ptr pointer): HRESULT {.inline.} =
  self.lpVtbl.createCommittedResource(self, pHeapProperties, heapFlags, pDesc, initialResourceState, pOptimizedClearValue, riidResource, ppvResource)

proc createFence*(self: ID3D12Device, initialValue: uint64, flags: uint32, riid: ptr GUID, ppFence: ptr pointer): HRESULT {.inline.} =
  self.lpVtbl.createFence(self, initialValue, flags, riid, ppFence)

proc getDeviceRemovedReason*(self: ID3D12Device): HRESULT {.inline.} =
  self.lpVtbl.getDeviceRemovedReason(self)

proc checkFeatureSupport*(self: ID3D12Device, feature: uint32, pFeatureSupportData: pointer, featureSupportDataSize: uint32): HRESULT {.inline.} =
  self.lpVtbl.checkFeatureSupport(self, feature, pFeatureSupportData, featureSupportDataSize)

proc executeCommandLists*(self: ID3D12CommandQueue, numCommandLists: uint32, ppCommandLists: ptr ID3D12CommandList): void {.inline.} =
  self.lpVtbl.executeCommandLists(self, numCommandLists, ppCommandLists)

proc signal*(self: ID3D12CommandQueue, pFence: ID3D12Fence, value: uint64): HRESULT {.inline.} =
  self.lpVtbl.signal(self, pFence, value)

proc close*(self: ID3D12GraphicsCommandList): HRESULT {.inline.} =
  self.lpVtbl.close(self)

proc reset*(self: ID3D12GraphicsCommandList, pAllocator: ID3D12CommandAllocator, pInitialState: ID3D12PipelineState): HRESULT {.inline.} =
  self.lpVtbl.reset(self, pAllocator, pInitialState)

proc dispatch*(self: ID3D12GraphicsCommandList, threadGroupCountX: uint32, threadGroupCountY: uint32, threadGroupCountZ: uint32): void {.inline.} =
  self.lpVtbl.dispatch(self, threadGroupCountX, threadGroupCountY, threadGroupCountZ)

proc copyBufferRegion*(self: ID3D12GraphicsCommandList, pDstBuffer: ID3D12Resource, dstOffset: uint64, pSrcBuffer: ID3D12Resource, srcOffset: uint64, numBytes: uint64): void {.inline.} =
  self.lpVtbl.copyBufferRegion(self, pDstBuffer, dstOffset, pSrcBuffer, srcOffset, numBytes)

proc setPipelineState*(self: ID3D12GraphicsCommandList, pipelineState: ID3D12PipelineState): void {.inline.} =
  self.lpVtbl.setPipelineState(self, pipelineState)

proc setComputeRootSignature*(self: ID3D12GraphicsCommandList, pRootSignature: ID3D12RootSignature): void {.inline.} =
  self.lpVtbl.setComputeRootSignature(self, pRootSignature)

proc setComputeRootConstantBufferView*(self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, bufferLocation: D3D12_GPU_VIRTUAL_ADDRESS): void {.inline.} =
  self.lpVtbl.setComputeRootConstantBufferView(self, rootParameterIndex, bufferLocation)

proc setComputeRootShaderResourceView*(self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, bufferLocation: D3D12_GPU_VIRTUAL_ADDRESS): void {.inline.} =
  self.lpVtbl.setComputeRootShaderResourceView(self, rootParameterIndex, bufferLocation)

proc setComputeRootUnorderedAccessView*(self: ID3D12GraphicsCommandList, rootParameterIndex: uint32, bufferLocation: D3D12_GPU_VIRTUAL_ADDRESS): void {.inline.} =
  self.lpVtbl.setComputeRootUnorderedAccessView(self, rootParameterIndex, bufferLocation)

proc map*(self: ID3D12Resource, subresource: uint32, pReadRange: ptr D3D12_RANGE, ppData: ptr pointer): HRESULT {.inline.} =
  self.lpVtbl.map(self, subresource, pReadRange, ppData)

proc unmap*(self: ID3D12Resource, subresource: uint32, pWrittenRange: ptr D3D12_RANGE): void {.inline.} =
  self.lpVtbl.unmap(self, subresource, pWrittenRange)

proc getGPUVirtualAddress*(self: ID3D12Resource): D3D12_GPU_VIRTUAL_ADDRESS {.inline.} =
  self.lpVtbl.getGPUVirtualAddress(self)

proc getCompletedValue*(self: ID3D12Fence): uint64 {.inline.} =
  self.lpVtbl.getCompletedValue(self)

proc getBufferPointer*(self: ID3DBlob): pointer {.inline.} =
  self.lpVtbl.getBufferPointer(self)

proc getBufferSize*(self: ID3DBlob): SIZE_T {.inline.} =
  self.lpVtbl.getBufferSize(self)

# ═══════════════════════════════════════════════════════════════════════
# Library loading
# ═══════════════════════════════════════════════════════════════════════

const D3D12Lib* = "d3d12.dll"

type
  D3D12Api* = object
    lib*: LibHandle
    createDevice*: proc (pAdapter: pointer, featureLevel: D3D_FEATURE_LEVEL, riid: ptr GUID, ppDevice: ptr pointer): HRESULT {.stdcall.}
    serializeRootSignature*: proc (desc: ptr D3D12_ROOT_SIGNATURE_DESC, version: uint32, blob: ptr ID3DBlob, errorBlob: ptr ID3DBlob): HRESULT {.stdcall.}

proc loadD3D12*(): D3D12Api =
  result.lib = loadLib(D3D12Lib)
  if result.lib == nil:
    failLoud("Cannot load " & D3D12Lib)
  result.createDevice = cast[typeof(result.createDevice)](result.lib.symAddr("D3D12CreateDevice"))
  if result.createDevice == nil:
    failLoud(D3D12Lib & " missing D3D12CreateDevice")
  result.serializeRootSignature = cast[typeof(result.serializeRootSignature)](result.lib.symAddr("D3D12SerializeRootSignature"))
  if result.serializeRootSignature == nil:
    failLoud(D3D12Lib & " missing D3D12SerializeRootSignature")
