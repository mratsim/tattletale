## D3D12 axpy smoke: full device dispatch and readback on Windows,
## DXBC/DXIL fixture checks elsewhere.
##
## Test that the baked axpy DXBC container drives a complete device →
## dispatch → readback on Windows with exact element-wise results.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/d3d12/app/t_d3d12_smoke.nim

import
  workspace/crucible,
  workspace/crucible/src/runtime/errors,
  workspace/crucible/src/runtime/backends/d3d12_compile,
  workspace/crucible/src/abis/d3d12_abi,
  workspace/crucible/tests/codegen/d3d12/app/d3d12_axpy_fixture,
  std/[dynlib, os, strutils]

# DXBC container layout, little-endian throughout:
#
# | bytes     | content                         |
# |-----------|---------------------------------|
# | 0-3       | magic "DXBC"                    |
# | 4-19      | checksum                        |
# | 20-23     | container version               |
# | 24-27     | total container size in bytes   |
# | 28-31     | part count                      |
# | 32 onward | one uint32 byte offset per part |
#
# Each part opens on its fourcc, its size in bytes comes next.

# Compile-time drift checks: the sizeof lines lock the x64 C struct layouts
# from the real d3d12.h, the numthreads line ties ThreadCount to the fixture
# kernel's entry point.
static:
  doAssert sizeof(D3D12_ROOT_PARAMETER) == 32
  doAssert sizeof(D3D12_ROOT_SIGNATURE_DESC) == 40
  doAssert sizeof(D3D12_COMPUTE_PIPELINE_STATE_DESC) == 56
  doAssert sizeof(D3D12_RESOURCE_DESC) == 56

const ThreadCount = 256
static: doAssert "numthreads(" & $ThreadCount & ", 1, 1)" in axpyKernelHlsl

proc fourccToString(word: uint32): string =
  result = newString(4)
  result[0] = chr(word and 0xFF'u32)
  result[1] = chr((word shr 8) and 0xFF'u32)
  result[2] = chr((word shr 16) and 0xFF'u32)
  result[3] = chr((word shr 24) and 0xFF'u32)

proc dxbcParts(container: openArray[uint32]): seq[tuple[fourcc: string, sizeBytes: int]] =
  let partCount = int(container[7])
  if 8 + partCount > container.len:
    failLoud("the DXBC container's part count " & $partCount & " points past the " &
        $container.len & "-word container, it is corrupt")
  result = newSeq[tuple[fourcc: string, sizeBytes: int]](partCount)
  for i in 0 ..< partCount:
    let partOffset = int(container[8 + i])
    if partOffset < 0 or partOffset div 4 + 1 >= container.len:
      failLoud("the DXBC container's part " & $i & " offset " & $partOffset &
          " points past the " & $container.len & "-word container, it is corrupt")
    result[i] = (fourccToString(container[partOffset div 4]),
        int(container[partOffset div 4 + 1]))

proc dxilPartSize(container: openArray[uint32]): int =
  result = -1
  for part in dxbcParts(container):
    if part.fourcc == "DXIL":
      result = part.sizeBytes

proc comRelease(iface: pointer) =
  # IUnknown.release is vtable slot 2, the vtable pointer is the object's first field.
  type ReleaseFn = proc (self: pointer): uint32 {.stdcall.}
  let vtbl = cast[ptr pointer](iface)[]
  let slot = cast[ptr array[3, pointer]](vtbl)[2]
  discard cast[ReleaseFn](slot)(iface)

proc checkHr(hr: HRESULT, step: string) =
  if hr != 0:
    failLoud("[d3d12] " & step & " failed: HRESULT 0x" & toHex(uint32(hr), 8))

proc waitForFence(dev: ID3D12Device, fence: ID3D12Fence, target: uint64, step: string) =
  # A fence that never signals is the visible symptom of a device removal,
  # the removed reason distinguishes a TDR from a logic bug.
  var waitedMs = 0
  while fence.getCompletedValue() < target:
    if waitedMs >= 10000:
      let removed = dev.getDeviceRemovedReason()
      failLoud("[d3d12] " & step & " timed out after 10000 ms, fence at " &
          $fence.getCompletedValue() & ", device removed reason HRESULT 0x" &
          toHex(uint32(removed), 8))
    os.sleep(1)
    inc waitedMs
  echo "[d3d12] ok: " & step

when defined(windows):
  proc runTest() =
    const
      DataElements = 8
      UploadBytes = ThreadCount * 4
      PushBytes = 512
    let alpha = 2.5'f32

    echo "[d3d12] step: load d3d12.dll (loadD3D12)"
    let api = loadD3D12()
    echo "[d3d12] ok: load d3d12.dll"

    echo "[d3d12] note: no adapter fallback, this smoke requires a D3D12 hardware adapter (WARP not attempted)"
    echo "[d3d12] step: create device (D3D12CreateDevice)"
    var dev: ID3D12Device = nil
    let hrDevice = api.createDevice(nil, D3D_FEATURE_LEVEL_11_0, IID_ID3D12Device.addr,
        cast[ptr pointer](dev.addr))
    checkHr(hrDevice, "create device (D3D12CreateDevice)")
    echo "[d3d12] ok: create device"

    echo "[d3d12] step: shader model pre-check (ID3D12Device.checkFeatureSupport)"
    # The fixture's DXIL targets SM 6.4, the pre-check echoes the driver's ceiling.
    # PSO creation would only report a bare E_INVALIDARG.
    var smData = D3D12_FEATURE_DATA_SHADER_MODEL(highestShaderModel: D3D_SHADER_MODEL_6_4)
    let hrSm = dev.checkFeatureSupport(D3D12_FEATURE_SHADER_MODEL, smData.addr,
        uint32(sizeof(smData)))
    checkHr(hrSm, "shader model pre-check (ID3D12Device.checkFeatureSupport)")
    if smData.highestShaderModel < D3D_SHADER_MODEL_6_4:
      failLoud("[d3d12] shader model pre-check (ID3D12Device.checkFeatureSupport): " &
          "the driver tops out at shader model 0x" & toHex(smData.highestShaderModel, 2) &
          ", the fixture needs SM 6.4")
    echo "[d3d12] ok: shader model pre-check, the driver accepts SM 6.4"

    echo "[d3d12] step: create command queue (ID3D12Device.createCommandQueue)"
    var queueDesc = D3D12_COMMAND_QUEUE_DESC(
      `type`: D3D12_COMMAND_LIST_TYPE_DIRECT,
      priority: 0,
      flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
      nodeMask: 0)
    var queue: ID3D12CommandQueue = nil
    let hrQueue = dev.createCommandQueue(queueDesc.addr, IID_ID3D12CommandQueue.addr,
        cast[ptr pointer](queue.addr))
    checkHr(hrQueue, "create command queue (ID3D12Device.createCommandQueue)")
    echo "[d3d12] ok: create command queue"

    echo "[d3d12] step: serialize root signature (D3D12SerializeRootSignature)"
    var params: array[4, D3D12_ROOT_PARAMETER]
    params[0] = D3D12_ROOT_PARAMETER(
      parameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
      descriptor: D3D12_ROOT_DESCRIPTOR(shaderRegister: 1, registerSpace: 0),
      shaderVisibility: D3D12_SHADER_VISIBILITY_ALL)
    params[1] = D3D12_ROOT_PARAMETER(
      parameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
      descriptor: D3D12_ROOT_DESCRIPTOR(shaderRegister: 2, registerSpace: 0),
      shaderVisibility: D3D12_SHADER_VISIBILITY_ALL)
    params[2] = D3D12_ROOT_PARAMETER(
      parameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
      descriptor: D3D12_ROOT_DESCRIPTOR(shaderRegister: 0, registerSpace: 0),
      shaderVisibility: D3D12_SHADER_VISIBILITY_ALL)
    params[3] = D3D12_ROOT_PARAMETER(
      parameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
      descriptor: D3D12_ROOT_DESCRIPTOR(shaderRegister: 3, registerSpace: 0),
      shaderVisibility: D3D12_SHADER_VISIBILITY_ALL)
    var rootSigDesc = D3D12_ROOT_SIGNATURE_DESC(
      numParameters: 4,
      pParameters: params[0].addr,
      numStaticSamplers: 0,
      pStaticSamplers: nil,
      flags: D3D12_ROOT_SIGNATURE_FLAG_NONE)
    var rootSigBlob: ID3DBlob = nil
    var errorBlob: ID3DBlob = nil
    let hrSerialize = api.serializeRootSignature(rootSigDesc.addr,
        D3D_ROOT_SIGNATURE_VERSION_1_0, rootSigBlob.addr, errorBlob.addr)
    if hrSerialize != 0:
      var detail = ""
      if errorBlob != nil and errorBlob.getBufferSize() > 0:
        let errorLen = int(errorBlob.getBufferSize())
        detail = newString(errorLen)
        copyMem(detail[0].addr, errorBlob.getBufferPointer(), errorLen)
      failLoud("[d3d12] serialize root signature (D3D12SerializeRootSignature) failed: HRESULT 0x" &
          toHex(uint32(hrSerialize), 8) & " diagnostics: " & detail)
    echo "[d3d12] ok: serialize root signature"

    echo "[d3d12] step: create root signature (ID3D12Device.createRootSignature)"
    var rootSig: ID3D12RootSignature = nil
    let hrRootSig = dev.createRootSignature(0, rootSigBlob.getBufferPointer(),
        rootSigBlob.getBufferSize(), IID_ID3D12RootSignature.addr,
        cast[ptr pointer](rootSig.addr))
    checkHr(hrRootSig, "create root signature (ID3D12Device.createRootSignature)")
    echo "[d3d12] ok: create root signature"

    echo "[d3d12] step: create compute pipeline state (ID3D12Device.createComputePipelineState)"
    var shaderWords = axpyDxbcWords
    var psoDesc = D3D12_COMPUTE_PIPELINE_STATE_DESC(
      pRootSignature: cast[pointer](rootSig),
      cs: D3D12_SHADER_BYTECODE(
        pShaderBytecode: cast[pointer](shaderWords[0].addr),
        bytecodeLength: SIZE_T(shaderWords.len * 4)),
      nodeMask: 0,
      cachedPSO: D3D12_CACHED_PIPELINE_STATE(pCachedBlob: nil, cachedBlobSizeInBytes: 0),
      flags: 0)
    var pso: ID3D12PipelineState = nil
    let hrPso = dev.createComputePipelineState(psoDesc.addr, IID_ID3D12PipelineState.addr,
        cast[ptr pointer](pso.addr))
    checkHr(hrPso, "create compute pipeline state (ID3D12Device.createComputePipelineState)")
    echo "[d3d12] ok: create compute pipeline state"

    proc createBuffer(resource: var ID3D12Resource, heapType: uint32, initialState: uint32, resourceFlags: uint32, byteLen: uint64, step: string) =
      echo "[d3d12] step: " & step & " (ID3D12Device.createCommittedResource)"
      var heapProps = D3D12_HEAP_PROPERTIES(
        `type`: heapType,
        cpuPageProperty: 0,
        memoryPoolPreference: 0,
        creationNodeMask: 0,
        visibleNodeMask: 0)
      var desc = D3D12_RESOURCE_DESC(
        dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        alignment: 0,
        width: byteLen,
        height: 0,
        depthOrArraySize: 0,
        mipLevels: 0,
        format: DXGI_FORMAT_UNKNOWN,
        sampleDesc: D3D12_SAMPLE_DESC(count: 1, quality: 0),
        layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        flags: resourceFlags)
      let hr = dev.createCommittedResource(heapProps.addr, D3D12_HEAP_FLAG_NONE, desc.addr,
          initialState, nil, IID_ID3D12Resource.addr, cast[ptr pointer](resource.addr))
      checkHr(hr, step & " (ID3D12Device.createCommittedResource)")
      echo "[d3d12] ok: " & step

    var xUpload: ID3D12Resource = nil
    var yUpload: ID3D12Resource = nil
    var pushUpload: ID3D12Resource = nil
    var output: ID3D12Resource = nil
    var readback: ID3D12Resource = nil
    createBuffer(xUpload, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, 0,
        UploadBytes, "create x upload buffer")
    createBuffer(yUpload, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, 0,
        UploadBytes, "create y upload buffer")
    createBuffer(pushUpload, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, 0,
        PushBytes, "create push-constant upload buffer")
    createBuffer(output, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, UploadBytes, "create output buffer")
    createBuffer(readback, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, 0,
        UploadBytes, "create readback buffer")

    proc mapAndWrite(resource: ID3D12Resource, data: pointer, byteLen: int, step: string) =
      echo "[d3d12] step: " & step & " (ID3D12Resource.map)"
      var mapped: pointer = nil
      let hr = resource.map(0, nil, mapped.addr)
      checkHr(hr, step & " (ID3D12Resource.map)")
      copyMem(mapped, data, byteLen)
      resource.unmap(0, nil)
      echo "[d3d12] ok: " & step

    var xHost: array[ThreadCount, float32]
    var yHost: array[ThreadCount, float32]
    var pushHost: array[PushBytes div 4, float32]
    for i in 0 ..< DataElements:
      xHost[i] = float32(i + 1)
      yHost[i] = float32(10 * (i + 1))
    pushHost[0] = alpha
    mapAndWrite(xUpload, xHost[0].addr, UploadBytes, "write x upload buffer")
    mapAndWrite(yUpload, yHost[0].addr, UploadBytes, "write y upload buffer")
    mapAndWrite(pushUpload, pushHost[0].addr, PushBytes, "write push-constant upload buffer")

    echo "[d3d12] step: create command allocator (ID3D12Device.createCommandAllocator)"
    var allocator: ID3D12CommandAllocator = nil
    let hrAllocator = dev.createCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_ID3D12CommandAllocator.addr, cast[ptr pointer](allocator.addr))
    checkHr(hrAllocator, "create command allocator (ID3D12Device.createCommandAllocator)")
    echo "[d3d12] ok: create command allocator"

    echo "[d3d12] step: create command list (ID3D12Device.createCommandList)"
    var cmdList: ID3D12GraphicsCommandList = nil
    let hrList = dev.createCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nil,
        IID_ID3D12GraphicsCommandList.addr, cast[ptr pointer](cmdList.addr))
    checkHr(hrList, "create command list (ID3D12Device.createCommandList)")
    echo "[d3d12] ok: create command list"

    echo "[d3d12] step: record dispatch (ID3D12GraphicsCommandList.reset)"
    checkHr(cmdList.reset(allocator, nil), "reset command list (ID3D12GraphicsCommandList.reset)")
    cmdList.setPipelineState(pso)
    cmdList.setComputeRootSignature(rootSig)
    cmdList.setComputeRootShaderResourceView(0, xUpload.getGPUVirtualAddress())
    cmdList.setComputeRootShaderResourceView(1, yUpload.getGPUVirtualAddress())
    cmdList.setComputeRootUnorderedAccessView(2, output.getGPUVirtualAddress())
    cmdList.setComputeRootConstantBufferView(3, pushUpload.getGPUVirtualAddress())
    cmdList.dispatch(1, 1, 1)
    checkHr(cmdList.close(), "close command list (ID3D12GraphicsCommandList.close)")
    echo "[d3d12] ok: record dispatch"

    echo "[d3d12] step: execute dispatch (ID3D12CommandQueue.executeCommandLists)"
    var dispatchLists: array[1, ID3D12CommandList]
    dispatchLists[0] = cast[ID3D12CommandList](cmdList)
    queue.executeCommandLists(1, dispatchLists[0].addr)
    echo "[d3d12] ok: execute dispatch"

    echo "[d3d12] step: create fence (ID3D12Device.createFence)"
    var fence: ID3D12Fence = nil
    let hrFence = dev.createFence(0, D3D12_FENCE_FLAG_NONE, IID_ID3D12Fence.addr,
        cast[ptr pointer](fence.addr))
    checkHr(hrFence, "create fence (ID3D12Device.createFence)")
    echo "[d3d12] ok: create fence"

    echo "[d3d12] step: signal fence value 1 (ID3D12CommandQueue.signal)"
    checkHr(queue.signal(fence, 1), "signal fence value 1 (ID3D12CommandQueue.signal)")
    waitForFence(dev, fence, 1, "wait fence value 1")

    # The dispatch and the copy run as two separate ExecuteCommandLists.
    # Each ExecuteCommandLists lets its buffers decay to COMMON on completion.
    # The copy list therefore re-promotes the output buffer to COPY_SRC,
    # no resource barrier is needed.

    echo "[d3d12] step: record copy (ID3D12GraphicsCommandList.reset)"
    checkHr(cmdList.reset(allocator, nil), "reset command list (ID3D12GraphicsCommandList.reset)")
    cmdList.copyBufferRegion(readback, 0, output, 0, UploadBytes)
    checkHr(cmdList.close(), "close command list (ID3D12GraphicsCommandList.close)")
    echo "[d3d12] ok: record copy"

    echo "[d3d12] step: execute copy (ID3D12CommandQueue.executeCommandLists)"
    var copyLists: array[1, ID3D12CommandList]
    copyLists[0] = cast[ID3D12CommandList](cmdList)
    queue.executeCommandLists(1, copyLists[0].addr)
    echo "[d3d12] ok: execute copy"

    echo "[d3d12] step: signal fence value 2 (ID3D12CommandQueue.signal)"
    checkHr(queue.signal(fence, 2), "signal fence value 2 (ID3D12CommandQueue.signal)")
    waitForFence(dev, fence, 2, "wait fence value 2")

    echo "[d3d12] step: map readback (ID3D12Resource.map)"
    var mapped: pointer = nil
    let hrMap = readback.map(0, nil, mapped.addr)
    checkHr(hrMap, "map readback (ID3D12Resource.map)")
    let got = cast[ptr UncheckedArray[float32]](mapped)
    # Reference closed form: alpha = 2.5, x[i] = i + 1, y[i] = 10 * (i + 1).
    # The kernel result alpha * x[i] + y[i] equals 12.5 * (i + 1).
    # Both sides are exact in float32. The tail elements compare against 0,
    # Nim's default initialization of the host arrays.
    for i in 0 ..< ThreadCount:
      let want = if i < DataElements: 12.5'f32 * float32(i + 1) else: 0.0'f32
      if got[i] != want:
        failLoud("[d3d12] readback mismatch at element " & $i & ": got " & $got[i] &
            " want " & $want)
    readback.unmap(0, nil)
    echo "[d3d12] ok: readback compare over " & $ThreadCount & " elements"

    comRelease(cast[pointer](cmdList))
    comRelease(cast[pointer](fence))
    comRelease(cast[pointer](readback))
    comRelease(cast[pointer](output))
    comRelease(cast[pointer](pushUpload))
    comRelease(cast[pointer](yUpload))
    comRelease(cast[pointer](xUpload))
    comRelease(cast[pointer](pso))
    comRelease(cast[pointer](rootSig))
    comRelease(cast[pointer](allocator))
    comRelease(cast[pointer](queue))
    comRelease(cast[pointer](dev))
    unloadLib(api.lib)
    echo "[d3d12] axpy smoke PASSED"
else:
  proc runTest() =
    let hlsl = hlsl:
      proc axpyKernel(output, x, y: ptr UncheckedArray[float32], alpha: float32) {.global.} =
        let i = thread_position_in_grid.x
        output[i] = alpha * x[i] + y[i]
    if hlsl != axpyKernelHlsl:
      failLoud("the fixture's baked HLSL differs from the current kernel output, " &
          "regenerate d3d12_axpy_fixture.nim with " &
          "workspace/crucible/tests/codegen/d3d12/app/d3d12_axpy_fixture_gen.nim")
    echo "[d3d12] ok: fixture HLSL matches the kernel output byte for byte"

    if fourccToString(axpyDxbcWords[0]) != "DXBC":
      failLoud("the fixture container lost the DXBC magic, regenerate it with " &
          "workspace/crucible/tests/codegen/d3d12/app/d3d12_axpy_fixture_gen.nim")
    let fixtureDxil = dxilPartSize(axpyDxbcWords)
    if fixtureDxil <= 0:
      failLoud("the fixture container carries no DXIL part, regenerate it with " &
          "workspace/crucible/tests/codegen/d3d12/app/d3d12_axpy_fixture_gen.nim")
    echo "[d3d12] ok: fixture DXBC container carries a DXIL part of " & $fixtureDxil & " bytes"

    echo "[d3d12] step: compile HLSL through dxc (compileHlslToDxilViaDxc)"
    let fresh = compileHlslToDxilViaDxc(hlsl)
    if fourccToString(fresh[0]) != "DXBC":
      failLoud("dxc produced a container without the DXBC magic")
    let freshDxil = dxilPartSize(fresh)
    if freshDxil <= 0:
      failLoud("dxc produced a container without a DXIL part")
    echo "[d3d12] ok: fresh DXBC container carries a DXIL part of " & $freshDxil & " bytes"
    if fresh != @axpyDxbcWords:
      failLoud("the freshly compiled container differs from the fixture bytes, " &
          "a dxc upgrade or a kernel edit changed the DXIL, regenerate " &
          "d3d12_axpy_fixture.nim with " &
          "workspace/crucible/tests/codegen/d3d12/app/d3d12_axpy_fixture_gen.nim")
    echo "[d3d12] ok: fresh container matches the fixture bytes"

    echo "[d3d12] summary: fixture " & $axpyDxbcWords.len & " words, freshly compiled " &
        $fresh.len & " words, DXIL part " & $freshDxil & " bytes"
    echo "[d3d12] DXIL generation PASSED"

when isMainModule:
  runTest()
