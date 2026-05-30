# cutile/cuda_driver.nim
# Layer 0: CUDA Driver API wrappers
#
# Self-contained CUDA Driver API bindings and safe wrappers.
# Based on constantine's nvidia_abi.nim patterns.
#
# Test: tests/t0_cuda_driver.nim

import
  std/[os, strutils]

# ############################################################
# CUDA Driver API types (from cuda.h)
# ############################################################

type
  CUresult* {.size: sizeof(int32).} = enum
    CUDA_SUCCESS = 0
    CUDA_ERROR_INVALID_VALUE = 1
    CUDA_ERROR_OUT_OF_MEMORY = 2
    CUDA_ERROR_NOT_INITIALIZED = 3
    CUDA_ERROR_DEINITIALIZED = 4
    CUDA_ERROR_PROFILER_DISABLED = 5
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
    CUDA_ERROR_STUB_LIBRARY = 34
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46
    CUDA_ERROR_NO_DEVICE = 100
    CUDA_ERROR_INVALID_DEVICE = 101
    CUDA_ERROR_INVALID_IMAGE = 200
    CUDA_ERROR_INVALID_CONTEXT = 201
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221
    CUDA_ERROR_INVALID_HANDLE = 400
    CUDA_ERROR_NOT_FOUND = 500
    CUDA_ERROR_NOT_READY = 600
    CUDA_ERROR_ILLEGAL_ADDRESS = 700
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
    CUDA_ERROR_LAUNCH_TIMEOUT = 702
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 704
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
    CUDA_ERROR_ASSERT = 710
    CUDA_ERROR_TOO_MANY_PEERS = 711
    CUDA_ERROR_LAUNCH_FAILED = 719
    CUDA_ERROR_NOT_PERMITTED = 800
    CUDA_ERROR_NOT_SUPPORTED = 801
    CUDA_ERROR_UNKNOWN = 999

  CUdevice_attribute* {.size: sizeof(int32).} = enum
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

  CUdevice* = distinct int32
  CUcontext* = distinct pointer
  CUmodule* = distinct pointer
  CUfunction* = distinct pointer
  CUstream* = distinct pointer
  CUevent* = distinct pointer
  CUdeviceptr* = distinct int64

# ############################################################
# CUDA Driver API bindings (from libcuda.so)
# ############################################################

const libCuda* = "(libcuda.so|cuda.lib)"

proc cuInit*(flags: uint32): CUresult {.
  importc: "cuInit", dynlib: libCuda, noconv.}

proc cuDeviceGetCount*(count: var int32): CUresult {.
  importc: "cuDeviceGetCount", dynlib: libCuda, noconv.}

proc cuDeviceGet*(device: var CUdevice, ordinal: int32): CUresult {.
  importc: "cuDeviceGet", dynlib: libCuda, noconv.}

proc cuDeviceGetName*(name: ptr char, len: int32, dev: CUdevice): CUresult {.
  importc: "cuDeviceGetName", dynlib: libCuda, noconv.}

proc cuDeviceGetAttribute*(r: var int32, attrib: CUdevice_attribute,
                            dev: CUdevice): CUresult {.
  importc: "cuDeviceGetAttribute", dynlib: libCuda, noconv.}

proc cuCtxCreate*(pctx: var CUcontext, flags: uint32,
                   dev: CUdevice): CUresult {.
  importc: "cuCtxCreate_v2", dynlib: libCuda, noconv.}

proc cuCtxDestroy*(ctx: CUcontext): CUresult {.
  importc: "cuCtxDestroy_v2", dynlib: libCuda, noconv.}

proc cuCtxSynchronize*(): CUresult {.
  importc: "cuCtxSynchronize", dynlib: libCuda, noconv.}

proc cuModuleLoad*(module: var CUmodule, fname: cstring): CUresult {.
  importc: "cuModuleLoad", dynlib: libCuda, noconv.}

proc cuModuleLoadData*(module: var CUmodule, image: pointer): CUresult {.
  importc: "cuModuleLoadData", dynlib: libCuda, noconv.}

proc cuModuleUnload*(module: CUmodule): CUresult {.
  importc: "cuModuleUnload", dynlib: libCuda, noconv.}

proc cuModuleGetFunction*(hfunc: var CUfunction, hmod: CUmodule,
                           name: cstring): CUresult {.
  importc: "cuModuleGetFunction", dynlib: libCuda, noconv.}

proc cuLaunchKernel*(
    kernel: CUfunction,
    gridDimX, gridDimY, gridDimZ: uint32,
    blockDimX, blockDimY, blockDimZ: uint32,
    sharedMemBytes: uint32,
    stream: CUstream,
    kernelParams: ptr pointer,
    extra: ptr pointer
  ): CUresult {.
  importc: "cuLaunchKernel", dynlib: libCuda, noconv.}

proc cuMemAlloc*(devptr: var CUdeviceptr, size: csize_t): CUresult {.
  importc: "cuMemAlloc_v2", dynlib: libCuda, noconv.}

proc cuMemFree*(devptr: CUdeviceptr): CUresult {.
  importc: "cuMemFree_v2", dynlib: libCuda, noconv.}

proc cuMemcpyHtoD*(dst: CUdeviceptr, src: pointer, size: csize_t): CUresult {.
  importc: "cuMemcpyHtoD_v2", dynlib: libCuda, noconv.}

proc cuMemcpyDtoH*(dst: pointer, src: CUdeviceptr, size: csize_t): CUresult {.
  importc: "cuMemcpyDtoH_v2", dynlib: libCuda, noconv.}

proc cuGetErrorString*(error: CUresult, pStr: var cstring): CUresult {.
  importc: "cuGetErrorString", dynlib: libCuda, noconv.}

proc cuEventCreate*(event: var CUevent, flags: cuint): CUresult {.
  importc: "cuEventCreate", dynlib: libCuda, noconv.}

proc cuEventDestroy*(event: CUevent): CUresult {.
  importc: "cuEventDestroy_v2", dynlib: libCuda, noconv.}

proc cuEventRecord*(event: CUevent, stream: CUstream): CUresult {.
  importc: "cuEventRecord", dynlib: libCuda, noconv.}

proc cuEventSynchronize*(event: CUevent): CUresult {.
  importc: "cuEventSynchronize", dynlib: libCuda, noconv.}

proc cuEventElapsedTime*(ms: var cfloat, start, stop: CUevent): CUresult {.
  importc: "cuEventElapsedTime", dynlib: libCuda, noconv.}

proc cuStreamCreate*(pstream: var CUstream, flags: cuint): CUresult {.
  importc: "cuStreamCreate", dynlib: libCuda, noconv.}

proc cuStreamSynchronize*(stream: CUstream): CUresult {.
  importc: "cuStreamSynchronize", dynlib: libCuda, noconv.}

proc cuStreamDestroy*(stream: CUstream): CUresult {.
  importc: "cuStreamDestroy_v2", dynlib: libCuda, noconv.}

# ############################################################
# Error handling
# ############################################################

type
  CudaError* = object of Exception

proc getErrorString*(err: CUresult): string =
  var pStr: cstring
  let r = cuGetErrorString(err, pStr)
  if r == CUDA_SUCCESS:
    result = $pStr
  else:
    result = "unknown error"

template cudaCheck*(stmt: untyped): untyped =
  ## Inline CUDA error checking. Usage: `cudaCheck cuInit(0)`
  let res = stmt
  if res != CUDA_SUCCESS:
    raise newException(CudaError,
      "CUDA error " & $res & ": " & getErrorString(res) & " at " & $stmt)

# ############################################################
# Device management
# ############################################################

proc initCuda*(deviceOrdinal = 0'i32): tuple[device: CUdevice, ctx: CUcontext] =
  ## Initialize CUDA driver and create context.
  cudaCheck cuInit(0)

  var devCount: int32
  cudaCheck cuDeviceGetCount(devCount)
  if devCount == 0:
    raise newException(CudaError, "No CUDA devices available")
  if deviceOrdinal >= devCount:
    raise newException(CudaError,
      "Device ordinal " & $deviceOrdinal & " >= " & $devCount)

  cudaCheck cuDeviceGet(result.device, deviceOrdinal)

  var name = newString(128)
  cudaCheck cuDeviceGetName(name[0].addr, name.len.int32, result.device)
  echo "[cutile] Device [", deviceOrdinal, "]: ", $cstring(name)

  var major, minor: int32
  cudaCheck cuDeviceGetAttribute(
    major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, result.device)
  cudaCheck cuDeviceGetAttribute(
    minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, result.device)
  echo "[cutile] Compute capability: SM ", major, ".", minor

  cudaCheck cuCtxCreate(result.ctx, 0, result.device)

proc closeCuda*(ctx: CUcontext) =
  cudaCheck cuCtxSynchronize()
  cudaCheck cuCtxDestroy(ctx)

proc getSMArch*(device: CUdevice): string =
  ## Get SM architecture string (e.g. "sm_120").
  var major, minor: int32
  cudaCheck cuDeviceGetAttribute(
    major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
  cudaCheck cuDeviceGetAttribute(
    minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
  return "sm_" & $major & $minor

# ############################################################
# Memory management
# ############################################################

type
  DeviceMem* = object
    devPtr*: CUdeviceptr
    size*: csize_t

proc allocDevice*(size: csize_t): DeviceMem =
  result.devPtr = cast[CUdeviceptr](0)
  result.size = size
  cudaCheck cuMemAlloc(result.devPtr, size)

proc freeDevice*(mem: var DeviceMem) =
  if int64(mem.devPtr) != 0:
    cudaCheck cuMemFree(mem.devPtr)
    mem.devPtr = cast[CUdeviceptr](0)
    mem.size = 0

proc h2d*(dst: var DeviceMem, src: pointer, sz: csize_t) =
  cudaCheck cuMemcpyHtoD(dst.devPtr, src, sz)

proc d2h*(src: DeviceMem, dst: pointer, sz: csize_t) =
  cudaCheck cuMemcpyDtoH(dst, src.devPtr, sz)

# ############################################################
# Module management
# ############################################################

proc loadModuleFromFile*(path: string): CUmodule =
  ## Load .cubin or .tilebc file using cuModuleLoad.
  result = cast[CUmodule](nil)
  cudaCheck cuModuleLoad(result, path.cstring)

proc loadModuleFromData*(data: openArray[byte]): CUmodule =
  ## Load module from in-memory binary data.
  result = cast[CUmodule](nil)
  cudaCheck cuModuleLoadData(result, data[0].unsafeAddr)

proc unloadModule*(module: var CUmodule) =
  if pointer(module) != nil:
    cudaCheck cuModuleUnload(module)
    module = cast[CUmodule](nil)

proc getFunction*(module: CUmodule, name: string): CUfunction =
  result = cast[CUfunction](nil)
  cudaCheck cuModuleGetFunction(result, module, name.cstring)

# ############################################################
# Kernel launch (TileIR: block=(1,1,1), shmem=0)
# ############################################################

proc launchKernel*(
    kernel: CUfunction,
    gridX: uint32 = 1, gridY: uint32 = 1, gridZ: uint32 = 1,
    args: openArray[pointer]
  ) =
  ## Launch a TileIR kernel. block=(1,1,1), shmem=0 (TileIR requirement).
  if args.len > 0:
    var params = newSeq[pointer](args.len)
    for i in 0 ..< args.len:
      params[i] = args[i]
    cudaCheck cuLaunchKernel(
      kernel, gridX, gridY, gridZ,  # grid
      1, 1, 1,                       # block (MUST be 1,1,1 for TileIR)
      0,                             # shmem (MUST be 0 for TileIR)
      cast[CUstream](nil),                 # stream
      params[0].unsafeAddr,          # kernelParams
      nil                            # extra
    )
  else:
    cudaCheck cuLaunchKernel(
      kernel, gridX, gridY, gridZ,
      1, 1, 1, 0,
      cast[CUstream](nil), nil, nil)

proc synchronize*() =
  cudaCheck cuCtxSynchronize()

# ############################################################
