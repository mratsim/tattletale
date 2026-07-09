# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[strformat, os]

import workspace/crucible/src/abis/nvidia_abi
import workspace/crucible/src/abis/nvidia_paths

import ./gpu_compiler
import ./exec/cuda_runtime
export cuda_runtime
export gpu_compiler
export nvidia_abi
import workspace/crucible/src/abis/c_abi
export c_abi
import ./exec/runtime_utils

## Debug output (driver version, PTX size, files) — controlled by -d:debug

type
  NVRTC* = object
    numBlocks* = 32
    threadsPerBlock* = 128
    name*: string # Name of the program (of the generated in memory CUDA file)
    prog*: nvrtcProgram
    log*: string # The compilation log
    ptx*: string # PTX of the program
    cubin*: pointer
    cubinSize*: csize_t
    device*: CUdevice
    kernel*: CUfunction
    module*: CUmodule
    context*: CUcontext
    moduleLoaded*: bool

proc `=destroy`(nvrtc: NVRTC) =
  if nvrtc.module.pointer != nil:
    check cuModuleUnload nvrtc.module
  if nvrtc.context.pointer != nil:
    check cuCtxDestroy nvrtc.context

proc initNvrtc*(cuda: string, name = "sample.cu"): NVRTC =
  ## Initializes an NVRTC object for the given program `cuda`
  when defined(debug):
    var x: cint
    check cuDriverGetVersion(x)
    echo "Driver version: ", x

    var rtVer: cint
    check cudaRuntimeGetVersion(rtVer)
    echo "Runtime ver: ", rtVer

    var prop: cudaDeviceProp
    check cudaGetDeviceProperties(prop, 0);
    echo "Compute capability: ", prop.major, " ", prop.minor

    writeFile(dumpPath, cuda)
    echo "Kernel dump: ", dumpPath

  ## TODO: consider in-memory and on-disk caching option for compiled PTX.
  ## (Compile once, reuse PTX or CUmodule for subsequent runs.)
  var
    context: CUcontext
    device: CUdevice

  check cuInit(0)
  check cuDeviceGet(device, 0)
  check cuCtxCreate(context, 0, device)

  # Create an instance of nvrtcProgram based on the passed code
  var prog: nvrtcProgram
  check nvrtcCreateProgram(prog, cstring cuda, cstring name, 0, nil, nil)

  result = NVRTC(prog: prog, name: name,
                 device: device,
                 context: context)


proc log*(nvrtc: var NVRTC) =
  ## Retrieve the compilation log.
  var logSize: csize_t
  check nvrtcGetProgramLogSize(nvrtc.prog, logSize)

  var log = cstring newString(Natural logSize)

  check nvrtcGetProgramLog(nvrtc.prog, log)
  nvrtc.log = $log # usually empty if no issues found by the compiler

proc compile*(nvrtc: var NVRTC) =
  # Compile the program
  # Note: Can specify GPU target architecture explicitly with '-arch' flag.
  var options = @[
    cstring "--gpu-architecture=sm_120", # Blackwell (sm_120)
    cstring "-default-device",           # namespace-scope vars default to __device__
    # "--fmad=false", # and whatever other options for example
  ]
  when defined(debug):
    options.add cstring "--device-debug"       # Equivalent to -g
    options.add cstring "--generate-line-info" # Equivalent to -lineinfo

  let numberOfOptions = cint options.len
  let compileResult =  nvrtcCompileProgram(nvrtc.prog, numberOfOptions,
                                           cast[cstringArray](addr options[0]))
  ## XXX: only in `DebugCuda`?
  if compileResult != NVRTC_SUCCESS:
    nvrtc.log()
    echo "Compilation log:\n------------------------------"
    echo nvrtc.log
    echo "------------------------------"
  check compileResult

proc getPtx*(nvrtc: var NVRTC) =
  ## Obtain PTX from the program.
  var ptxSize: csize_t
  check nvrtcGetPTXSize(nvrtc.prog, ptxSize)

  var ptx = newString(int ptxSize)
  check nvrtcGetPTX(nvrtc.prog, ptx)

  check nvrtcDestroyProgram(nvrtc.prog) # Destroy the program.
  nvrtc.ptx = ptx

  when defined(debug):
    writeFile(getDebugPath("kernel.ptx"), nvrtc.ptx)
    #echo "-------------------- PTX --------------------\n", nvrtc.ptx

proc load*(nvrtc: var NVRTC) =
  # After getting the PTX...
  var error_log = newString(8192)
  var info_log = newString(8192)

  ## NOTE: if you wish to use the `link` approach, pass `nvrtc.cubin` instead of `PTX`
  #let status = cuModuleLoadData(addr nvrtc.module, nvrtc.cubin)
  let status = cuModuleLoadData(nvrtc.module, cstring nvrtc.ptx)
  if status != CUDA_SUCCESS:
    var error_str: cstring #const char* error_str;
    check cuGetErrorString(status, (error_str));
    echo "Module load failed: ", error_str
    echo "JIT Error log: ", error_log
    echo "JIT Info log: ", info_log
    quit(1)

  nvrtc.moduleLoaded = true

proc link*(nvrtc: var NVRTC) =
  ## OPTIONAL STEP. Alternative to passing the PTX to `cuModuleLoadData`.
  # Create linker
  var linkState: CUlinkState
  var linkOptions: array[4, CUjit_option]
  var linkOptionValues: array[4, pointer]
  var errorLog = newString(4096)
  var infoLog = newString(4096)
  var walltime: float32

  linkOptions[0] = CU_JIT_WALL_TIME
  linkOptionValues[0] = addr walltime
  linkOptions[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
  linkOptionValues[1] = cast[pointer](4096)
  linkOptions[2] = CU_JIT_ERROR_LOG_BUFFER
  linkOptionValues[2] = addr errorLog[0]
  linkOptions[3] = CU_JIT_INFO_LOG_BUFFER
  linkOptionValues[3] = addr infoLog[0]

  check cuLinkCreate(3, addr linkOptions[0], addr linkOptionValues[0], addr linkState)

  # Add PTX
  var res = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
                          cast[pointer](cstring nvrtc.ptx),
                          csize_t nvrtc.ptx.len,
                          "kernel.ptx", 0, nil, nil)

  var status: CUresult
  if res != CUDA_SUCCESS:
    var error_str: cstring
    check cuGetErrorString(res, error_str)
    echo "Link add PTX failed: ", error_str
    echo "Error log: ", errorLog
    quit(1)

  # Add the device runtime (provides printf support)
  ## NOTE: Linking requires you to pass the path to `libcudadevrt.a` at CT
  let devrtPath = findCudaDevrt()
  if devrtPath.len > 0:
    res = cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY,
                        cstring devrtPath, 0, nil, nil)
    if res != CUDA_SUCCESS:
      var error_str: cstring
      check cuGetErrorString(res, error_str)
      echo "Link add device runtime failed: ", error_str
      echo "Error log: ", errorLog
      quit(1)

  # Complete linking
  var cubn: pointer
  var cubinSize: csize_t
  res = cuLinkComplete(linkState, cubn.addr, cubinSize.addr)
  nvrtc.cubinSize = cubinSize
  if res != CUDA_SUCCESS:
    var error_str: cstring
    check cuGetErrorString(res, error_str);
    echo "Link complete failed: ", error_str
    echo "Error log: ", errorLog
    quit(1)

  when defined(debug):
    let cubinDump = getDebugPath("test.cubin")
    echo "[INFO]: Writing CUBIN data to file ", cubinDump
    echo "Cubin size: ", cubinSize
    var f = open(cubinDump, fmWrite)
    defer: f.close()
    discard f.writeBuffer(cubn, cubinSize)

  # Assign the cubin
  nvrtc.cubin = cubn

proc copyToSymbol*[T](nvrtc: NVRTC, symbol: string, data: T, offset = 0) =
  ## Copies `data` to the symbol in the current CUDA kernel.
  ## There is NO type safety involved here. We only check that the amount of
  ## data you wish to copy to the global matches the size of the global storage.
  ## This function does help you with automatically copying `seq[T]` for example.
  ##
  ## `offset` is an optional offset of the number of bytes at the target we want
  ## to copy to. Useful to copy only individual elements of a constant array for example.
  ##
  ## Say you define in a kernel:
  ##
  ## ```nim
  ## let foo = cuda:
  ##   var data {.constant.}: array[1024, uint32]
  ## # ...
  ## # later in host code after getting the kernel from the `nvrtc` object:
  ## let data = calcSomeArray1024() # runtime calculation
  ## copyToSymbols("data", # name of the variable in CUDA code
  ##               data)
  ## ```
  var devPtr: CUdeviceptr
  var size: csize_t
  check cuModuleGetGlobal(devPtr, size.addr, nvrtc.module, symbol.cstring)
  var totSize: int
  var srcPtr: pointer
  when T is seq: # copy len * sizeof(element)
    doAssert data.len > 0, "Input data is empty!"
    let elSize = sizeof(data[0])
    totSize = data.len * elSize
    srcPtr = data[0].addr

  else:
    # For now just copy by `sizeof`!
    totSize = sizeof(data)
    srcPtr = data.addr
  doAssert csize_t(totSize) <= size, "Input data size does not fit target: " & $totSize & " vs. " & $size
  check cuMemcpyHtoD(devPtr + csize_t(offset), srcPtr, csize_t(totSize))

template execute*(nvrtc: var NVRTC, fn: string, res, inputs: typed, sharedMemSize: typed) =
  ## Load the generated PTX, get the target kernel `fn` and execute it with the `res` and `inputs`

  if not nvrtc.moduleLoaded:
    nvrtc.load()

  check cuModuleGetFunction(nvrtc.kernel, nvrtc.module, fn)

  # now execute the kernel
  execCuda(nvrtc.kernel, nvrtc.numBlocks, nvrtc.threadsPerBlock, res, inputs, sharedMemSize)

  # synchronize so that e.g. `printf` statements will be printed before we (possibly) quit
  check cuCtxSynchronize() #

template execute*(nvrtc: var NVRTC, fn: string, res, inputs: typed) =
  nvrtc.execute(fn, res, inputs, 0)
