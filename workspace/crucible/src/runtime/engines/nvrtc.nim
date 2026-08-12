# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CudaEngine — NVRTC JIT + CUDA driver execution (moved from codegen/nvrtc.nim).
##
## The NVRTC JIT driver (initNvrtc/compile/getPtx/load and the low-level
## `execute` templates for explicit 2D/3D extents) is runtime code and lives
## on here, decoupled from the compile-time DSL: this module no longer
## imports `codegen/gpu_compiler`.
##
## The engine is a `ref object` with fields directly (no XxxObj indirection).
## Resources live in RAII value fields (`NVRTC` carries its own `=destroy`)
## because Nim 2.2.10 refuses `=destroy` on ref types. Re-ingest replaces the
## RAII field → the old NVRTC context/module are auto-released.

import std/os

import workspace/crucible/src/abis/nvidia_abi
import workspace/crucible/src/abis/nvidia_paths
import workspace/crucible/src/abis/c_abi

import ../exec/cuda_runtime
import ../exec/runtime_utils
import ../engines


## Debug output (driver version, PTX size, files) — controlled by -d:debug

type
  NVRTC* = object
    numBlocks* = 32
    threadsPerBlock* = 128
    # NOTE: there are intentionally NO per-launch `gridDim`/`blockDim` fields on
    # this object. 2D/3D launch extents are passed explicitly to the `execute`
    # template overload that accepts them (see below), so the NVRTC object carries
    # no cross-call launch state: a later scalar `execute` on a reused object can
    # never inherit stale 2D/3D extents, and an incomplete extent is rejected at
    # the call site instead of silently collapsing to 1D.
    name*: string # Name of the program (of the generated in memory CUDA file)
    prog*: nvrtcProgram
    log*: string # The compilation log
    ptx*: string # PTX of the program
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

    var nvrtcMajor, nvrtcMinor: cint
    check nvrtcVersion(nvrtcMajor, nvrtcMinor)
    echo "NVRTC ver: ", nvrtcMajor, ".", nvrtcMinor
    echo "CUDA toolkit (CudaHome): ", CudaHome
    echo "CUDA device runtime (libcudadevrt.a): ", findCudaDevrt()

    var prop: cudaDeviceProp
    check cudaGetDeviceProperties(prop, 0);
    echo "Compute capability: ", prop.major, " ", prop.minor

    writeFile(getDebugPath("kernel.cu"), cuda)
    echo "Kernel dump: ", getDebugPath("kernel.cu")

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

  let status = cuModuleLoadData(nvrtc.module, cstring nvrtc.ptx)
  if status != CUDA_SUCCESS:
    var error_str: cstring #const char* error_str;
    check cuGetErrorString(status, (error_str));
    echo "Module load failed: ", error_str
    echo "JIT Error log: ", error_log
    echo "JIT Info log: ", info_log
    quit(1)

  nvrtc.moduleLoaded = true

# CudaEngine — the HwEngine implementation
# ═════════════════════════════════════════════════════════════════════════

type
  CudaEngine* = ref object
    ## Fields directly (no Obj indirection); resources in the RAII `NVRTC`
    ## value field (fires `=destroy` when the ref dies or is re-ingested).
    source: string
    ptx: string
    nvrtc: NVRTC
    grid, blk: int   # engine-default geometry for the plain `run`

proc newCudaEngine(grid, blk: int): CudaEngine =
  ## Private factory — engines.nim reaches it via `import {.all.}`.
  CudaEngine(grid: grid, blk: blk)
proc ingest*(engine: CudaEngine, source: string) =
  ## NVRTC-compile `source` → PTX. Re-entrant: replaces the previous artifact
  ## and NVRTC context/module (the old RAII field is destroyed).
  if engine.ptx.len > 0:
    when defined(debug):
      echo "[INFO]: cuda ingest: invalidating previous artifact"
  engine.source = source
  engine.nvrtc = initNvrtc(source)
  engine.nvrtc.compile()
  engine.nvrtc.getPtx()
  engine.ptx = engine.nvrtc.ptx

proc getArtifact*(engine: CudaEngine): string =
  ## The compiled PTX.
  engine.ptx

proc runImpl(engine: CudaEngine, kernel: string, output: ArgBlob,
             blobs: seq[ArgBlob], cfg: LaunchConfig) =
  ## Lazy cuModuleLoadData + cuLaunchKernel with ArgBlob marshalling:
  ##   size >= 0 → device buffer (cuMemAlloc + H2D, param = CUdeviceptr)
  ##   size <  0 → by-value scalar (param = host pointer, CUDA reads -size bytes)
  ## The output is always a device buffer, uploaded before launch and read
  ## back after (in-place β·C works). The output is the kernel's first
  ## parameter (binding 0 — output first, per CONVENTIONS.md).
  if not engine.nvrtc.moduleLoaded:
    engine.nvrtc.load()
  check cuModuleGetFunction(engine.nvrtc.kernel, engine.nvrtc.module, kernel)

  let outSize = abs(output.size)

  # Allocate + upload device buffers for every non-scalar arg
  var devPtrs = newSeq[CUdeviceptr](blobs.len + 1)
  var di = 0
  for b in blobs:
    if b.size >= 0:
      if b.size > 0:
        check cuMemAlloc(devPtrs[di], csize_t(b.size))
        check cuMemcpyHtoD(devPtrs[di], b.data, csize_t(b.size))
      inc di
  if outSize > 0:
    check cuMemAlloc(devPtrs[di], csize_t(outSize))
    check cuMemcpyHtoD(devPtrs[di], output.data, csize_t(outSize))
  let outDev = devPtrs[di]
  defer:
    for i in 0 ..< di + (if outSize > 0: 1 else: 0):
      check cuMemFree(devPtrs[i])

  # Assemble the kernel param array: output first (binding 0), then each
  # blob — device ptr value for buffers, host data ptr for scalars.
  var params = newSeq[pointer](blobs.len + 1)
  params[0] = addr outDev
  var bi = 0   # buffer index into devPtrs
  for i, b in blobs:
    if b.size >= 0:
      params[i + 1] = addr devPtrs[bi]
      inc bi
    else:
      params[i + 1] = b.data

  let stream = if cfg.stream == 0: CUstream(nil) else: cast[CUstream](cfg.stream)

  when defined(debug):
    var start, stop: CUevent
    check cuEventCreate(start)
    check cuEventCreate(stop)
    check cuEventRecord(start, CUstream(nil))

  check cuLaunchKernel(
    engine.nvrtc.kernel,
    uint32(cfg.grid), 1'u32, 1'u32,
    uint32(cfg.blk), 1'u32, 1'u32,
    uint32(cfg.sharedMem),
    stream,
    params[0].addr, nil)

  check cuCtxSynchronize()

  when defined(debug):
    check cuEventRecord(stop, CUstream(nil))
    check cuEventSynchronize(stop)
    var elapsedTime: cfloat
    check cuEventElapsedTime(elapsedTime, start, stop)
    echo "[INFO]: Kernel execution took: ", elapsedTime, " ms"
    check cuEventDestroy(start)
    check cuEventDestroy(stop)

  # Read the output back
  if outSize > 0:
    check cuMemcpyDtoH(output.data, outDev, csize_t(outSize))

template run*[T](engine: CudaEngine, kernel: string, output: var T, args: untyped,
              cfg: LaunchConfig): untyped =
  var blobStorage: seq[byte]   # backing store for by-value scalars; lives until scope exit
  runImpl(engine, kernel, outBlob(output), flattenBlobs(args, blobStorage), cfg)

template run*[T](engine: CudaEngine, kernel: string, output: var T, args: untyped): untyped =
  run(engine, kernel, output, args,
      (grid: engine.grid, blk: engine.blk, sharedMem: 0, stream: 0))

# ═════════════════════════════════════════════════════════════════════════
# Low-level NVRTC execute templates (kept for explicit 2D/3D launch extents)
# ═════════════════════════════════════════════════════════════════════════

template execute*(nvrtc: var NVRTC, fn: string, res, inputs: typed, sharedMemSize: typed) =
  ## Load the generated PTX, get the target kernel `fn` and execute it with the
  ## `res` and `inputs`.
  ##
  ## The launch configuration is resolved afresh from the scalar `numBlocks` and
  ## `threadsPerBlock` fields on every call: a 1D `(numBlocks, 1, 1)` grid with a
  ## `(threadsPerBlock, 1, 1)` block. Because the NVRTC object stores no 2D/3D
  ## launch state, a scalar launch can never inherit stale extents from an earlier
  ## explicit-dims launch on the same object. For 2D/3D launches pass explicit
  ## `gridDim`/`blockDim` extents to the dedicated overload below.

  if not nvrtc.moduleLoaded:
    nvrtc.load()

  check cuModuleGetFunction(nvrtc.kernel, nvrtc.module, fn)

  # 1D launch from the scalar fields; y/z default to 1.
  let grid = dim3(nvrtc.numBlocks)
  let blk = dim3(nvrtc.threadsPerBlock)

  # now execute the kernel
  execCuda(nvrtc.kernel, grid, blk, res, inputs, sharedMemSize)

  # synchronize so that e.g. `printf` statements will be printed before we (possibly) quit
  check cuCtxSynchronize() #

template execute*(nvrtc: var NVRTC, fn: string, res, inputs: typed) =
  ## 1D convenience overload: see the shared-memory variant above.
  nvrtc.execute(fn, res, inputs, 0)

template execute*(nvrtc: var NVRTC, fn: string,
                  gridDim, blockDim: CudaDim3,
                  res, inputs: typed, sharedMemSize: typed) =
  ## Load the generated PTX, get the target kernel `fn` and execute it with the
  ## `res` and `inputs`, using `gridDim`/`blockDim` as the explicit 2D/3D launch
  ## extents. The extents are passed per call -- nothing is stored on `nvrtc` --
  ## so a later scalar launch on the same object is unaffected. Every axis of both
  ## extents must be >= 1 (CUDA's valid range); an incomplete extent such as a
  ## y/z-only entry is rejected here rather than silently collapsing to 1D.

  if not nvrtc.moduleLoaded:
    nvrtc.load()

  check cuModuleGetFunction(nvrtc.kernel, nvrtc.module, fn)

  doAssert gridDim.x > 0 and gridDim.y > 0 and gridDim.z > 0,
    "explicit grid extent must have every axis >= 1, got " &
    $gridDim.x & ", " & $gridDim.y & ", " & $gridDim.z
  doAssert blockDim.x > 0 and blockDim.y > 0 and blockDim.z > 0,
    "explicit block extent must have every axis >= 1, got " &
    $blockDim.x & ", " & $blockDim.y & ", " & $blockDim.z

  # now execute the kernel
  execCuda(nvrtc.kernel, gridDim, blockDim, res, inputs, sharedMemSize)

  # synchronize so that e.g. `printf` statements will be printed before we (possibly) quit
  check cuCtxSynchronize() #

template execute*(nvrtc: var NVRTC, fn: string,
                  gridDim, blockDim: CudaDim3,
                  res, inputs: typed) =
  ## 2D/3D convenience overload: see the shared-memory variant above.
  nvrtc.execute(fn, gridDim, blockDim, res, inputs, 0)
