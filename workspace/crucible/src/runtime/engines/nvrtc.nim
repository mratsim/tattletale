# Constantine
# Copyright (c) 2018-2019    Status Research & Development GmbH
# Copyright (c) 2020-Present Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CudaEngine — NVRTC JIT + CUDA driver execution (moved from codegen/nvrtc.nim).
##
## The NVRTC JIT driver helpers (initNvrtc/compile/getPtx/load) are private
## engine internals now: `ingest` compiles via NVRTC and `runImpl` loads the
## module and launches. Launch extents come from the chevron `LaunchConfig`
## (grid/blk are full 3D Dim3) — there is no public low-level `execute`
## entry point. This module no longer imports `codegen/gpu_compiler`.
##
## The engine is a `ref object` with fields directly (no XxxObj indirection).
## Resources live in RAII value fields (`NVRTC` carries its own `=destroy`)
## because Nim 2.2.10 refuses `=destroy` on ref types. Re-ingest replaces the
## RAII field → the old NVRTC context/module are auto-released.
##
## Structure: PUBLIC API block first (exported `*`); PRIVATE machinery below
## (no `*`). `{.experimental: "codeReordering".}` lifts Nim's
## declaration-before-use rule so the private types/helpers may follow the
## public surface that calls them.
{.experimental: "codeReordering".}

import workspace/crucible/src/abis/nvidia_abi
import workspace/crucible/src/abis/c_abi

import ../exec/cuda_runtime
import ../exec/runtime_utils
import ./arg_blobs
import ../chevrons
# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  NVRTC = object
    name*: string # Name of the program (of the generated in memory CUDA file)
    prog*: nvrtcProgram
    log*: string # The compilation log
    ptx*: string # PTX of the program
    device*: CUdevice
    kernel*: CUfunction
    module*: CUmodule
    context*: CUcontext
    moduleLoaded*: bool

  CudaEngine* = ref object
    ## Fields directly (no Obj indirection); resources in the RAII `NVRTC`
    ## value field (fires `=destroy` when the ref dies or is re-ingested).
    source: string
    ptx: string
    nvrtc: NVRTC
    grid, blk: int   # engine-default geometry for the plain `run`

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════
proc `=destroy`(nvrtc: NVRTC) =
  if nvrtc.module.pointer != nil:
    check cuCtxSetCurrent(nvrtc.context)
    check cuModuleUnload nvrtc.module
  if nvrtc.context.pointer != nil:
    check cuCtxSetCurrent(nvrtc.context)
    check cuCtxDestroy nvrtc.context

proc newCudaEngine(grid, blk: int): CudaEngine =
  ## Private factory — engines.nim reaches it via `import {.all.}`.
  CudaEngine(grid: grid, blk: blk)

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc ingest*(engine: CudaEngine, source: string) =
  ## NVRTC-compile `source` → PTX. Re-entrant: replaces the previous artifact
  ## and NVRTC context/module (the old RAII field is destroyed).
  engine.source = source
  engine.nvrtc = initNvrtc(source)
  engine.nvrtc.compile()
  engine.nvrtc.getPtx()
  engine.ptx = engine.nvrtc.ptx

proc getArtifact*(engine: CudaEngine): string =
  ## The compiled PTX.
  engine.ptx

template run*[T](engine: CudaEngine, kernel: string, output: var T, args: untyped,
              cfg: LaunchConfig): untyped =
  var blobStorage: seq[byte]   # backing store for by-value scalars; lives until scope exit
  runImpl(engine, kernel, outBlob(output), flattenBlobs(args, blobStorage), cfg)

template run*[T](engine: CudaEngine, kernel: string, output: var T, args: untyped): untyped =
  run(engine, kernel, output, args,
      LaunchConfig(grid: Dim3(x: engine.grid), blk: Dim3(x: engine.blk)))

# ─────────────────────────────────────────────────────────────────────────
# ▸ PRIVATE
# ─────────────────────────────────────────────────────────────────────────


proc initNvrtc(cuda: string, name = "sample.cu"): NVRTC =
  ## Initializes an NVRTC object for the given program `cuda`
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

proc log(nvrtc: var NVRTC) =
  ## Retrieve the compilation log.
  var logSize: csize_t
  check nvrtcGetProgramLogSize(nvrtc.prog, logSize)

  var log = cstring newString(Natural logSize)

  check nvrtcGetProgramLog(nvrtc.prog, log)
  nvrtc.log = $log # usually empty if no issues found by the compiler

proc compile(nvrtc: var NVRTC) =
  # Compile the program
  # Note: Can specify GPU target architecture explicitly with '-arch' flag.
  var options = @[
    cstring "--gpu-architecture=sm_120", # Blackwell (sm_120)
    cstring "-default-device",           # namespace-scope vars default to __device__
    # "--fmad=false", # and whatever other options for example
  ]
  let numberOfOptions = cint options.len
  let compileResult =  nvrtcCompileProgram(nvrtc.prog, numberOfOptions,
                                           cast[cstringArray](addr options[0]))
  if compileResult != NVRTC_SUCCESS:
    nvrtc.log()
    echo "Compilation log:\n------------------------------"
    echo nvrtc.log
    echo "------------------------------"
  check compileResult

proc getPtx(nvrtc: var NVRTC) =
  ## Obtain PTX from the program.
  var ptxSize: csize_t
  check nvrtcGetPTXSize(nvrtc.prog, ptxSize)

  var ptx = newString(int ptxSize)
  check nvrtcGetPTX(nvrtc.prog, ptx)

  check nvrtcDestroyProgram(nvrtc.prog) # Destroy the program.
  nvrtc.ptx = ptx

proc load(nvrtc: var NVRTC) =
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

proc runImpl(engine: CudaEngine, kernel: string, output: ArgBlob,
             blobs: seq[ArgBlob], cfg: LaunchConfig) =
  ## Lazy cuModuleLoadData + cuLaunchKernel with ArgBlob marshalling:
  ##   size >= 0 → device buffer (cuMemAlloc + H2D, param = CUdeviceptr)
  ##   size <  0 → by-value scalar (param = host pointer, CUDA reads -size bytes)
  ## The output is always a device buffer, uploaded before launch and read
  ## back after (in-place β·C works). The output is the kernel's first
  ## parameter (binding 0 — output first, per CONVENTIONS.md).
  ## The driver API operates on the thread-local current context, so the
  ## engine's context is made current explicitly on every launch (with 2+
  ## engines alive the wrong context could otherwise be targeted).
  check cuCtxSetCurrent(engine.nvrtc.context)
  doAssert cfg.grid.x > 0 and cfg.grid.y > 0 and cfg.grid.z > 0,
    "CUDA grid extent must have every axis > 0, got " &
    $cfg.grid.x & ", " & $cfg.grid.y & ", " & $cfg.grid.z
  doAssert cfg.blk.x > 0 and cfg.blk.y > 0 and cfg.blk.z > 0,
    "CUDA block extent must have every axis > 0, got " &
    $cfg.blk.x & ", " & $cfg.blk.y & ", " & $cfg.blk.z
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

  check cuLaunchKernel(
    engine.nvrtc.kernel,
    uint32(cfg.grid.x), uint32(cfg.grid.y), uint32(cfg.grid.z),
    uint32(cfg.blk.x), uint32(cfg.blk.y), uint32(cfg.blk.z),
    uint32(cfg.sharedMem),
    stream,
    params[0].addr, nil)

  check cuCtxSynchronize()

  # Read the output back
  if outSize > 0:
    check cuMemcpyDtoH(output.data, outDev, csize_t(outSize))
