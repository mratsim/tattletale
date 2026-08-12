# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CudaEngine — NVRTC JIT + CUDA driver execution (moved from codegen/nvrtc.nim).
##
## The NVRTC JIT driver helpers (initNvrtc/compile/getPtx/load) are private
## engine internals: `ingest` compiles via NVRTC and `runImpl` loads the
## module and launches. Launch extents come from the chevron `LaunchConfig`
## (grid/blk are full 3D Dim3) — there is no public low-level `execute`
## entry point. This module imports no codegen modules. Kernels travel as
## source strings.
##
## The engine is a `ref object` with fields directly (no XxxObj indirection).
## Resources live in RAII value fields (`NVRTC` carries its own `=destroy`)
## because Nim 2.2.10 refuses `=destroy` on ref types. `init` creates the
## CUDA context once. Re-ingest replaces only the program and module. The
## context is released by `=destroy` when the engine dies.
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

proc newCudaEngine(): CudaEngine =
  ## Private factory — engines.nim reaches it via `import {.all.}`.
  ## `init` semantics: live CUDA context, no kernel yet.
  result = CudaEngine()
  result.nvrtc = initNvrtc()

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc ingest*(engine: CudaEngine, source: string) =
  ## NVRTC-compile `source` → PTX. Re-entrant: replaces the previous artifact
  ## and program, while the CUDA context stays alive (created once at init).
  engine.source = source
  engine.nvrtc.createProgram(source)
  engine.nvrtc.compile()
  engine.nvrtc.getPtx()
  engine.ptx = engine.nvrtc.ptx

proc getArtifact*(engine: CudaEngine): string =
  ## The compiled PTX.
  engine.ptx

proc deviceName*(engine: CudaEngine): string {.inline.} =
  ## The CUDA device name (e.g. "NVIDIA RTX PRO 6000 Blackwell ...").
  var name = newString(256)
  check cuDeviceGetName(addr name[0], cint(name.len), engine.nvrtc.device)
  $cast[cstring](addr name[0])


# ─────────────────────────────────────────────────────────────────────────
# ▸ PRIVATE
# ─────────────────────────────────────────────────────────────────────────


proc initNvrtc(): NVRTC =
  ## CUDA context + device handle, no program yet. `ingest` builds the
  ## NVRTC program from the source (createProgram) and compiles it.
  var
    context: CUcontext
    device: CUdevice

  check cuInit(0)
  check cuDeviceGet(device, 0)
  check cuCtxCreate(context, 0, device)

  result = NVRTC(device: device, context: context)

proc createProgram(nvrtc: var NVRTC, source: string) =
  ## Build the NVRTC program from `source`, replacing any previous program.
  ## Re-ingest unloads the previous module (the context stays alive).
  if nvrtc.module.pointer != nil:
    check cuCtxSetCurrent(nvrtc.context)
    check cuModuleUnload nvrtc.module
    nvrtc.module = CUmodule(nil)
    nvrtc.moduleLoaded = false

  # The program name only prefixes NVRTC compile-error lines
  check nvrtcCreateProgram(nvrtc.prog, cstring source, "kernel.cu", 0, nil, nil)

proc log(nvrtc: var NVRTC) =
  ## Retrieve the compilation log.
  var logSize: csize_t
  check nvrtcGetProgramLogSize(nvrtc.prog, logSize)

  var log = cstring newString(Natural logSize)

  check nvrtcGetProgramLog(nvrtc.prog, log)
  nvrtc.log = $log # usually empty if no issues found by the compiler

proc compile(nvrtc: var NVRTC, compute_capability = cudaGetComputeCapability()) =
  # Compile the program. The arch defaults to the queried device capability
  # (compute_120 on a 12.0 GPU) so pre-sm_120 hardware still JIT-lowers the
  # PTX. NVRTC rejects archs below compute_75 with a loud compile error
  # listing the valid targets.
  let arch = "--gpu-architecture=compute_" & $compute_capability
  var options = @[
    cstring arch,
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
    var error_str: cstring
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
  var allocated = 0
  for b in blobs:
    if b.size >= 0:
      if b.size > 0:
        check cuMemAlloc(devPtrs[di], csize_t(b.size))
        check cuMemcpyHtoD(devPtrs[di], b.data, csize_t(b.size))
        inc allocated
      inc di
  if outSize > 0:
    check cuMemAlloc(devPtrs[di], csize_t(outSize))
    check cuMemcpyHtoD(devPtrs[di], output.data, csize_t(outSize))
  let outDev = devPtrs[di]
  defer:
    # Free exactly the slots that were allocated (the shared run layer
    # rejects size-0 blobs, so allocated == di here; tracking separately
    # keeps the defer safe even if that invariant is ever relaxed).
    for i in 0 ..< allocated:
      check cuMemFree(devPtrs[i])
    if outSize > 0:
      check cuMemFree(devPtrs[di])

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
