# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import workspace/crucible/src/abis/nvidia_abi
import workspace/crucible/src/abis/c_abi

import ../../codegen/ir/gpu_types
import ../exec/cuda_runtime
import ../engines_common
import ../errors
import ../launch/launch_serialization
# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  NVRTC = object
    prog: nvrtcProgram
    log: string
    ptx: string
    device: CUdevice
    kernel: CUfunction
    module: CUmodule
    context: CUcontext
    moduleLoaded: bool

  CudaEngine* = ref object
    ptx: string
    nvrtc: NVRTC

proc backendKind*(_: typedesc[CudaEngine]): static BackendKind = bkCuda

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
  result = CudaEngine()
  result.nvrtc = initNvrtc()

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc ingest*(engine: CudaEngine, source: string) =
  engine.nvrtc.createProgram(source)
  engine.nvrtc.compile()
  engine.nvrtc.getPtx()
  engine.ptx = engine.nvrtc.ptx

proc getArtifact*(engine: CudaEngine): string =
  engine.ptx

proc deviceName*(engine: CudaEngine): string {.inline.} =
  var name = newString(256)
  check cuDeviceGetName(addr name[0], cint(name.len), engine.nvrtc.device)
  $cast[cstring](addr name[0])


# ─────────────────────────────────────────────────────────────────────────
# ▸ PRIVATE
# ─────────────────────────────────────────────────────────────────────────


proc initNvrtc(): NVRTC =
  var
    context: CUcontext
    device: CUdevice

  check cuInit(0)
  check cuDeviceGet(device, 0)
  check cuCtxCreate(context, 0, device)

  result = NVRTC(device: device, context: context)

proc createProgram(nvrtc: var NVRTC, source: string) =
  if nvrtc.module.pointer != nil:
    check cuCtxSetCurrent(nvrtc.context)
    check cuModuleUnload nvrtc.module
    nvrtc.module = CUmodule(nil)
    nvrtc.moduleLoaded = false

  check nvrtcCreateProgram(nvrtc.prog, cstring source, "kernel.cu", 0, nil, nil)

proc log(nvrtc: var NVRTC) =
  var logSize: csize_t
  check nvrtcGetProgramLogSize(nvrtc.prog, logSize)

  var log = cstring newString(Natural logSize)

  check nvrtcGetProgramLog(nvrtc.prog, log)
  nvrtc.log = $log # usually empty if no issues found by the compiler

proc compile(nvrtc: var NVRTC, compute_capability = cudaGetComputeCapability()) =
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
  var ptxSize: csize_t
  check nvrtcGetPTXSize(nvrtc.prog, ptxSize)

  var ptx = newString(int ptxSize)
  check nvrtcGetPTX(nvrtc.prog, ptx)

  check nvrtcDestroyProgram(nvrtc.prog)
  nvrtc.ptx = ptx

proc load(nvrtc: var NVRTC) =
  let status = cuModuleLoadData(nvrtc.module, cstring nvrtc.ptx)
  if status != CUDA_SUCCESS:
    var error_str: cstring
    check cuGetErrorString(status, error_str)
    failLoud("Module load failed: " & $error_str)

  nvrtc.moduleLoaded = true


proc launchKernelImpl*(engine: CudaEngine, kernel: string, output: CUdeviceptr, args: tuple, grid, blk = Dim3()) =
  ## Runs `kernel` on the engine's GPU, loading the module on first use.
  ## Usage:
  ##   engine.launchKernel<<1024, 256>>("saxpy", outDev, (xDev, yDev, 2.0'f32))

  check cuCtxSetCurrent(engine.nvrtc.context)
  doAssert grid.x > 0 and grid.y > 0 and grid.z > 0,
    "CUDA grid dims must have every axis > 0, got " &
    $grid.x & ", " & $grid.y & ", " & $grid.z
  doAssert blk.x > 0 and blk.y > 0 and blk.z > 0,
    "CUDA block dims must have every axis > 0, got " &
    $blk.x & ", " & $blk.y & ", " & $blk.z
  if not engine.nvrtc.moduleLoaded:
    engine.nvrtc.load()
  check cuModuleGetFunction(engine.nvrtc.kernel, engine.nvrtc.module, kernel)

  const n = launchArgCount(typeof(output), typeof(args), CUdeviceptr, bkCuda)

  type SerializationCtx = object
    addrs: array[n, pointer]
    hvals: array[n, CUdeviceptr]

  proc serializer(ctx: var SerializationCtx, bufferHandle: CUdeviceptr, p: pointer, len: int, argIdx: int, viaBuffer: bool) {.nimcall.} =
    if viaBuffer:
      ctx.hvals[argIdx] = bufferHandle
      ctx.addrs[argIdx] = ctx.hvals[argIdx].addr
    else:
      ctx.addrs[argIdx] = p
      
  var dctx = SerializationCtx()
  bindArgs(dctx, output, args, serializer, bkCuda, false)

  check cuLaunchKernel(
    engine.nvrtc.kernel,
    uint32(grid.x), uint32(grid.y), uint32(grid.z),
    uint32(blk.x), uint32(blk.y), uint32(blk.z),
    0, 0,  # no shared memory, no stream
    dctx.addrs[0].addr, nil)

  check cuCtxSynchronize()

proc allocDevice*(engine: CudaEngine, n: int): CUdeviceptr =
  check cuMemAlloc(result, csize_t(n))

proc freeDevice*(engine: CudaEngine, h: CUdeviceptr) =
  check cuMemFree(h)

proc copyToDeviceFromHost*(engine: CudaEngine, dst: CUdeviceptr, src: pointer, n: int) =
  check cuMemcpyHtoD(dst, src, csize_t(n))

proc copyToHostFromDevice*(engine: CudaEngine, dst: pointer, src: CUdeviceptr, n: int) =
  check cuMemcpyDtoH(dst, src, csize_t(n))
