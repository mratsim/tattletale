# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import workspace/crucible/src/abis/c_abi
import workspace/crucible/src/abis/nvidia_abi

import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/runtime/errors
import workspace/crucible/src/runtime/exec/cuda_runtime
import workspace/crucible/src/runtime/launch/launch_serialization
import workspace/crucible/src/runtime/engines/engines_common

# ════════════════════════════════════════════════════════════════════════
# ▸ Types
# ════════════════════════════════════════════════════════════════════════

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
    source: string
    ptx: string
    nvrtc: NVRTC

template backendKind*(_: typedesc[CudaEngine]): BackendKind = bkCuda

# ════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ════════════════════════════════════════════════════════════════════════

proc ingest*(engine: CudaEngine, source: string) =
  engine.nvrtc.createProgram(source)
  engine.nvrtc.compile()
  engine.nvrtc.getPtx()
  engine.source = source
  engine.ptx = engine.nvrtc.ptx

proc getArtifact*(engine: CudaEngine): string {.inline.} =
  engine.source

proc deviceName*(engine: CudaEngine): string {.inline.} =
  var name = newString(256)
  check cuDeviceGetName(addr name[0], cint(name.len), engine.nvrtc.device)
  $cast[cstring](addr name[0])

proc allocDevice*[T](engine: CudaEngine, _: typedesc[T], len: int): DeviceMem[CUdeviceptr, T] {.inline.} =
  ## Allocates len elements of T on the engine
  DeviceMem[CUdeviceptr, T](mem: engine.allocDeviceImpl(len * sizeof(T)))

proc copyToDeviceFromHost*[T](engine: CudaEngine, dst: DeviceMem[CUdeviceptr, T], src: ptr T or ptr UncheckedArray[T], n: int) {.inline.} =
  ## Copies n elements from host to device
  engine.copyToDeviceFromHostImpl(dst.mem, src, n * sizeof(T))

proc copyToHostFromDevice*[T](engine: CudaEngine, dst: ptr T or ptr UncheckedArray[T], src: DeviceMem[CUdeviceptr, T], n: int) {.inline.} =
  ## Copies n elements from device to host
  engine.copyToHostFromDeviceImpl(dst, src.mem, n * sizeof(T))

# ════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ════════════════════════════════════════════════════════════════════════

proc `=destroy`(nvrtc: NVRTC) =
  if nvrtc.module.pointer != nil:
    check cuCtxSetCurrent(nvrtc.context)
    check cuModuleUnload nvrtc.module
  if nvrtc.context.pointer != nil:
    check cuCtxSetCurrent(nvrtc.context)
    check cuCtxDestroy nvrtc.context

proc newCudaEngine*(): CudaEngine =
  result = CudaEngine()
  result.nvrtc = initNvrtc()

# ════════════════════════════════════════════════════════════════════════
# ▸ Launch serialization
# ════════════════════════════════════════════════════════════════════════

type SerializationCtx[n: static int] = object
  addrs: array[n, pointer]
  bytes: array[n, seq[byte]]

proc store[n: static int, T](ctx: var SerializationCtx[n], v: T, argIdx: int) =
  when T is ptr|pointer:
    {.error: "Pointer kernel args are not supported, pass the DeviceMem handle from allocDevice, got " & $T.}
  # CUdeviceptr values pack 8 bytes into heap-backed storage, addrs stay valid across the by-value return
  const size = deviceLayout(T, bkCuda).size
  ctx.bytes[argIdx] = newSeq[byte](size)
  if size > 0:
    var off = 0
    ctx.bytes[argIdx].packFrom(off, v, bkCuda)
  ctx.addrs[argIdx] = ctx.bytes[argIdx][0].addr

proc cudaBindArgs(output: auto, args: tuple): auto =
  const n = launchArgCount(typeof(output), typeof(args), CUdeviceptr, bkCuda)
  result = SerializationCtx[n]()
  bindArgs(result, output, args, store, CUdeviceptr, bkCuda)

proc launchKernelImpl*(engine: CudaEngine, kernel: string, output: auto, args: var tuple, cfg: LaunchConfig) =
  ## Runs `kernel` on the engine's GPU, loading the module on first use.
  ## Usage:
  ##   engine.launchKernel<<1024, 256>>("saxpy", outDev, (xDev, yDev, 2.0'f32))

  check cuCtxSetCurrent(engine.nvrtc.context)
  doAssert cfg.grid.x > 0 and cfg.grid.y > 0 and cfg.grid.z > 0,
    "CUDA cfg.grid dims must have every axis > 0, got " &
    $cfg.grid.x & ", " & $cfg.grid.y & ", " & $cfg.grid.z
  doAssert cfg.blk.x > 0 and cfg.blk.y > 0 and cfg.blk.z > 0,
    "CUDA block dims must have every axis > 0, got " &
    $cfg.blk.x & ", " & $cfg.blk.y & ", " & $cfg.blk.z
  if not engine.nvrtc.moduleLoaded:
    engine.nvrtc.load()
  check cuModuleGetFunction(engine.nvrtc.kernel, engine.nvrtc.module, kernel)

  let dctx = cudaBindArgs(output, args)

  let stream = if cfg.stream == 0: CUstream(nil) else: cast[CUstream](cfg.stream)
  check cuLaunchKernel(
    engine.nvrtc.kernel,
    uint32(cfg.grid.x), uint32(cfg.grid.y), uint32(cfg.grid.z),
    uint32(cfg.blk.x), uint32(cfg.blk.y), uint32(cfg.blk.z),
    uint32(cfg.sharedMem),
    stream,
    dctx.addrs[0].addr, nil)

  check cuCtxSynchronize()

proc allocDeviceImpl(engine: CudaEngine, n: int): CUdeviceptr {.inline.} =
  check cuMemAlloc(result, csize_t(n))

proc copyToDeviceFromHostImpl(engine: CudaEngine, dst: CUdeviceptr, src: pointer, n: int) {.inline.} =
  check cuMemcpyHtoD(dst, src, csize_t(n))

proc copyToHostFromDeviceImpl(engine: CudaEngine, dst: pointer, src: CUdeviceptr, n: int) {.inline.} =
  check cuMemcpyDtoH(dst, src, csize_t(n))

# ════════════════════════════════════════════════════════════════════════
# ▸ PRIVATE
# ════════════════════════════════════════════════════════════════════════

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
