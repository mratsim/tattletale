# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import std/[macros, tables]

import workspace/crucible/src/abis/cl_abi

import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/runtime/errors
import workspace/crucible/src/runtime/exec/opencl_runtime
import workspace/crucible/src/runtime/launch/launch_serialization
import workspace/crucible/src/runtime/engines/engines_common

# ════════════════════════════════════════════════════════════════════════
# ▸ Types
# ════════════════════════════════════════════════════════════════════════

type
  OpenCLKernelCache = object
    kernels: Table[string, OpenCLKernel]

  OpenCLEngine* = ref object
    source: string
    ctx: OpenCLContext
    kernelCache: OpenCLKernelCache

template backendKind*(_: typedesc[OpenCLEngine]): BackendKind =
  bkOpenCL

proc newOpenCLEngine*(): OpenCLEngine {.inline.} =
  OpenCLEngine(ctx: initOpenCL())

# ════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ════════════════════════════════════════════════════════════════════════

proc ingest*(engine: OpenCLEngine, source: string) =
  if engine.source.len > 0:
    when defined(debug):
      echo "[INFO]: opencl ingest: invalidating previous artifact"
  engine.kernelCache.kernels = initTable[string, OpenCLKernel]()
  engine.source = source

proc getArtifact*(engine: OpenCLEngine): string {.inline.} =
  engine.source

proc deviceName*(engine: OpenCLEngine): string {.inline.} =
  engine.ctx.device.name()

proc allocDevice*[T](engine: OpenCLEngine, _: typedesc[T], len: int): DeviceMem[OpenCLBuffer, T] {.inline.} =
  ## Allocates len elements of T on the engine
  DeviceMem[OpenCLBuffer, T](mem: engine.allocDeviceImpl(len * sizeof(T)))

proc copyToDeviceFromHost*[T](engine: OpenCLEngine, dst: DeviceMem[OpenCLBuffer, T], src: ptr T or ptr UncheckedArray[T], n: int) {.inline.} =
  ## Copies n elements from host to device
  engine.copyToDeviceFromHostImpl(dst.mem, src, n * sizeof(T))

proc copyToHostFromDevice*[T](engine: OpenCLEngine, dst: ptr T or ptr UncheckedArray[T], src: DeviceMem[OpenCLBuffer, T], n: int) {.inline.} =
  ## Copies n elements from device to host
  engine.copyToHostFromDeviceImpl(dst, src.mem, n * sizeof(T))

# ════════════════════════════════════════════════════════════════════════
# ▸ Launch serialization
# ════════════════════════════════════════════════════════════════════════

type SerializationCtx = object
  kern: OpenCLKernel

proc store(ctx: var SerializationCtx, v: OpenCLBuffer, argIdx: int) =
  ctx.kern.setArg(argIdx, v)

proc store[T](ctx: var SerializationCtx, v: T, argIdx: int) =
  when T is ptr|pointer:
    static:
      error "Pointer kernel args are not supported, pass the DeviceMem handle from allocDevice, got " & $T
  const size = deviceLayout(T, bkOpenCL).size
  var blob: array[size, byte]
  var off = 0
  blob.packFrom(off, v, bkOpenCL)
  ctx.kern.setArg(argIdx, size, blob[0].addr)

# ════════════════════════════════════════════════════════════════════════
# ▸ PRIVATE
# ════════════════════════════════════════════════════════════════════════

proc launchKernelImpl*(engine: OpenCLEngine, kernel: string, output: auto, args: var tuple, cfg: LaunchConfig) =
  ## Runs `kernel` on the engine's device, building the program on first use.
  ##   engine.launchKernel<<1024, 256>>("saxpy", outDev, (xDev, yDev, 2.0'f32))
  if cfg.sharedMem != 0 or cfg.stream != 0:
    failLoud("OpenCL run: launch-time sharedMem/stream are unsupported, they are Metal/CUDA features")
  if kernel notin engine.kernelCache.kernels:
    engine.kernelCache.kernels[kernel] = engine.ctx.compileKernel(kernel, engine.source)
  let kern = engine.kernelCache.kernels[kernel]

  var dctx = SerializationCtx(kern: kern)
  bindArgs(dctx, output, args, store, OpenCLBuffer, bkOpenCL)

  let globalSize = [cl_size_t(cfg.grid.x * cfg.blk.x),
                    cl_size_t(cfg.grid.y * cfg.blk.y),
                    cl_size_t(cfg.grid.z * cfg.blk.z)]
  let localSize = [cl_size_t(cfg.blk.x), cl_size_t(cfg.blk.y), cl_size_t(cfg.blk.z)]
  kern.runKernel(globalSize, localSize)

proc allocDeviceImpl(engine: OpenCLEngine, n: int): OpenCLBuffer {.inline.} =
  engine.ctx.allocBuffer(n)

proc copyToDeviceFromHostImpl(engine: OpenCLEngine, dst: OpenCLBuffer, src: pointer, n: int) {.inline.} =
  dst.writeBuffer(src, n)

proc copyToHostFromDeviceImpl(engine: OpenCLEngine, dst: pointer, src: OpenCLBuffer, n: int) {.inline.} =
  src.readBuffer(dst, n)
