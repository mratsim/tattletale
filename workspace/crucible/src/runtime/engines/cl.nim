# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import std/[macros, tables]
import workspace/crucible/src/abis/cl_abi

import ../../codegen/ir/gpu_types
import ../engines_common
import ../errors
import ../launch/launch_serialization
import ../exec/opencl_runtime

export opencl_runtime

# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  OpenCLKernelCache = object
    kernels: Table[string, OpenCLKernel]

  OpenCLEngine* = ref object
    source: string
    ctx: OpenCLContext
    kernelCache: OpenCLKernelCache

proc backendKind*(_: typedesc[OpenCLEngine]): static BackendKind = bkOpenCL

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════
proc newOpenCLEngine(): OpenCLEngine =
  OpenCLEngine(ctx: initOpenCL())

# ═════════════════════════════════════════════════════════════════════════
# ▸ Launch serialization
# ═════════════════════════════════════════════════════════════════════════

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

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc ingest*(engine: OpenCLEngine, source: string) =
  if engine.source.len > 0:
    when defined(debug):
      echo "[INFO]: opencl ingest: invalidating previous artifact"
  engine.kernelCache.kernels = initTable[string, OpenCLKernel]()
  engine.source = source

proc getArtifact*(engine: OpenCLEngine): string =
  engine.source

proc deviceName*(engine: OpenCLEngine): string {.inline.} =
  engine.ctx.device.name()


# ─────────────────────────────────────────────────────────────────────────
# ▸ PRIVATE
# ─────────────────────────────────────────────────────────────────────────


proc launchKernelImpl*(engine: OpenCLEngine, kernel: string, output: OpenCLBuffer, args: tuple, grid, blk = Dim3()) =
  ## Runs `kernel` on the engine's device, building the program on first use.
  ##   engine.launchKernel<<1024, 256>>("saxpy", outDev, (xDev, yDev, 2.0'f32))
  if kernel notin engine.kernelCache.kernels:
    engine.kernelCache.kernels[kernel] = engine.ctx.compileKernel(kernel, engine.source)
  let kern = engine.kernelCache.kernels[kernel]

  var dctx = SerializationCtx(kern: kern)
  bindArgs(dctx, output, args, store, OpenCLBuffer)

  let globalSize = [cl_size_t(grid.x * blk.x),
                    cl_size_t(grid.y * blk.y),
                    cl_size_t(grid.z * blk.z)]
  let localSize = [cl_size_t(blk.x), cl_size_t(blk.y), cl_size_t(blk.z)]
  kern.runKernel(globalSize, localSize)

proc allocDevice*(engine: OpenCLEngine, n: int): OpenCLBuffer =
  engine.ctx.allocBuffer(n)

proc freeDevice*(engine: OpenCLEngine, h: OpenCLBuffer) =
  var h = h
  h.dealloc()

proc copyToDeviceFromHost*(engine: OpenCLEngine, dst: OpenCLBuffer, src: pointer, n: int) =
  dst.writeBuffer(src, n)

proc copyToHostFromDevice*(engine: OpenCLEngine, dst: pointer, src: OpenCLBuffer, n: int) =
  src.readBuffer(dst, n)
