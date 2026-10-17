# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import std/tables
import workspace/crucible/src/abis/cl_abi

import ../../codegen/ir/gpu_types
import ../engines_common
import ../launch/launch_common
import ../exec/opencl_runtime

export opencl_runtime
# ═════════════════════════════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════════════════════════════
type
  OpenCLCtx = object
    ## RAII value wrapper, `=destroy` fires when the engine ref dies.
    ctx: OpenCLContext

  OpenCLKernelCache = object
    ## RAII value wrapper, cached kernels released when the engine ref dies.
    kernels: Table[string, OpenCLKernel]

  OpenCLEngine* = ref object
    ## An OpenCL engine, build with `bkOpenCL.init()`: OpenCL C kernels on one device.
    ## The context and cached kernels release when the engine dies.
    source: string
    ctx: OpenCLCtx
    kernelCache: OpenCLKernelCache   # after ctx: released before ctx shutdown

proc backendKind*(_: typedesc[OpenCLEngine]): static BackendKind = bkOpenCL

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════
proc `=destroy`(c: var OpenCLCtx) =
  c.ctx.shutdown()   # shutdown is idempotent (nil-guarded)

proc `=destroy`(cache: var OpenCLKernelCache) =
  for _, kern in cache.kernels.mpairs:
    kern.destroyKernel()

proc newOpenCLEngine(): OpenCLEngine =
  OpenCLEngine(ctx: OpenCLCtx(ctx: initOpenCL()))

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc ingest*(engine: OpenCLEngine, source: string) =
  ## Stores the OpenCL C source, re-entrant, cached kernels rebuild lazily.
  if engine.source.len > 0:
    when defined(debug):
      echo "[INFO]: opencl ingest: invalidating previous artifact"
  for _, kern in engine.kernelCache.kernels.mpairs:
    kern.destroyKernel()
  engine.kernelCache.kernels.clear()
  engine.source = source

proc getArtifact*(engine: OpenCLEngine): string =
  ## Returns the OpenCL C kernel source.
  engine.source

proc deviceName*(engine: OpenCLEngine): string {.inline.} =
  ## Returns the OpenCL device name (e.g. "NVIDIA RTX PRO 6000 Blackwell ...").
  engine.ctx.ctx.device.name()


# ─────────────────────────────────────────────────────────────────────────
# ▸ PRIVATE
# ─────────────────────────────────────────────────────────────────────────


proc launchKernelImpl*(engine: OpenCLEngine, kernel: string, output: OpenCLBuffer, args: tuple, cfg = LaunchConfig()) =
  ## Runs `kernel` on the engine's device, building the program on first use.
  ##   engine.launchKernel<<1024, 256>>("saxpy", outDev, (xDev, yDev, 2.0'f32))
  let ctx = engine.ctx.ctx
  # NVIDIA-OpenCL only: the codegen emits Nim `asm(...)` as GCC-style inline
  # asm in the OpenCL C kernel, only NVIDIA's LLVM-based OpenCL compiler
  # accepts it. PTX-embedding kernels (tensor-core mma.sync) also need
  # a work-group of exactly 32 work-items, mma.sync is warp-synchronous.
  if kernel notin engine.kernelCache.kernels:
    engine.kernelCache.kernels[kernel] = ctx.compileKernel(kernel, engine.source)
  var kern = engine.kernelCache.kernels[kernel]

  type SinkCtx = object
    kern: OpenCLKernel
  proc sinkArg(ctx: var SinkCtx, isBuffer: bool, h: OpenCLBuffer, idx: int, p: pointer, len: int) {.nimcall.} =
    if isBuffer:
      ctx.kern.setArg(idx, h)
    else:
      ctx.kern.setArg(idx, len, p)
  var dctx = SinkCtx(kern: kern)
  bindArgs(dctx, bkOpenCL, output, args, false, sinkArg)

  # global = grid·blk work-items, local = blk (per axis, 3D NDRange)
  let globalSize = [cl_size_t(cfg.grid.x * cfg.blk.x),
                    cl_size_t(cfg.grid.y * cfg.blk.y),
                    cl_size_t(cfg.grid.z * cfg.blk.z)]
  let localSize = [cl_size_t(cfg.blk.x), cl_size_t(cfg.blk.y), cl_size_t(cfg.blk.z)]
  kern.runKernel(globalSize, localSize)

proc allocDevice*(engine: OpenCLEngine, n: int): OpenCLBuffer =
  engine.ctx.ctx.allocBuffer(n)

proc freeDevice*(engine: OpenCLEngine, h: OpenCLBuffer) =
  var h = h
  h.dealloc()

proc copyToDeviceFromHost*(engine: OpenCLEngine, dst: OpenCLBuffer, src: pointer, n: int) =
  dst.writeBuffer(src, n)

proc copyToHostFromDevice*(engine: OpenCLEngine, dst: pointer, src: OpenCLBuffer, n: int) =
  src.readBuffer(dst, n)
