# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}

import std/tables
import workspace/crucible/src/abis/cl_abi

import ../exec/opencl_runtime
import ../chevrons

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

# ═════════════════════════════════════════════════════════════════════════
# ▸ Constructors/destructors
# ═════════════════════════════════════════════════════════════════════════
proc `=destroy`(c: var OpenCLCtx) =
  c.ctx.shutdown()   # shutdown is idempotent (nil-guarded)

proc `=destroy`(cache: var OpenCLKernelCache) =
  ## Release cached kernels while the context is still alive.
  for _, kern in cache.kernels.mpairs:
    kern.destroyKernel()

proc newOpenCLEngine(): OpenCLEngine =
  ## Private factory, engines.nim reaches it via `import {.all.}`.
  OpenCLEngine(ctx: OpenCLCtx(ctx: initOpenCL()))

# ═════════════════════════════════════════════════════════════════════════
# ▸ PUBLIC API
# ═════════════════════════════════════════════════════════════════════════

proc ingest*(engine: OpenCLEngine, source: string) =
  ## Store the OpenCL C source. Re-entrant: replaces the previous artifact
  ## and releases cached kernels, which are rebuilt lazily on the next run.
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


proc run*(engine: OpenCLEngine, kernel: string, output: OpenCLBuffer, args: tuple, cfg = LaunchConfig()) =
  ## Runs `kernel` on the engine's device, building the program on first use.
  ##
  ## Usage:
  ##   engine.run<<1024, 256>>("saxpy", outDev, (xDev, yDev, 2.0'f32))
  let ctx = engine.ctx.ctx
  # NVIDIA-OpenCL only: the codegen emits Nim `asm(...)` as GCC-style inline
  # asm in the OpenCL C kernel, only NVIDIA's LLVM-based OpenCL compiler
  # accepts it. PTX-embedding kernels (tensor-core mma.sync) also need
  # a work-group of exactly 32 work-items, mma.sync is warp-synchronous.
  if kernel notin engine.kernelCache.kernels:
    engine.kernelCache.kernels[kernel] = ctx.compileKernel(kernel, engine.source)
  var kern = engine.kernelCache.kernels[kernel]

  kern.setArg(0, output)
  var idx = 1
  var argsCopy = args
  for f in fields(argsCopy):
    when typeof(f) is OpenCLBuffer:
      kern.setArg(idx, f)
    elif typeof(f) is bool:
      # OpenCL C has no bool, the codegen declares scalar bool as `int`
      var v = int32(f)
      kern.setArg(idx, sizeof(int32), addr v)
    else:
      kern.setArg(idx, sizeof(f), addr f)
    inc idx

  # global = grid·blk work-items, local = blk (per axis, 3D NDRange)
  let globalSize = [cl_size_t(cfg.grid.x * cfg.blk.x),
                    cl_size_t(cfg.grid.y * cfg.blk.y),
                    cl_size_t(cfg.grid.z * cfg.blk.z)]
  let localSize = [cl_size_t(cfg.blk.x), cl_size_t(cfg.blk.y), cl_size_t(cfg.blk.z)]
  kern.runKernel(globalSize, localSize)

  proc allocDevice*(engine: OpenCLEngine, n: int): OpenCLBuffer =
    engine.ctx.ctx.allocBuffer(n)

  proc freeDevice*(engine: OpenCLEngine, h: OpenCLBuffer) =
    h.dealloc()

  proc copyToDeviceFromHost*(engine: OpenCLEngine, dst: OpenCLBuffer, src: pointer, n: int) =
    dst.writeBuffer(src, n)

  proc copyToHostFromDevice*(engine: OpenCLEngine, dst: pointer, src: OpenCLBuffer, n: int) =
    src.readBuffer(dst, n)
