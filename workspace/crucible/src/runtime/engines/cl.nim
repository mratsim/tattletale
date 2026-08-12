# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## OpenCLEngine — OpenCL runtime compilation and execution (moved from
## codegen/cl.nim, decoupled from the compile-time DSL: no gpu_compiler import).
##
## ingest = store source; getArtifact = source; run = build program +
## enqueueNDRangeKernel with grid/blk → global/local sizes (global = grid·blk,
## local = blk). The output's current bytes are uploaded before launch
## (replaces opencl_runtime's `outputInit` hack — in-place β·C works on all
## backends). Scalars bind by value via the taggedArgs isValue semantics
## (ArgBlob size < 0).
##
## Backend constraint — NVIDIA-OpenCL only: the OpenCL codegen emits Nim
## `asm(...)` statements as GCC-style inline asm in the OpenCL C kernel.
## Only NVIDIA's OpenCL compiler (nvopencl, LLVM-based) accepts inline PTX
## asm; Intel/AMD/POCL implementations reject such kernels at build time.
## Kernels that embed PTX (e.g. tensor-core mma.sync) must also run with a
## work-group of exactly one warp (32 work-items) — mma.sync is
## warp-synchronous. See
## workspace/ceramic/tests/gemm/manual_sm80_tensor_cores_opencl.nim
## for a full example with both the vendor check and the warp guard.

import std/strformat

import workspace/crucible/src/abis/cl_abi

import ../exec/opencl_runtime
import ../engines

export opencl_runtime

type
  OpenCLCtx = object
    ## RAII value wrapper — `=destroy` fires when the engine ref dies.
    ctx: OpenCLContext

  OpenCLEngine* = ref object
    ## Fields directly (no Obj indirection); resources in the RAII value field.
    source: string
    ctx: OpenCLCtx
    grid, blk: int   # engine-default geometry for the plain `run`

proc `=destroy`(c: var OpenCLCtx) =
  c.ctx.shutdown()   # shutdown is idempotent (nil-guarded)

proc newOpenCLEngine(grid, blk: int): OpenCLEngine =
  ## Private factory — engines.nim reaches it via `import {.all.}`.
  OpenCLEngine(ctx: OpenCLCtx(ctx: initOpenCL()), grid: grid, blk: blk)

proc ingest*(engine: OpenCLEngine, source: string) =
  ## Store the OpenCL C source. Re-entrant: replaces the previous artifact.
  if engine.source.len > 0:
    when defined(debug):
      echo "[INFO]: opencl ingest: invalidating previous artifact"
  engine.source = source

proc getArtifact*(engine: OpenCLEngine): string =
  ## The OpenCL C kernel source.
  engine.source

proc deviceVendor*(engine: OpenCLEngine): string =
  ## The OpenCL device vendor (e.g. NVIDIA) — used by tests that require
  ## NVIDIA's OpenCL compiler for inline-PTX asm kernels.
  engine.ctx.ctx.device.vendor()

proc runImpl(engine: OpenCLEngine, kernel: string, output: ArgBlob,
             blobs: seq[ArgBlob], cfg: LaunchConfig) =
  ## Build program + enqueueNDRangeKernel. Bindings follow OpenCL kernel
  ## parameter order: args 0..N-1 (buffers as cl_mem, scalars by value),
  ## arg N = output (output-last convention; param-order unification across backends is pending).
  let ctx = engine.ctx.ctx
  let numInputs = blobs.len
  let outSize = abs(output.size)

  # Allocate input buffers + output buffer
  var inputBuffers = newSeq[OpenCLBuffer](numInputs)
  defer:
    for i in 0 ..< numInputs:
      if blobs[i].size >= 0:
        inputBuffers[i].dealloc()
  for i in 0 ..< numInputs:
    if blobs[i].size >= 0:
      inputBuffers[i] = ctx.allocBuffer(blobs[i].size)
  var outBuf = ctx.allocBuffer(outSize)
  defer:
    outBuf.dealloc()

  # Upload the output's current contents before launch (in-place β·C)
  if outSize > 0:
    outBuf.writeBuffer(output.data, outSize)

  var kern = ctx.compileKernel(kernel, engine.source)
  defer:
    kern.destroyKernel()

  # Write input data
  for i in 0 ..< numInputs:
    if blobs[i].size >= 0:
      inputBuffers[i].writeBuffer(blobs[i].data, blobs[i].size)

  # Set kernel args: inputs first (buffers as cl_mem, scalars by value),
  # then output
  for i in 0 ..< numInputs:
    if blobs[i].size >= 0:
      kern.setArg(i, inputBuffers[i])
    else:
      kern.setArg(i, -blobs[i].size, blobs[i].data)
  kern.setArg(numInputs, outBuf)

  # global = grid·blk work-items, local = blk
  let globalSize = [cl_size_t(cfg.grid * cfg.blk)]
  let localSize = [cl_size_t(cfg.blk)]
  kern.runKernel(globalSize, localSize)

  # Read output
  if outSize > 0:
    outBuf.readBuffer(output.data, outSize)

template run*[T](engine: OpenCLEngine, kernel: string, output: var T, args: untyped,
              cfg: LaunchConfig): untyped =
  var blobStorage: seq[byte]   # backing store for by-value scalars; lives until scope exit
  runImpl(engine, kernel, outBlob(output), flattenBlobs(args, blobStorage), cfg)

template run*[T](engine: OpenCLEngine, kernel: string, output: var T, args: untyped): untyped =
  run(engine, kernel, output, args,
      (grid: engine.grid, blk: engine.blk, sharedMem: 0, stream: 0))
