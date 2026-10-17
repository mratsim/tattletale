# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
import std/[strutils, tables]

import ../../codegen/ir/gpu_types
import ../engines_common
import ../errors
import ../launch/launch_serialization
import workspace/crucible/src/abis/objc_abi as objc
import ../exec/metal_runtime

export metal_runtime

when defined(macosx):
  import ../launch/launch_kernel
# ═════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════
type
  MetalCache = object
    ## Two per-engine caches:
    ## - library, keyed by source, replaced by re-ingest
    ## - pipeline states, keyed by kernel name, cleared by re-ingest
    library: objc.ID
    psos: Table[string, objc.ID]

  MetalEngine* = ref object
    source: string
    ctx: MetalCtx
    cache: MetalCache   # after ctx: released before the context destroy

proc backendKind*(_: typedesc[MetalEngine]): static BackendKind = bkMetal

# ═════════════════════════════════════════════════
# ▸ macOS-only engine implementation
# ═════════════════════════════════════════════════

when defined(macosx):
  # ─────────────────────────────────────────────────────────────────────────
  # ▸ PRIVATE constants
  # ─────────────────────────────────────────────────────────────────────────

  const
    ## `MTLCommandBufferStatus` values (MTLCommandBuffer.h): 4 = completed,
    ## 5 = error. `waitUntilCompleted` leaves the buffer in one of these.
    MTLCommandBufferStatusCompleted = 4
    MTLCommandBufferStatusError = 5

  # ─────────────────────────────────────────────────────────────────────────
  # ▸ Constructors/destructors
  # ─────────────────────────────────────────────────────────────────────────

  proc newMetalCtx(): MetalCtx =
    result.device = objc.MTLCreateSystemDefaultDevice()
    if objc.isNil(result.device):
      failLoud("no Metal device")
    result.queue = objc.msgSend(result.device, objc.`$$`("newCommandQueue"))
    if objc.isNil(result.queue):
      failLoud("newCommandQueue returned nil")


  proc `=destroy`(cache: var MetalCache) =
    objc.withMemPool:
      objc.release(cache.library)
      for pso in cache.psos.values:
        objc.release(pso)
      cache.psos.clear()

  proc newMetalEngine(): MetalEngine =
    objc.withMemPool:
      result = MetalEngine(
        ctx: newMetalCtx(),
        cache: MetalCache(psos: initTable[string, objc.ID]())
      )

  # ─────────────────────────────────────────────────────────────────────────
  # ▸ PUBLIC API
  # ─────────────────────────────────────────────────────────────────────────

  proc ingest*(engine: MetalEngine, source: string) =
    objc.withMemPool:
      let opts = compileOptions()
      let library = compileLibrary(engine.ctx.device, source, opts)
      objc.release(opts)
      when defined(debug):
        echo "[INFO]: metal ingest: invalidating previous artifact"
      objc.release(engine.cache.library)
      for pso in engine.cache.psos.values:
        objc.release(pso)
      engine.cache.psos.clear()
      engine.cache.library = library
      engine.source = source

  proc getArtifact*(engine: MetalEngine): string =
    engine.source

  proc deviceName*(engine: MetalEngine): string =
    objc.withMemPool:
      result = objc.nsStringToNimString(objc.msgSend(engine.ctx.device, objc.`$$`("name")))

  # ─────────────────────────────────────────────────────────────────────────
  # ▸ PRIVATE run path
  # ─────────────────────────────────────────────────────────────────────────

  proc bufferContents(h: objc.ID): pointer =
    result = cast[pointer](objc.msgSend(h, objc.`$$`("contents")))
    if result == nil:
      failLoud("contents() returned nil for a shared buffer")

  proc allocDevice*(engine: MetalEngine, n: int): objc.ID =
    if n <= 0:
      failLoud("allocDevice: size must be positive, got " & $n)
    result = objc.msgSend(engine.ctx.device, objc.`$$`("newBufferWithLength:options:"),
                          objc.NSUInteger(n), SharedUntrackedOptions)
    if objc.isNil(result):
      failLoud("allocDevice: newBufferWithLength:options: returned nil (length " & $n & ")")

  proc freeDevice*(engine: MetalEngine, h: objc.ID) =
    if not objc.isNil(h):
      objc.release(h)

  proc copyToDeviceFromHost*(engine: MetalEngine, dst: objc.ID, src: pointer, n: int) =
    if n <= 0:
      failLoud("copyToDeviceFromHost: n must be positive, got " & $n)
    copyMem(bufferContents(dst), src, n)

  proc copyToHostFromDevice*(engine: MetalEngine, dst: pointer, src: objc.ID, n: int) =
    if n <= 0:
      failLoud("copyToHostFromDevice: n must be positive, got " & $n)
    copyMem(dst, bufferContents(src), n)


  proc launchKernelImpl*(engine: MetalEngine, kernel: string, output: auto, args: tuple, grid, blk = Dim3()) =
    ## Runs `kernel` with the launch values bound whole, bridges and direct runs alike.
    ##   engine.launchKernel<<(1024, 256)>>("saxpy", outDev, (xDev, yDev, 2.0'f32))
    objc.withMemPool:
      if grid.x < 1 or grid.y < 1 or grid.z < 1:
        failLoud("Metal run: grid must be ≥ 1 per axis, got " & $grid.x & "x" & $grid.y & "x" & $grid.z)
      if blk.x < 1 or blk.y < 1 or blk.z < 1:
        failLoud("Metal run: blk must be ≥ 1 per axis, got " & $blk.x & "x" & $blk.y & "x" & $blk.z)
      if blk.x * blk.y * blk.z > 1024:
        failLoud("Metal run: blk " & $blk.x & "x" & $blk.y & "x" & $blk.z & " exceeds the 1024 threads-per-threadgroup max")

      var pso: objc.ID
      if engine.cache.psos.hasKey(kernel):
        pso = engine.cache.psos[kernel]
      else:
        pso = compilePipelineState(engine.ctx.device, engine.cache.library, kernel)
        engine.cache.psos[kernel] = pso

      let cmdBuf = objc.msgSend(engine.ctx.queue, objc.`$$`("commandBuffer"))
      let encoder = objc.msgSend(cmdBuf, objc.`$$`("computeCommandEncoder"))
      discard objc.msgSend(encoder, objc.`$$`("setComputePipelineState:"), pso)


      type SerializationCtx = object
        encoder: objc.ID

      proc serializer(ctx: var SerializationCtx, bufferHandle: objc.ID, p: pointer, len: int, argIdx: int, viaBuffer: bool) {.nimcall.} =
        if viaBuffer:
          discard objc.msgSend(ctx.encoder, objc.`$$`("setBuffer:offset:atIndex:"),
                          bufferHandle, objc.NSUInteger(0), objc.NSUInteger(argIdx))
        else:
          discard objc.msgSend(ctx.encoder, objc.`$$`("setBytes:length:atIndex:"),
                          p, objc.NSUInteger(len), objc.NSUInteger(argIdx))

      var dctx = SerializationCtx(encoder: encoder)
      bindArgs(dctx, output, args, serializer, bkMetal, true)
      let grid = objc.MTLSize(width: objc.NSUInteger(grid.x),
                         height: objc.NSUInteger(grid.y),
                         depth: objc.NSUInteger(grid.z))
      let blk = objc.MTLSize(width: objc.NSUInteger(blk.x),
                        height: objc.NSUInteger(blk.y),
                        depth: objc.NSUInteger(blk.z))
      discard objc.msgSend(encoder, objc.`$$`("dispatchThreadgroups:threadsPerThreadgroup:"),
                      grid, blk)
      discard objc.msgSend(encoder, objc.`$$`("endEncoding"))
      discard objc.msgSend(cmdBuf, objc.`$$`("commit"))
      discard objc.msgSend(cmdBuf, objc.`$$`("waitUntilCompleted"))

      let status = objc.msgSendUInt(cmdBuf, objc.`$$`("status"))
      if status != MTLCommandBufferStatusCompleted:
        var detail = "no NSError object provided"
        let err = objc.msgSend(cmdBuf, objc.`$$`("error"))
        if not objc.isNil(err):
          detail = objc.nsStringToNimString(objc.msgSend(err, objc.`$$`("localizedDescription")))
        failLoud("Metal run: command buffer failed (status " & $status & " [" & $MTLCommandBufferStatusError & "=error]): " & detail)

# ═════════════════════════════════════════════════
# ▸ Non-macOS entry point
# ═════════════════════════════════════════════════

else:
  proc newMetalEngine(): MetalEngine =
    failLoud("bkMetal requires macOS")
