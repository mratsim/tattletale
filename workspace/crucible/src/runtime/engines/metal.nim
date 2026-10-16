# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
import std/[strutils, tables]

import ../../codegen/ir/gpu_types
import ../launch_common
import workspace/crucible/src/abis/objc_abi as objc
import ../exec/metal_runtime

export metal_runtime

when defined(macosx):
  import ../launch_kernel
# ═════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════
type
  MetalDeviceCtx = object
    ## RAII value wrapper: `=destroy` fires when the engine ref dies.
    ctx: MetalCtx

  MetalCache = object
    ## Two per-engine caches:
    ## - library, keyed by source, replaced by re-ingest
    ## - pipeline states, keyed by kernel name, cleared by re-ingest
    library: objc.ID
    psos: Table[string, objc.ID]

  MetalEngine* = ref object
    ## A Metal engine, build with `bkMetal.init()`: runtime-compiled MSL kernels
    ## on one GPU. All Objective-C resources release when the engine dies.
    source: string
    ctx: MetalDeviceCtx
    cache: MetalCache   # after ctx: released before ctx shutdown

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

  proc `=destroy`(ctx: var MetalDeviceCtx) =
    objc.withMemPool:
      objc.release(ctx.ctx.queue)
      objc.release(ctx.ctx.device)

  proc `=destroy`(cache: var MetalCache) =
    objc.withMemPool:
      objc.release(cache.library)
      for pso in cache.psos.values:
        objc.release(pso)
      cache.psos.clear()

  proc newMetalEngine(): MetalEngine =
    objc.withMemPool:
      result = MetalEngine(
        ctx: MetalDeviceCtx(ctx: newMetalCtx()),
        cache: MetalCache(psos: initTable[string, objc.ID]())
      )

  # ─────────────────────────────────────────────────────────────────────────
  # ▸ PUBLIC API
  # ─────────────────────────────────────────────────────────────────────────

  proc ingest*(engine: MetalEngine, source: string) =
    ## Store the MSL source and compile it into a library. Calling ingest again
    ## replaces the previous artifact and invalidates both cache levels.
    objc.withMemPool:
      let opts = compileOptions()
      let library = compileLibrary(engine.ctx.ctx.device, source, opts)
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
    ## Returns the MSL kernel source.
    engine.source

  proc deviceName*(engine: MetalEngine): string =
    objc.withMemPool:
      result = objc.nsStringToNimString(objc.msgSend(engine.ctx.ctx.device, objc.`$$`("name")))

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
    result = objc.msgSend(engine.ctx.ctx.device, objc.`$$`("newBufferWithLength:options:"),
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


  proc launchKernel*(engine: MetalEngine, kernel: string, output: auto, args: tuple, cfg = LaunchConfig()) =
    ## Runs `kernel` with the launch values bound whole, bridges and direct runs alike.
  ##   engine.launchKernel<<1024, 256>>("saxpy", outDev, (xDev, yDev, 2.0'f32))
    objc.withMemPool:
      # blk is dispatch-time, so run validates the launch geometry.
      if cfg.grid.x < 1 or cfg.grid.y < 1 or cfg.grid.z < 1:
        failLoud("Metal run: grid must be ≥ 1 per axis, got " &
                 $cfg.grid.x & "x" & $cfg.grid.y & "x" & $cfg.grid.z)
      if cfg.blk.x < 1 or cfg.blk.y < 1 or cfg.blk.z < 1:
        failLoud("Metal run: blk must be ≥ 1 per axis, got " &
                 $cfg.blk.x & "x" & $cfg.blk.y & "x" & $cfg.blk.z)
      # Per-axis cap before the product. Three ≤ 1024 axes cannot overflow,
      # so a wrapped product cannot pass the 1024-thread guard.
      if cfg.blk.x > 1024 or cfg.blk.y > 1024 or cfg.blk.z > 1024:
        failLoud("Metal run: blk " & $cfg.blk.x & "x" & $cfg.blk.y & "x" &
                 $cfg.blk.z & " has an axis above 1024, exceeds the Apple " &
                 "Silicon maximum of 1024 threads per threadgroup")
      let threadsPerThreadgroup = cfg.blk.x * cfg.blk.y * cfg.blk.z
      if threadsPerThreadgroup > 1024:
        failLoud("Metal run: blk " & $cfg.blk.x & "x" & $cfg.blk.y & "x" &
                 $cfg.blk.z & " = " & $threadsPerThreadgroup &
                 " threads per threadgroup, exceeds the Apple Silicon " &
                 "maximum of 1024")

      var pso: objc.ID
      if engine.cache.psos.hasKey(kernel):
        pso = engine.cache.psos[kernel]
      else:
        pso = compilePipelineState(engine.ctx.ctx.device, engine.cache.library, kernel)
        engine.cache.psos[kernel] = pso

      let cmdBuf = objc.msgSend(engine.ctx.ctx.queue, objc.`$$`("commandBuffer"))
      let encoder = objc.msgSend(cmdBuf, objc.`$$`("computeCommandEncoder"))
      discard objc.msgSend(encoder, objc.`$$`("setComputePipelineState:"), pso)
      # deliver the launch values whole, positional
      type SinkCtx = object
        encoder: objc.ID
      proc sinkArg(ctx: var SinkCtx, isBuffer: bool, h: objc.ID, idx: int, p: pointer, len: int) {.nimcall.} =
        if isBuffer:
          discard objc.msgSend(ctx.encoder, objc.`$$`("setBuffer:offset:atIndex:"),
                          h, objc.NSUInteger(0), objc.NSUInteger(idx))
        else:
          discard objc.msgSend(ctx.encoder, objc.`$$`("setBytes:length:atIndex:"),
                          p, objc.NSUInteger(len), objc.NSUInteger(idx))
      var dctx = SinkCtx(encoder: encoder)
      bindArgs(dctx, bkMetal, output, args, true, sinkArg)
      if cfg.sharedMem > 0:
        discard objc.msgSend(encoder, objc.`$$`("setThreadgroupMemoryLength:atIndex:"),
                        objc.NSUInteger(cfg.sharedMem), objc.NSUInteger(0))
      let grid = objc.MTLSize(width: objc.NSUInteger(cfg.grid.x),
                         height: objc.NSUInteger(cfg.grid.y),
                         depth: objc.NSUInteger(cfg.grid.z))
      let blk = objc.MTLSize(width: objc.NSUInteger(cfg.blk.x),
                        height: objc.NSUInteger(cfg.blk.y),
                        depth: objc.NSUInteger(cfg.blk.z))
      discard objc.msgSend(encoder, objc.`$$`("dispatchThreadgroups:threadsPerThreadgroup:"),
                      grid, blk)
      discard objc.msgSend(encoder, objc.`$$`("endEncoding"))
      discard objc.msgSend(cmdBuf, objc.`$$`("commit"))
      discard objc.msgSend(cmdBuf, objc.`$$`("waitUntilCompleted"))

      # The device can reject a dispatch after commit (oversized grid, bad binding).
      # Without a status check, the stale output would read back as success.
      let status = objc.msgSendUInt(cmdBuf, objc.`$$`("status"))
      if status != MTLCommandBufferStatusCompleted:
        var detail = "no NSError object provided"
        let err = objc.msgSend(cmdBuf, objc.`$$`("error"))
        if not objc.isNil(err):
          detail = objc.nsStringToNimString(objc.msgSend(err, objc.`$$`("localizedDescription")))
        failLoud("Metal run: command buffer failed (status " & $status &
                 " [" & $MTLCommandBufferStatusError & "=error]): " & detail)

# ═════════════════════════════════════════════════
# ▸ Non-macOS entry point
# ═════════════════════════════════════════════════

else:
  proc newMetalEngine(): MetalEngine =
    quit("bkMetal requires macOS")
