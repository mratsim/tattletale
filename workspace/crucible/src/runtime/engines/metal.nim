# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal engine: runtime MSL compilation and launch.
##
## | API                        | contract                                                                                                                 |
## | -------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
## | ingest                     | compiles the MSL source into a library, cached by source                                                                 |
## | run                        | pipeline state per kernel name, cached by name, then dispatches `dispatchThreadgroups(grid, blk)`                        |
## | buffers                    | shared allocation, memcpy in before launch, memcpy back after `waitUntilCompleted`                                       |
## | pointers                   | passed as-is, no copy and no probing, the caller is responsible for a valid target, on-device or unified-memory-resident |
## | scalars (ArgBlob size < 0) | packed into one shared constant buffer, 16-byte slots, bound via `setBuffer:offset:atIndex:`                             |
## | blk                        | validated against the Apple Silicon 1024-thread limit at dispatch                                                        |
##
## macOS only:
## - the implementation is `when defined(macosx)`-guarded
## - on other platforms `newMetalEngine` exists and quits at construction
##
## The device context (`initMetal`) lives in `exec/metal_runtime`.

import std/[strutils, tables]

import workspace/crucible/src/abis/objc_abi as objc
import ../exec/metal_runtime

export metal_runtime

when defined(macosx):
  import ./arg_blobs
  import ../chevrons
# ═════════════════════════════════════════════════
# ▸ Types
# ═════════════════════════════════════════════════
type
  MetalDeviceCtx = object
    ## RAII value wrapper: `=destroy` fires when the engine ref dies.
    ctx: MetalCtx

  MetalCache = object
    ## Four per-engine caches: the compiled library, the compute pipeline states,
    ## the no-copy wrapper table, the packed-scalar buffer.
    ## Three per-engine caches:
    ## - library, keyed by source, replaced by re-ingest
    ## - pipeline states, keyed by kernel name, cleared by re-ingest
    ## - scalarBuf, packed scalar args, contents memcpy'd in per run,
    ##   capacity grown on demand
    library: objc.ID
    psos: Table[string, objc.ID]
    scalarBuf: MetalBuffer
    scalarCap: int

  MetalEngine* = ref object
    ## `source` plus the RAII value fields `ctx` (device + queue) and `cache`
    ## (library, pipeline states, wrapper table, scalar buffer), which release
    ## their +1 Objective-C objects when the engine ref is destroyed.
    source: string
    ctx: MetalDeviceCtx
    cache: MetalCache   # after ctx: released before ctx shutdown

# ═════════════════════════════════════════════════
# ▸ macOS-only engine implementation
# ═════════════════════════════════════════════════

when defined(macosx):
  # ─────────────────────────────────────────────────────────────────────────
  # ▸ PRIVATE constants
  # ─────────────────────────────────────────────────────────────────────────

  const
    ## Constant-buffer slot stride for packed scalar args, in bytes.
    ## Verified on-device: 16-byte alignment on Apple Silicon.
    ScalarSlotStride = 16

  const
    ## `MTLCommandBufferStatus` values (MTLCommandBuffer.h): 4 = completed,
    ## 5 = error. `waitUntilCompleted` leaves the buffer in one of these.
    MTLCommandBufferStatusCompleted = 4
    MTLCommandBufferStatusError = 5

  # ─────────────────────────────────────────────────────────────────────────
  # ▸ Constructors/destructors
  # ─────────────────────────────────────────────────────────────────────────

  proc `=destroy`(ctx: var MetalDeviceCtx) =
    ## Releases the +1 device and command queue (queue first, the reverse of creation order).
    ## The releases run inside a pool: the queue's dealloc autoreleases the device,
    ## and destroy runs outside any public call, so a bare release would trip OBJC_DEBUG_MISSING_POOLS=YES.
    objc.withMemPool:
      objc.release(ctx.ctx.queue)
      objc.release(ctx.ctx.device)

  proc `=destroy`(cache: var MetalCache) =
    objc.withMemPool:
      objc.release(cache.library)
      for pso in cache.psos.values:
        objc.release(pso)
      cache.psos.clear()
      var sb = cache.scalarBuf
      releaseBuffer(sb)
      cache.scalarBuf = MetalBuffer(buffer: objc.ID(nil), data: nil)
      cache.scalarCap = 0

  proc newMetalEngine*(): MetalEngine =
    ## Factory reached by engines.nim via `import {.all.}`.
    objc.withMemPool:
      result = MetalEngine(
        ctx: MetalDeviceCtx(ctx: initMetal()),
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
    ## The Metal device name (e.g. "Apple M4 Max").
    objc.withMemPool:
      result = objc.nsStringToNimString(objc.msgSend(engine.ctx.ctx.device, objc.`$$`("name")))

  # ─────────────────────────────────────────────────────────────────────────
  # ▸ PRIVATE run path
  # ─────────────────────────────────────────────────────────────────────────

  proc runImpl(engine: MetalEngine, kernel: string, output: ArgBlob,
               blobs: seq[ArgBlob], cfg: LaunchConfig) =
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
        pso = compilePipelineState(engine.ctx.ctx.device,
                                   engine.cache.library, kernel)
        engine.cache.psos[kernel] = pso

      let outSize = output.size
      var outBuf = allocBuffer(engine.ctx.ctx.device, outSize)
      if outSize > 0:
        copyMem(outBuf.data, output.data, outSize)
      defer:
        releaseBuffer(outBuf)

      var inputBuffers = newSeq[MetalBuffer](blobs.len)
      var perCallBuffers = newSeq[MetalBuffer]()
      var scalarCount = 0
      for i in 0 ..< blobs.len:
        if blobs[i].size >= 0:
          inputBuffers[i] = allocBuffer(engine.ctx.ctx.device, blobs[i].size)
          perCallBuffers.add inputBuffers[i]
          copyMem(inputBuffers[i].data, blobs[i].data, blobs[i].size)
        else:
          inc scalarCount
      defer:
        for b in mitems(perCallBuffers):
          releaseBuffer(b)

      var scalarBuf: MetalBuffer
      if scalarCount > 0:
        let needed = scalarCount * ScalarSlotStride
        if needed > engine.cache.scalarCap:
          var old = engine.cache.scalarBuf
          releaseBuffer(old)
          engine.cache.scalarBuf = allocBuffer(engine.ctx.ctx.device, needed)
          engine.cache.scalarCap = needed
        scalarBuf = engine.cache.scalarBuf
        let dst = cast[ptr UncheckedArray[byte]](scalarBuf.data)
        var slot = 0
        for i in 0 ..< blobs.len:
          if blobs[i].size < 0:
            let sz = -blobs[i].size
            if sz > ScalarSlotStride:
              failLoud("Metal run: scalar arg " & $i & " is " & $sz &
                       " bytes, exceeds the " & $ScalarSlotStride &
                       "-byte constant-buffer slot")
            copyMem(addr dst[slot * ScalarSlotStride], blobs[i].data, sz)
            inc slot

      # Encode: pipeline, buffers, dispatch, commit, wait.
      let cmdBuf = objc.msgSend(engine.ctx.ctx.queue, objc.`$$`("commandBuffer"))
      let encoder = objc.msgSend(cmdBuf, objc.`$$`("computeCommandEncoder"))
      discard objc.msgSend(encoder, objc.`$$`("setComputePipelineState:"), pso)
      discard objc.msgSend(encoder, objc.`$$`("setBuffer:offset:atIndex:"), outBuf.buffer,
                      objc.NSUInteger(0), objc.NSUInteger(0))
      var slot = 0
      for i in 0 ..< blobs.len:
        if blobs[i].size >= 0:
          discard objc.msgSend(encoder, objc.`$$`("setBuffer:offset:atIndex:"),
                          inputBuffers[i].buffer, objc.NSUInteger(blobs[i].off),
                          objc.NSUInteger(i + 1))
        else:
          discard objc.msgSend(encoder, objc.`$$`("setBuffer:offset:atIndex:"),
                          scalarBuf.buffer, objc.NSUInteger(slot * ScalarSlotStride), objc.NSUInteger(i + 1))
          inc slot
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

      if outSize > 0:
        copyMem(output.data, outBuf.data, outSize)

# ═════════════════════════════════════════════════
# ▸ Non-macOS entry point
# ═════════════════════════════════════════════════

else:
  proc newMetalEngine*(): MetalEngine =
    quit("bkMetal requires macOS")
