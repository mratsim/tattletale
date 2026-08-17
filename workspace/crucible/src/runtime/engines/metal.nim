# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## MetalEngine: Metal (MSL) runtime compilation and execution.
##
## ingest compiles the MSL source into a library
## (level-1 cache, keyed by the source it was built from).
## run gets or builds the compute pipeline state per (kernel, argSizes)
## (level-2 cache), then dispatches `dispatchThreadgroups(grid, blk)`.
## blk is dispatch-time, so run validates it against the Apple Silicon 1024-thread limit.
## Scalars (ArgBlob size < 0) pack into one shared constant buffer at 16-byte slots,
## bound per index via `setBuffer:offset:atIndex:`.
## The output reads back directly from `contents()` after `waitUntilCompleted`.
## No staging buffer exists in the code path.
##
## The device context (`initMetal`) lives in `exec/metal_runtime`
## (imported below). This module owns the engine and its RAII `=destroy` hook.
##
## The engine implementation is `when defined(macosx)`-guarded. Only macOS
## provides the Objective-C surface. On other platforms, `newMetalEngine`
## still exists so `bkMetal.init()` compiles everywhere, and the engine
## quits loudly when constructed.
##
## Structure: platform-neutral types at the top, then the macOS-only
## implementation. Inside the guard, private helpers and constants
## precede the public API they serve. Nim's `codeReordering` does not
## see through `when` blocks, so declaration order is what the compiler
## sees.

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
    ## Two-level cache: the compiled library (level 1, keyed by the source it was built from)
    ## and per-(kernel, argSizes) compute pipeline states (level 2).
    ## Re-ingest replaces the library and clears the pipeline states.
    library: objc.ID
    psos: Table[(string, seq[int]), objc.ID]

  MetalEngine* = ref object
    ## `source` plus the RAII value fields `ctx` (device + queue) and `cache` (library + PSOs),
    ## which release their +1 Objective-C objects when the engine ref is destroyed.
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
    ## Releases the library and cached pipeline states while the device is still alive:
    ## this field is declared after `ctx`,
    ## so reverse-order field destruction runs it before `ctx`'s `=destroy` releases the device.
    ## Runs inside a pool for the same reason as `MetalDeviceCtx`.
    objc.withMemPool:
      objc.release(cache.library)
      for pso in cache.psos.values:
        objc.release(pso)
      cache.psos.clear()

  proc newMetalEngine*(): MetalEngine =
    ## Factory reached by engines.nim via `import {.all.}`.
    objc.withMemPool:
      result = MetalEngine(
        ctx: MetalDeviceCtx(ctx: initMetal()),
        cache: MetalCache(psos: initTable[(string, seq[int]), objc.ID]())
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
    ## The MSL kernel source.
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
    ## Get-or-build the pipeline state, then encode and dispatch.
    ## The output is the kernel's first parameter (binding 0), then the input args in order:
    ##   device buffers for size ≥ 0,
    ##   constant-buffer slots for size < 0.
    ## The output's current bytes are uploaded before launch (in-place β·C)
    ## and read back after waitUntilCompleted.
    ## A grid beyond the device's maxTotalThreadgroupsPerGrid is undefined:
    ## Metal may reject the dispatch after commit or complete it with stale
    ## output, and the status check cannot catch that geometry class.
    ## Callers must validate grid against the device limit.
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

      # Compute pipeline state, cached per (kernel, argSizes). Created once per shape
      # and reused by every run, per the ingest-once/run-many contract.
      var argSizes = newSeq[int](blobs.len)
      for i in 0 ..< blobs.len:
        argSizes[i] = blobs[i].size   # signed: negative = scalar
      let cacheKey = (kernel, argSizes)
      var pso: objc.ID
      if engine.cache.psos.hasKey(cacheKey):
        pso = engine.cache.psos[cacheKey]
      else:
        pso = compilePipelineState(engine.ctx.ctx.device,
                                   engine.cache.library, kernel)
        engine.cache.psos[cacheKey] = pso

      # Buffers: output + size ≥ 0 inputs (shared storage, memcpy in).
      # The size < 0 scalars pack into one shared constant buffer at 16-byte slots.
      let outSize = output.size
      var outBuf = allocBuffer(engine.ctx.ctx.device, outSize)
      defer:
        releaseBuffer(outBuf)
      if outSize > 0:
        copyMem(outBuf.data, output.data, outSize)

      var inputBuffers = newSeq[MetalBuffer](blobs.len)
      var scalarCount = 0
      for i in 0 ..< blobs.len:
        if blobs[i].size >= 0:
          inputBuffers[i] = allocBuffer(engine.ctx.ctx.device, blobs[i].size)
          copyMem(inputBuffers[i].data, blobs[i].data, blobs[i].size)
        else:
          inc scalarCount
      defer:
        for b in mitems(inputBuffers):
          releaseBuffer(b)

      var scalarBuf: MetalBuffer
      if scalarCount > 0:
        scalarBuf = allocBuffer(engine.ctx.ctx.device, scalarCount * ScalarSlotStride)
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
      defer:
        releaseBuffer(scalarBuf)

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
                          inputBuffers[i].buffer, objc.NSUInteger(0), objc.NSUInteger(i + 1))
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

      # Direct readback: contents() is CPU-visible after waitUntilCompleted on shared-storage buffers.
      # No staging buffer exists here.
      if outSize > 0:
        copyMem(output.data, outBuf.data, outSize)

# ═════════════════════════════════════════════════
# ▸ Non-macOS entry point
# ═════════════════════════════════════════════════

else:
  proc newMetalEngine*(): MetalEngine =
    ## Exists so `bkMetal.init()` compiles on every platform and fails loudly
    ## at construction instead of at link time.
    quit("bkMetal requires macOS")
