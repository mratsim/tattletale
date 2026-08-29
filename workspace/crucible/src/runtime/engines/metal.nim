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
## run gets or builds the compute pipeline state per kernel name
## (level-2 cache), then dispatches `dispatchThreadgroups(grid, blk)`.
## blk is dispatch-time, so run validates it against the Apple Silicon 1024-thread limit.
## Scalars (ArgBlob size < 0) pack into one shared constant buffer at 16-byte slots,
## bound per index via `setBuffer:offset:atIndex:`.
## No-copy binding: page-aligned host memory with a byte length that is a multiple
## of the page size yields a wrapper over the caller's bytes, cached
## per (pointer, size), so binding allocates nothing and copies nothing.
## Any other length gets a shared buffer, copied in before launch and copied back
## after the wait.
## A wrap the driver refuses is a hard error rather than a slower run, there is no
## fallback to the copied path.
## Output reads back in place for a no-copy binding, or through `contents()` once
## `waitUntilCompleted` returns for an allocated buffer. No staging buffer exists
## in the code path.
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
    ## Four per-engine caches: the compiled library, the compute pipeline states,
    ## the no-copy wrapper table, the packed-scalar buffer.
    ## The library is keyed by the source it was built from, the pipeline states
    ## by kernel name, the wrapper table by (host pointer, byte length). Re-ingest
    ## replaces the library and clears the pipeline states.
    ## A wrapper hit is revalidated against the VM map before reuse, see
    ## `cachedWrap`. `scalarBuf` holds the packed scalar args, contents memcpy'd
    ## in per run, capacity grown on demand.
    library: objc.ID
    psos: Table[string, objc.ID]
    bufs: Table[(pointer, int), objc.ID]
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

  proc dropBufferCache(cache: var MetalCache) =
    ## Release and empty all cached wrappers. Sound only between runs: each run ends
    ## with waitUntilCompleted, so no in-flight command still holds a binding.
    ## Release drops GPU mappings, never host pages.
    for buf in cache.bufs.values:
      var b = MetalBuffer(buffer: buf, data: nil)
      objc.release(b.buffer)
    cache.bufs.clear()

  proc `=destroy`(cache: var MetalCache) =
    ## Frees the library, pipeline states, wrapper cache and scalar buffer
    ## while the device is still alive: declared after `ctx`, so reverse-order
    ## field destruction runs it first. Pooled, same reason as `MetalDeviceCtx`.
    objc.withMemPool:
      objc.release(cache.library)
      for pso in cache.psos.values:
        objc.release(pso)
      cache.psos.clear()
      dropBufferCache(cache)
      var sb = cache.scalarBuf
      releaseBuffer(sb)
      cache.scalarBuf = MetalBuffer(buffer: objc.ID(nil), data: nil)
      cache.scalarCap = 0

  proc newMetalEngine*(): MetalEngine =
    ## Factory reached by engines.nim via `import {.all.}`.
    objc.withMemPool:
      result = MetalEngine(
        ctx: MetalDeviceCtx(ctx: initMetal()),
        cache: MetalCache(psos: initTable[string, objc.ID](),
                          bufs: initTable[(pointer, int), objc.ID]())
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

  const BufCacheMax = 64

  func eligibleNoCopy(data: pointer, size: int): bool {.inline.} =
    ## Page-aligned `data` and a `size` that is a multiple of the host page size:
    ## the two newBufferWithBytesNoCopy requirements. The predicate reports whether
    ## a binding would be legal for the length given, it cannot tell whether
    ## the caller's memory covers that length.
    size > 0 and data != nil and
      (cast[uint](data) mod hostPageSize().uint) == 0'u and
      (size mod hostPageSize()) == 0

  proc cachedWrap(engine: MetalEngine, data: pointer, size: int): MetalBuffer =
    ## No-copy view cache: one MTLBuffer wrapper per (pointer, size), reused
    ## across runs, avoiding the per-dispatch create+release cost.
    ##
    ## Dataflow: caller-owned host memory → no-copy wrapper over shared storage →
    ## GPU reads and writes alias the caller's bytes directly. The cache holds
    ## wrappers only, memory ownership never transfers.
    ##
    ## Liveness, one syscall per hit (`hostRangeLive`), never an allocation:
    ## - memory still mapped under the same (pointer, size): hit and rewrap give
    ##   the same view of the same bytes, return the cached wrapper.
    ## - memory unmapped, or remapped so the old range is only partly live:
    ##   the wrapper no longer describes what the caller named, drop it
    ##   (release, then rewrap).
    ## - memory remapped at the same address with the same byte length, entirely:
    ##   indistinguishable from a hit, the one hole in a VM-map check. See TODO.
    ##
    ## Preconditions: `data` page-aligned, `size` a multiple of the page size and no
    ## larger than the caller's allocation behind it, and that memory outliving
    ## every wrapper on it.
    ## `wrapBufferNoCopy` inherits Apple's nil-deallocator obligation.
    ## Capacity is bounded at run entry (`runImpl`) only. Releasing an entry
    ## the current run already bound would drop its last reference mid-encode.
    ##
    ## TODO: close the same-address remap hole before long-lived or foreign callers
    ## bind through this cache. Tag entries with an allocator epoch or generation
    ## (register/unregister at alloc/free) so a recycled (pointer, size) key misses
    ## the cache and rewraps.
    let key = (data, size)
    if engine.cache.bufs.hasKey(key):
      if hostRangeLive(data, size):
        result = MetalBuffer(buffer: engine.cache.bufs[key], data: data)
        return
      # Unmapped or remapped in pieces: the wrapper no longer describes the memory
      # the caller named, so release it and wrap the current bytes.
      var stale = MetalBuffer(buffer: engine.cache.bufs[key], data: nil)
      objc.release(stale.buffer)
      engine.cache.bufs.del(key)
      when defined(debug):
        echo "[INFO]: metal no-copy cache: extent not mapped, re-wrapping"
    result = wrapBufferNoCopy(engine.ctx.ctx.device, data, size)
    engine.cache.bufs[key] = result.buffer

  # ─────────────────────────────────────────────────────────────────────────
  # ▸ PRIVATE run path
  # ─────────────────────────────────────────────────────────────────────────

  proc runImpl(engine: MetalEngine, kernel: string, output: ArgBlob,
               blobs: seq[ArgBlob], cfg: LaunchConfig) =
    ## Get-or-build the pipeline state, then encode and dispatch.
    ## The output is the kernel's first parameter (binding 0), then the input args in order:
    ##   device buffers for size ≥ 0,
    ##   constant-buffer slots for size < 0.
    ## Page-aligned memory with a byte length that is a page multiple binds no-copy,
    ## and the GPU aliases the caller's bytes, see `cachedWrap`. Any other length
    ## allocates a shared buffer, memcpy in before launch (in-place β·C) and memcpy
    ## back after waitUntilCompleted.
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

      # Compute pipeline state, cached per kernel name. The PSO derives solely
      # from the kernel function, so host buffer sizes never affect it; keying
      # on sizes would rebuild pipelines per shape and grow the table unboundedly.
      var pso: objc.ID
      if engine.cache.psos.hasKey(kernel):
        pso = engine.cache.psos[kernel]
      else:
        pso = compilePipelineState(engine.ctx.ctx.device,
                                   engine.cache.library, kernel)
        engine.cache.psos[kernel] = pso

      # Buffers: the output plus every input with size ≥ 0. Eligible ones wrap host
      # memory no-copy, so the kernel reads and writes the bytes the caller passed,
      # in place. Any other length allocates a fresh shared buffer and copies
      # in/out.
      # Per-call buffers (fresh allocations) release at scope exit, and no-copy
      # wrappers stay in the cache.
      # Wrapper-cache capacity, bounded between runs. At run entry no wrapper bound
      # by this run exists yet, so eviction cannot drop one the encoder still
      # references. One run may take the table past the bound through its own
      # distinct (pointer, size) keys.
      if engine.cache.bufs.len >= BufCacheMax:
        dropBufferCache(engine.cache)
      let outSize = output.size
      var outBuf: MetalBuffer
      var outWrapped = false
      if eligibleNoCopy(output.data, outSize):
        outBuf = cachedWrap(engine, output.data, outSize)
        outWrapped = true
      else:
        outBuf = allocBuffer(engine.ctx.ctx.device, outSize)
        if outSize > 0:
          copyMem(outBuf.data, output.data, outSize)
      # Body-level defer: a defer inside the else block would fire at the end
      # of that block, before the encode.
      defer:
        if not outWrapped:
          releaseBuffer(outBuf)

      var inputBuffers = newSeq[MetalBuffer](blobs.len)
      var perCallBuffers = newSeq[MetalBuffer]()
      var scalarCount = 0
      for i in 0 ..< blobs.len:
        if blobs[i].size >= 0:
          if eligibleNoCopy(blobs[i].data, blobs[i].size):
            inputBuffers[i] = cachedWrap(engine, blobs[i].data, blobs[i].size)
          else:
            inputBuffers[i] = allocBuffer(engine.ctx.ctx.device, blobs[i].size)
            perCallBuffers.add inputBuffers[i]
            copyMem(inputBuffers[i].data, blobs[i].data, blobs[i].size)
        else:
          inc scalarCount
      defer:
        for b in mitems(perCallBuffers):
          releaseBuffer(b)

      # The size < 0 scalars pack into one persistent shared constant buffer
      # at 16-byte slots (capacity grown on demand, contents memcpy'd in per run).
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

      # Readback: a no-copy output already holds its result in the memory the caller
      # passed, because for shared-storage buffers contents() is CPU-visible once
      # waitUntilCompleted returns. Allocated buffers copy back.
      # No staging buffer exists here.
      if outSize > 0 and not outWrapped:
        copyMem(output.data, outBuf.data, outSize)

# ═════════════════════════════════════════════════
# ▸ Non-macOS entry point
# ═════════════════════════════════════════════════

else:
  proc newMetalEngine*(): MetalEngine =
    ## Exists so `bkMetal.init()` compiles on every platform and fails loudly
    ## at construction instead of at link time.
    quit("bkMetal requires macOS")
