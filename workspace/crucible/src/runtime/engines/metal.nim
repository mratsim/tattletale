# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
{.experimental: "codeReordering".}
import std/[strutils, tables]

import ../../codegen/ir/gpu_types
import ./engines_common
import ../errors
import ../launch/launch_serialization
import workspace/crucible/src/abis/objc_abi as objc
import ../exec/metal_runtime

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

template backendKind*(_: typedesc[MetalEngine]): BackendKind = bkMetal

# ═════════════════════════════════════════════════
# ▸ macOS-only engine implementation
# ═════════════════════════════════════════════════

when defined(macosx):
  proc allocDeviceImpl(engine: MetalEngine, n: int): objc.ID
  proc copyToDeviceFromHostImpl(engine: MetalEngine, dst: objc.ID, src: pointer, n: int)
  proc copyToHostFromDeviceImpl(engine: MetalEngine, dst: pointer, src: objc.ID, n: int)

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

  proc newMetalEngine*(): MetalEngine =
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

  proc getArtifact*(engine: MetalEngine): string {.inline.} =
    engine.source

  proc deviceName*(engine: MetalEngine): string =
    objc.withMemPool:
      result = objc.nsStringToNimString(objc.msgSend(engine.ctx.device, objc.`$$`("name")))

  proc allocDevice*[T](engine: MetalEngine, _: typedesc[T], len: int): DeviceMem[objc.ID, T] {.inline.} =
    ## Allocates len elements of T on the engine
    DeviceMem[objc.ID, T](mem: engine.allocDeviceImpl(len * sizeof(T)))

  proc copyToDeviceFromHost*[T](engine: MetalEngine, dst: DeviceMem[objc.ID, T], src: ptr T or ptr UncheckedArray[T], n: int) {.inline.} =
    ## Copies n elements from host to device
    engine.copyToDeviceFromHostImpl(dst.mem, src, n * sizeof(T))

  proc copyToHostFromDevice*[T](engine: MetalEngine, dst: ptr T or ptr UncheckedArray[T], src: DeviceMem[objc.ID, T], n: int) {.inline.} =
    ## Copies n elements from device to host
    engine.copyToHostFromDeviceImpl(dst, src.mem, n * sizeof(T))

  # ─────────────────────────────────────────────────────────────────────────
  # ▸ Launch serialization
  # ─────────────────────────────────────────────────────────────────────────

  type SerializationCtx = object
    encoder: objc.ID

  proc store(ctx: var SerializationCtx, v: objc.ID, argIdx: int) =
    discard objc.msgSend(ctx.encoder, objc.`$$`("setBuffer:offset:atIndex:"),
                    v, objc.NSUInteger(0), objc.NSUInteger(argIdx))

  proc store[T](ctx: var SerializationCtx, v: T, argIdx: int) =
    when T is ptr|pointer:
      {.error: "Pointer kernel args are not supported, pass the DeviceMem handle from allocDevice, got " & $T.}
    const size = deviceLayout(T, bkMetal).size
    var blob: array[size, byte]
    var off = 0
    blob.packFrom(off, v, bkMetal)
    discard objc.msgSend(ctx.encoder, objc.`$$`("setBytes:length:atIndex:"),
                    blob[0].addr, objc.NSUInteger(size), objc.NSUInteger(argIdx))

  # ─────────────────────────────────────────────────────────────────────────
  # ▸ PRIVATE constants
  # ─────────────────────────────────────────────────────────────────────────

  const
    MTLCommandBufferStatusCompleted = 4
    MTLCommandBufferStatusError = 5

  # ─────────────────────────────────────────────────────────────────────────
  # ▸ PRIVATE run path
  # ─────────────────────────────────────────────────────────────────────────

  proc bufferContents(h: objc.ID): pointer =
    result = cast[pointer](objc.msgSend(h, objc.`$$`("contents")))
    if result == nil:
      failLoud("contents() returned nil for a shared buffer")

  proc allocDeviceImpl(engine: MetalEngine, n: int): objc.ID =
    if n <= 0:
      failLoud("allocDevice: size must be positive, got " & $n)
    result = objc.msgSend(engine.ctx.device, objc.`$$`("newBufferWithLength:options:"),
                          objc.NSUInteger(n), SharedUntrackedOptions)
    if objc.isNil(result):
      failLoud("allocDevice: newBufferWithLength:options: returned nil (length " & $n & ")")

  proc copyToDeviceFromHostImpl(engine: MetalEngine, dst: objc.ID, src: pointer, n: int) =
    if n <= 0:
      failLoud("copyToDeviceFromHost: n must be positive, got " & $n)
    copyMem(bufferContents(dst), src, n)

  proc copyToHostFromDeviceImpl(engine: MetalEngine, dst: pointer, src: objc.ID, n: int) =
    if n <= 0:
      failLoud("copyToHostFromDevice: n must be positive, got " & $n)
    copyMem(dst, bufferContents(src), n)

  proc launchKernelImpl*(engine: MetalEngine, kernel: string, output: auto, args: var tuple, cfg: LaunchConfig) =
    ## Runs `kernel` with the launch values bound whole, bridges and direct runs alike.
    ##   engine.launchKernel<<(1024, 256)>>("saxpy", outDev, (xDev, yDev, 2.0'f32))
    objc.withMemPool:
      if cfg.grid.x < 1 or cfg.grid.y < 1 or cfg.grid.z < 1:
        failLoud("Metal run: cfg.grid must be ≥ 1 per axis, got " & $cfg.grid.x & "x" & $cfg.grid.y & "x" & $cfg.grid.z)
      if cfg.blk.x < 1 or cfg.blk.y < 1 or cfg.blk.z < 1:
        failLoud("Metal run: cfg.blk must be ≥ 1 per axis, got " & $cfg.blk.x & "x" & $cfg.blk.y & "x" & $cfg.blk.z)
      if cfg.blk.x * cfg.blk.y * cfg.blk.z > 1024:
        failLoud("Metal run: cfg.blk " & $cfg.blk.x & "x" & $cfg.blk.y & "x" & $cfg.blk.z & " exceeds the 1024 threads-per-threadgroup max")

      var pso: objc.ID
      if engine.cache.psos.hasKey(kernel):
        pso = engine.cache.psos[kernel]
      else:
        pso = compilePipelineState(engine.ctx.device, engine.cache.library, kernel)
        engine.cache.psos[kernel] = pso

      let cmdBuf = objc.msgSend(engine.ctx.queue, objc.`$$`("commandBuffer"))
      let encoder = objc.msgSend(cmdBuf, objc.`$$`("computeCommandEncoder"))
      discard objc.msgSend(encoder, objc.`$$`("setComputePipelineState:"), pso)
      if cfg.sharedMem > 0:
        # Metal requires a 16-byte multiple of threadgroup memory
        let tgMem = roundStepUp(cfg.sharedMem, 16)
        discard objc.msgSend(encoder, objc.`$$`("setThreadgroupMemoryLength:atIndex:"),
                        objc.NSUInteger(tgMem), objc.NSUInteger(0))

      var dctx = SerializationCtx(encoder: encoder)
      bindArgs(dctx, output, args, store, objc.ID, bkMetal)
      let gridSize = objc.MTLSize(width: objc.NSUInteger(cfg.grid.x),
                         height: objc.NSUInteger(cfg.grid.y),
                         depth: objc.NSUInteger(cfg.grid.z))
      let blkSize = objc.MTLSize(width: objc.NSUInteger(cfg.blk.x),
                        height: objc.NSUInteger(cfg.blk.y),
                        depth: objc.NSUInteger(cfg.blk.z))
      discard objc.msgSend(encoder, objc.`$$`("dispatchThreadgroups:threadsPerThreadgroup:"),
                      gridSize, blkSize)
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
  proc newMetalEngine*(): MetalEngine =
    failLoud("bkMetal requires macOS")

