# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/tables
import workspace/crucible/src/abis/objc_abi as objc
import workspace/crucible/src/crucible_codegen
import workspace/crucible/src/runtime/backends/metal_runtime
import workspace/crucible/src/runtime/backends/runtime_utils
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/launch/launch_serialization

export engines_common.LaunchConfig, engines_common.dim3

# ############################################################
#
#                   Metal batch submission
#
# ############################################################

type
  SlotKind = enum
    skBuffer
    skBytes

  Slot = object
    case kind: SlotKind
    of skBuffer:
      buffer: int
    of skBytes:
      off, size: int

  BatchEntry = object
    pipelineState: objc.ID
    fleet, squadron: Dim3
    borrowedBuffers: seq[objc.ID]
    scalars: seq[byte]
    slots: seq[Slot]

  MetalBatchObj = object
    device, queue: objc.ID
    entries: seq[BatchEntry]
    pipelines: Table[string, objc.ID]
    recording: objc.ID
    constants: objc.ID

  MetalBatch* = ref MetalBatchObj

const
  MTLIndirectCommandTypeConcurrentDispatch = objc.NSUInteger(1 shl 5)
  MTLCommandBufferStatusCompleted = 4
  ResourceUsageReadWrite = objc.NSUInteger(3)
  MaxThreadsPerThreadgroup = 1024
    ## Apple Silicon caps one threadgroup at 1024 threads
  MaxKernelBufferBinds = 31
    ## Metal caps one dispatch at 31 bound buffers

proc `=destroy`*(batch: var MetalBatchObj) =
  ## Releases the batch's own objects, never the borrowed handles.
  if not objc.isNil(batch.recording):
    objc.release(batch.recording)
  for pso in batch.pipelines.values:
    objc.release(pso)
  if not objc.isNil(batch.constants):
    objc.release(batch.constants)
  `=destroy`(batch.entries)
  `=destroy`(batch.pipelines)

when defined(macosx):

  proc msgSendUseResource(self: objc.ID, op: objc.SEL, resource: objc.ID, usage: objc.NSUInteger): objc.ID {.importc: "objc_msgSend", dynlib: "libobjc.A.dylib".}

  proc compileIndirectPipelineState(device, library: objc.ID, entrypoint: string): objc.ID =
    let fn = objc.msgSend(library, objc.`$$`("newFunctionWithName:"),
                          objc.nsStringFromNimString(entrypoint))
    if objc.isNil(fn):
      raiseAssert("newFunctionWithName(" & entrypoint & ") returned nil")
    defer: objc.release(fn)
    let desc = objc.msgSend(objc.msgSend(objc.ID(objc.getClass("MTLComputePipelineDescriptor")), objc.`$$`("alloc")), objc.`$$`("init"))
    defer: objc.release(desc)
    discard objc.msgSend(desc, objc.`$$`("setComputeFunction:"), fn)
    discard objc.msgSend(desc, objc.`$$`("setSupportIndirectCommandBuffers:"), objc.BOOL(1))
    var psoError: objc.ID = objc.ID(nil)
    result = objc.msgSend(device, objc.`$$`("newComputePipelineStateWithDescriptor:options:reflection:error:"),
                          desc, objc.NSUInteger(0), nil, addr psoError)
    if objc.isNil(result):
      var detail = "no NSError object provided"
      if not objc.isNil(psoError):
        detail = objc.nsStringToNimString(objc.msgSend(psoError, objc.`$$`("localizedDescription")))
      raiseAssert("indirect compute pipeline state creation failed for " & entrypoint & ": " & detail)

  proc storeEntryArg(entry: var BatchEntry, v: auto, _: int) =
    when typeof(v) is objc.ID:
      entry.borrowedBuffers.add v
      entry.slots.add Slot(kind: skBuffer, buffer: entry.borrowedBuffers.high)
    else:
      when typeof(v) is ptr|pointer:
        {.error: "Pointer device function args are not supported, pass the DeviceMem handle from allocDevice, got " & $typeof(v).}
      const layout = computeDeviceLayout(typeof(v), cgMetal)
      let off = round_multiple_up(entry.scalars.len, 16)
      entry.scalars.setLen(off + layout.size)
      var w = off
      entry.scalars.packFrom(w, v, cgMetal)
      entry.slots.add Slot(kind: skBytes, off: off, size: layout.size)

  proc newRecording(device: objc.ID, maxCommands: int): objc.ID =
    let desc = objc.msgSend(objc.msgSend(objc.ID(objc.getClass("MTLIndirectCommandBufferDescriptor")), objc.`$$`("alloc")), objc.`$$`("init"))
    defer: objc.release(desc)
    discard objc.msgSend(desc, objc.`$$`("setCommandTypes:"), MTLIndirectCommandTypeConcurrentDispatch)
    discard objc.msgSend(desc, objc.`$$`("setInheritPipelineState:"), objc.BOOL(0))
    discard objc.msgSend(desc, objc.`$$`("setMaxKernelBufferBindCount:"), objc.NSUInteger(MaxKernelBufferBinds))
    result = objc.msgSend(device, objc.`$$`("newIndirectCommandBufferWithDescriptor:maxCommandCount:options:"),
                          desc, objc.NSUInteger(maxCommands), objc.NSUInteger(0))
    if objc.isNil(result):
      raiseAssert("newIndirectCommandBufferWithDescriptor returned nil, indirect command buffers are unsupported on this device")

  proc newMetalBatch*(device, queue: objc.ID): MetalBatch =
    ## Borrowed: the device, the queue and every bound buffer.
    ## Every bound buffer outlives the batch.
    MetalBatch(device: device, queue: queue)

  proc recordGlobalFn*(batch: MetalBatch, library: objc.ID, entrypoint: string, cfg: LaunchConfig, output: auto, args: tuple) =
    ## Usage:
    ##   batch.recordGlobalFn(library, "saxpy", LaunchConfig(grid: dim3(2), blk: dim3(64)), outMem, (xMem, yMem, 3.0'f32))
    if not objc.isNil(batch.recording):
      raiseAssert("recordGlobalFn: the batch is finalized, entries are frozen")
    if cfg.grid.x < 1 or cfg.grid.y < 1 or cfg.grid.z < 1:
      raiseAssert("Metal batch: cfg.grid must be >= 1 per axis, got " &
                  $cfg.grid.x & "x" & $cfg.grid.y & "x" & $cfg.grid.z)
    if cfg.blk.x < 1 or cfg.blk.y < 1 or cfg.blk.z < 1:
      raiseAssert("Metal batch: cfg.blk must be >= 1 per axis, got " &
                  $cfg.blk.x & "x" & $cfg.blk.y & "x" & $cfg.blk.z)
    if cfg.blk.x * cfg.blk.y * cfg.blk.z > MaxThreadsPerThreadgroup:
      raiseAssert("Metal batch: cfg.blk " & $cfg.blk.x & "x" & $cfg.blk.y & "x" & $cfg.blk.z &
                  " exceeds the " & $MaxThreadsPerThreadgroup & " threads-per-threadgroup max")
    if cfg.sharedMem > 0:
      raiseAssert("Metal batch: the recording does not set threadgroup memory, sharedMem must be 0, got " & $cfg.sharedMem)
    if cfg.stream != 0:
      raiseAssert("Metal batch: replay runs on the batch queue, stream must be 0, got " & $cfg.stream)
    # Two libraries can export the same entrypoint name
    let psoKey = $cast[int](library) & "\0" & entrypoint
    if psoKey notin batch.pipelines:
      batch.pipelines[psoKey] = compileIndirectPipelineState(batch.device, library, entrypoint)
    var entry = BatchEntry(pipelineState: batch.pipelines[psoKey], fleet: cfg.grid, squadron: cfg.blk)
    bindArgs(entry, output, args, storeEntryArg, objc.ID, cgMetal)
    batch.entries.add entry

  func getScalarSize*(batch: MetalBatch): int =
    for e in batch.entries:
      result = round_multiple_up(result + e.scalars.len, 16)

  proc finalize*(batch: MetalBatch) =
    if not objc.isNil(batch.recording):
      return
    if batch.entries.len == 0:
      raiseAssert("finalize: the batch has no entries")
    batch.recording = newRecording(batch.device, batch.entries.len)
    if batch.getScalarSize > 0:
      batch.constants = objc.msgSend(batch.device, objc.`$$`("newBufferWithLength:options:"),
                                     objc.NSUInteger(batch.getScalarSize), SharedUntrackedOptions)
      if objc.isNil(batch.constants):
        raiseAssert("Metal batch: newBufferWithLength(" & $batch.getScalarSize & " bytes) returned nil")
    let image = if objc.isNil(batch.constants):
                  objc.ID(nil)
                else:
                  objc.msgSend(batch.constants, objc.`$$`("contents"))
    var scalarBase = 0
    for i, e in batch.entries.mpairs:
      if not objc.isNil(image) and e.scalars.len > 0:
        copyMem(cast[ptr UncheckedArray[byte]](image)[scalarBase].addr, e.scalars[0].addr, e.scalars.len)
      let cmd = objc.msgSend(batch.recording, objc.`$$`("indirectComputeCommandAtIndex:"), objc.NSUInteger(i))
      if i > 0:
        discard objc.msgSend(cmd, objc.`$$`("setBarrier"))
      discard objc.msgSend(cmd, objc.`$$`("setComputePipelineState:"), e.pipelineState)
      var argIdx = 0
      for slot in e.slots:
        case slot.kind
        of skBuffer:
          discard objc.msgSend(cmd, objc.`$$`("setKernelBuffer:offset:atIndex:"),
                               e.borrowedBuffers[slot.buffer], objc.NSUInteger(0), objc.NSUInteger(argIdx))
        of skBytes:
          discard objc.msgSend(cmd, objc.`$$`("setKernelBuffer:offset:atIndex:"),
                               batch.constants, objc.NSUInteger(scalarBase + slot.off), objc.NSUInteger(argIdx))
        inc argIdx
      let g = objc.MTLSize(width: objc.NSUInteger(e.fleet.x),
                           height: objc.NSUInteger(e.fleet.y),
                           depth: objc.NSUInteger(e.fleet.z))
      let b = objc.MTLSize(width: objc.NSUInteger(e.squadron.x),
                           height: objc.NSUInteger(e.squadron.y),
                           depth: objc.NSUInteger(e.squadron.z))
      discard objc.msgSend(cmd, objc.`$$`("concurrentDispatchThreadgroups:threadsPerThreadgroup:"), g, b)
      scalarBase = round_multiple_up(scalarBase + e.scalars.len, 16)

  proc replay*(batch: MetalBatch) =
    if objc.isNil(batch.recording):
      raiseAssert("replay: finalize the batch first")
    let cmdBuf = objc.msgSend(batch.queue, objc.`$$`("commandBuffer"))
    let encoder = objc.msgSend(cmdBuf, objc.`$$`("computeCommandEncoder"))
    for e in batch.entries:
      for h in e.borrowedBuffers:
        discard msgSendUseResource(encoder, objc.`$$`("useResource:usage:"), h, ResourceUsageReadWrite)
    if not objc.isNil(batch.constants):
      discard msgSendUseResource(encoder, objc.`$$`("useResource:usage:"), batch.constants, ResourceUsageReadWrite)
    discard objc.msgSend(encoder, objc.`$$`("executeCommandsInBuffer:withRange:"),
                         batch.recording, objc.NSUInteger(0), objc.NSUInteger(batch.entries.len))
    discard objc.msgSend(encoder, objc.`$$`("endEncoding"))
    discard objc.msgSend(cmdBuf, objc.`$$`("commit"))
    discard objc.msgSend(cmdBuf, objc.`$$`("waitUntilCompleted"))
    let status = objc.msgSendUInt(cmdBuf, objc.`$$`("status"))
    if status != MTLCommandBufferStatusCompleted:
      var detail = "no NSError object provided"
      let err = objc.msgSend(cmdBuf, objc.`$$`("error"))
      if not objc.isNil(err):
        detail = objc.nsStringToNimString(objc.msgSend(err, objc.`$$`("localizedDescription")))
      raiseAssert("Metal batch replay failed (status " & $status & ", " &
                  $MTLCommandBufferStatusCompleted & "=completed): " & detail)

else:
  proc newMetalBatch*(device, queue: objc.ID): MetalBatch =
    raiseAssert("cgMetal requires macOS")
