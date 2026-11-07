# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/tables
import workspace/crucible/src/abis/opencl_abi
import workspace/crucible/src/crucible_codegen
import workspace/crucible/src/runtime/backends/opencl_runtime
import workspace/crucible/src/runtime/backends/runtime_utils
import workspace/crucible/src/runtime/engines/engines_common
import workspace/crucible/src/runtime/launch/launch_serialization

export engines_common.LaunchConfig, engines_common.dim3

# ############################################################
#
#                   OpenCL batch submission
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
    kernel: Pkernel
    fleet, squadron: Dim3
    borrowedBuffers: seq[Pmem]
    scalars: seq[byte]
    slots: seq[Slot]

  OpenCLBatchObj = object
    context: Pcontext
    queue: Pcommand_queue
    entries: seq[BatchEntry]
    pipelines*: Table[(string, string), Pkernel]
    programs: Table[string, Pprogram]
    finalized: bool

  OpenCLBatch* = ref OpenCLBatchObj

const
  QueueDevice = cl_uint(0x1091)
  QueueProperties = cl_uint(0x1093)
  # CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE is bit 0, bit 2 is CL_QUEUE_ON_DEVICE
  QueueOutOfOrderExec = cl_ulong(1 shl 0)

proc getCommandQueueInfo(
  queue: Pcommand_queue,
  param_name: cl_uint,
  param_value_size: cl_size_t,
  param_value: pointer,
  param_value_size_ret: ptr cl_size_t
): TClResult {.importc: "clGetCommandQueueInfo", dynlib: ClLib.}

proc `=destroy`*(batch: var OpenCLBatchObj) =
  ## Releases the batch's own objects, never the borrowed handles.
  for k in batch.pipelines.values:
    discard releaseKernel(k)
  for p in batch.programs.values:
    discard releaseProgram(p)

proc queueDevice(queue: Pcommand_queue): Pdevice_id =
  var device: Pdevice_id
  check getCommandQueueInfo(queue, QueueDevice, cl_size_t(sizeof(Pdevice_id)), device.addr, nil)
  result = device

proc compileEntrypoint(batch: OpenCLBatch, source, entrypoint: string): Pkernel =
  if source notin batch.programs:
    let strings = allocCStringArray([source])
    defer: deallocCStringArray(strings)

    var
      length = cl_size_t(source.len)
      status: TClResult
    let program = createProgramWithSource(batch.context, 1, strings, length.addr, status.addr)
    check status

    var device = queueDevice(batch.queue)
    status = buildProgram(program, 1, device.addr, nil, nil, nil)

    if status == BUILD_PROGRAM_FAILURE:
      var logLength: cl_size_t
      check getProgramBuildInfo(program, device, PROGRAM_BUILD_LOG, 0, nil, logLength.addr)
      var log = newString(logLength.int)
      if log.len > 0:
        check getProgramBuildInfo(program, device, PROGRAM_BUILD_LOG, logLength, log[0].addr, nil)
        if log[^1] == '\0':   # clGetProgramBuildInfo strings are NUL-terminated
          log.setLen(log.len - 1)
      raiseAssert("OpenCL batch failed to build program for '" & entrypoint & "': " & log)
    else:
      check status

    batch.programs[source] = program

  var status: TClResult
  result = createKernel(batch.programs[source], entrypoint.cstring, status.addr)
  check status

proc storeEntryArg(entry: var BatchEntry, v: auto, _: int) =
  when typeof(v) is ptr|pointer:
    entry.borrowedBuffers.add Pmem(v)
    entry.slots.add Slot(kind: skBuffer, buffer: entry.borrowedBuffers.high)
  else:
    const layout = computeDeviceLayout(typeof(v), cgOpenCL)
    let off = entry.scalars.len.round_multiple_up(layout.align)
    entry.scalars.setLen(off + layout.size)
    var w = off
    entry.scalars.packFrom(w, v, cgOpenCL)
    entry.slots.add Slot(kind: skBytes, off: off, size: layout.size)

proc newOpenCLBatch*(context: Pcontext, queue: Pcommand_queue): OpenCLBatch =
  ## Borrows context, queue and buffers
  var properties: cl_ulong
  check getCommandQueueInfo(queue, QueueProperties, cl_size_t(sizeof(cl_ulong)), properties.addr, nil)
  if (properties and QueueOutOfOrderExec) != 0:
    raiseAssert("OpenCL batch: replay preserves entry order through in-order-queue semantics, the queue enables out-of-order execution")
  OpenCLBatch(context: context, queue: queue)

proc recordGlobalFn*(batch: OpenCLBatch, source: string, entrypoint: string, cfg: LaunchConfig, output: auto, args: tuple) =
  if batch.finalized:
    raiseAssert("recordGlobalFn: the batch is finalized, entries are frozen")
  if cfg.grid.x < 1 or cfg.grid.y < 1 or cfg.grid.z < 1:
    raiseAssert("OpenCL batch: cfg.grid must be >= 1 per axis, got " &
                $cfg.grid.x & "x" & $cfg.grid.y & "x" & $cfg.grid.z)
  if cfg.blk.x < 1 or cfg.blk.y < 1 or cfg.blk.z < 1:
    raiseAssert("OpenCL batch: cfg.blk must be >= 1 per axis, got " &
                $cfg.blk.x & "x" & $cfg.blk.y & "x" & $cfg.blk.z)
  if cfg.sharedMem != 0 or cfg.stream != 0:
    raiseAssert("OpenCL batch: launch-time sharedMem/stream are unsupported, they are Metal/CUDA features")
  # One entrypoint name exists in many sources, the kernel is per (source, entrypoint) pair
  let kernelKey = (source, entrypoint)
  if kernelKey notin batch.pipelines:
    batch.pipelines[kernelKey] = compileEntrypoint(batch, source, entrypoint)
  var entry = BatchEntry(kernel: batch.pipelines[kernelKey], fleet: cfg.grid, squadron: cfg.blk)
  bindArgs(entry, output, args, storeEntryArg, Pmem, cgOpenCL)
  batch.entries.add entry

func getScalarSize*(batch: OpenCLBatch): int =
  for e in batch.entries:
    result += e.scalars.len

proc finalize*(batch: OpenCLBatch) =
  if batch.finalized:
    return
  if batch.entries.len == 0:
    raiseAssert("finalize: the batch has no entries")
  batch.finalized = true

proc replay*(batch: OpenCLBatch) =
  if not batch.finalized:
    raiseAssert("replay: finalize the batch first")
  for e in batch.entries.mitems:
    var argIdx = cl_uint(0)
    for slot in e.slots:
      case slot.kind
      of skBuffer:
        check setKernelArg(e.kernel, argIdx, cl_size_t(sizeof(Pmem)), e.borrowedBuffers[slot.buffer].unsafeAddr)
      of skBytes:
        check setKernelArg(e.kernel, argIdx, cl_size_t(slot.size), e.scalars[slot.off].unsafeAddr)
      inc argIdx
    let globalSize = [cl_size_t(e.fleet.x * e.squadron.x),
                      cl_size_t(e.fleet.y * e.squadron.y),
                      cl_size_t(e.fleet.z * e.squadron.z)]
    let localSize = [cl_size_t(e.squadron.x), cl_size_t(e.squadron.y), cl_size_t(e.squadron.z)]
    check enqueueNDRangeKernel(batch.queue, e.kernel, cl_uint(3), nil,
                               globalSize[0].unsafeAddr, localSize[0].unsafeAddr, 0, nil, nil)
  check flush(batch.queue)
