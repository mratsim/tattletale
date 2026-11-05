# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Metal backend: device storage for TensorImpl[float32, bkMetal] and the batched submission path.
##
## One session owns the default device, one command queue and one library.
## The library ingests the kernel sandbox source once, pipeline states cache per kernel name.
##
## eval() encodes every pending entry onto one persistent compute encoder of one command buffer.
## One dispatch per entry runs in record order, one commit, one wait, one submission.
## Uploads and readback memcpy through the shared buffer's contents, they never submit.
##
## Device buffers leak, tensorcery frees no device memory yet.

when not defined(macosx):
  {.error: "the Metal backend requires macOS".}

import std/tables
import pkg/iface

import
  workspace/crucible/src/runtime/backends/metal_runtime,
  workspace/crucible/src/runtime/errors,
  workspace/crucible/src/abis/objc_abi as objc

import
  workspace/data_structures/src/small_seqs,
  workspace/tensorcery/src/raw/dtypes,
  workspace/tensorcery/src/raw/impl,
  workspace/tensorcery/src/ctx,
  workspace/tensorcery/src/kernels/k_add,
  workspace/tensorcery/src/kernels/k_mul,
  workspace/tensorcery/src/kernels/k_mul2,
  workspace/tensorcery/src/kernels/k_fill

const
  workgroupSize = 256
    ## Threads per workgroup of the elementwise dispatches.
  cmdBufCompleted = 4
    ## MTLCommandBufferStatus: the command buffer ran to completion.
  cmdBufError = 5
    ## MTLCommandBufferStatus: the command buffer aborted with an error.

const metalSource* = kAddMsl & kMulMsl & kMul2Msl & kFillMsl

const kernelNames: array[KernelId, string] =
  [opAdd: "k_add", opMul: "k_mul", opMul2: "k_mul2", opFill: "k_fill",
   opReshape: "", opPermute: "", opSlice: ""]

var
  sessionDevice, sessionQueue, sessionLibrary: objc.ID
  sessionPsos: Table[string, objc.ID]
  submissionCount*: int
    ## Metal submissions since the last reset, one per committed command buffer.
    ## Tests reset it and assert the exact count.

proc resetSubmissionCount*() =
  submissionCount = 0

proc contents(h: objc.ID): pointer =
  ## Returns a shared buffer's host-visible contents pointer.
  result = cast[pointer](objc.msgSend(h, objc.`$$`("contents")))
  if result.isNil:
    failLoud("contents() returned nil for a shared buffer")

proc readHandle(blob: openArray[byte], off: int): objc.ID =
  var v: uint64
  copyMem(v.addr, unsafeAddr blob[off], 8)
  cast[objc.ID](v)

proc bindBlob(encoder: objc.ID, blob: openArray[byte], bufferSlots, scalarsSize: int) =
  ## Binds one recorded op in MSL parameter order: `bufferSlots` device buffers
  ## from the leading 8-byte address slots, then the scalar image as one setBytes.
  for i in 0 ..< bufferSlots:
    discard objc.msgSend(encoder, objc.`$$`("setBuffer:offset:atIndex:"),
        readHandle(blob, i * 8), objc.NSUInteger(0), objc.NSUInteger(i))
  discard objc.msgSend(encoder, objc.`$$`("setBytes:length:atIndex:"),
      unsafeAddr blob[bufferSlots * 8], objc.NSUInteger(scalarsSize), objc.NSUInteger(bufferSlots))

proc dispatchThreadgroups(encoder: objc.ID, cfg: LaunchCfg) =
  let grid = objc.MTLSize(
    width: objc.NSUInteger(cfg.gridX),
    height: objc.NSUInteger(cfg.gridY),
    depth: objc.NSUInteger(cfg.gridZ))
  let blk = objc.MTLSize(
    width: objc.NSUInteger(cfg.blkX),
    height: objc.NSUInteger(cfg.blkY),
    depth: objc.NSUInteger(cfg.blkZ))
  discard objc.msgSend(encoder, objc.`$$`("dispatchThreadgroups:threadsPerThreadgroup:"), grid, blk)

proc checkCmdBuf(cmdBuf: objc.ID) =
  ## Quits loudly unless the command buffer completed, reporting the NSError detail.
  let status = objc.msgSendUInt(cmdBuf, objc.`$$`("status"))
  if status != cmdBufCompleted:
    var detail = "no NSError object provided"
    let err = objc.msgSend(cmdBuf, objc.`$$`("error"))
    if not objc.isNil(err):
      detail = objc.nsStringToNimString(objc.msgSend(err, objc.`$$`("localizedDescription")))
    failLoud("Metal run: command buffer failed (status " & $status & ", " &
        $cmdBufError & "=error): " & detail)

proc cachePso(kernel: string): objc.ID =
  ## Session pipeline states: one per kernel name, cached on first use.
  ## Alternating kernels in one record reuse the cached states.
  if not sessionPsos.hasKey(kernel):
    sessionPsos[kernel] = compilePipelineState(sessionDevice, sessionLibrary, kernel)
  sessionPsos[kernel]

proc allocDeviceMem*(nbytes: int): DeviceHandle =
  ## Allocates a shared Metal buffer, returned as an opaque handle.
  result = DeviceHandle(cast[uint64](objc.msgSend(sessionDevice,
      objc.`$$`("newBufferWithLength:options:"),
      objc.NSUInteger(nbytes), SharedUntrackedOptions)))
  if cast[objc.ID](result).isNil:
    failLoud("allocDeviceMem: newBufferWithLength:options: returned nil (length " & $nbytes & ")")

proc uploadDeviceMem*(mem: DeviceHandle, data: pointer, nbytes: int) =
  copyMem(contents(cast[objc.ID](mem)), data, nbytes)

proc readbackDeviceMem*(mem: DeviceHandle, dst: pointer, nbytes: int) =
  copyMem(dst, contents(cast[objc.ID](mem)), nbytes)

proc drainMetal(ctx: Ctx) {.nimcall.} =
  ## One submission for the pending entries: every entry encodes onto one
  ## persistent compute encoder of one command buffer, one dispatch per entry
  ## in record order, then commit, wait, status check.
  objc.withMemPool:
    let cmdBuf = objc.msgSend(sessionQueue, objc.`$$`("commandBuffer"))
    # commandBuffer/computeCommandEncoder return autoreleased objects.
    # Retaining both prevents their deallocation across autorelease-pool drains until commit.
    discard objc.msgSend(cmdBuf, objc.`$$`("retain"))
    let encoder = objc.msgSend(cmdBuf, objc.`$$`("computeCommandEncoder"))
    discard objc.msgSend(encoder, objc.`$$`("retain"))
    for entry in ctx.record:
      if entry.backend != bkMetal or not ctx.isPending(entry.id):
        continue
      if isShapeOp(entry.kernel):
        # pure metadata, never dispatches
        continue
      discard objc.msgSend(encoder, objc.`$$`("setComputePipelineState:"),
          cachePso(kernelNames[entry.kernel]))
      case entry.kernel
      of opAdd:
        bindBlob(encoder, entry.blob, addBufferSlots, sizeof(AddScalars))
      of opMul:
        bindBlob(encoder, entry.blob, mulBufferSlots, sizeof(MulScalars))
      of opMul2:
        bindBlob(encoder, entry.blob, mul2BufferSlots, sizeof(Mul2Scalars))
      of opFill:
        bindBlob(encoder, entry.blob, fillBufferSlots, sizeof(FillScalars))
      of opReshape, opPermute, opSlice:
        # pure metadata, skipped above
        discard
      dispatchThreadgroups(encoder, entry.cfg)
    discard objc.msgSend(encoder, objc.`$$`("endEncoding"))
    discard objc.msgSend(cmdBuf, objc.`$$`("commit"))
    submissionCount.inc
    discard objc.msgSend(cmdBuf, objc.`$$`("waitUntilCompleted"))
    checkCmdBuf(cmdBuf)
    objc.release(encoder)
    objc.release(cmdBuf)

proc ensureMetal*(ctx: Ctx) =
  ## One-time session setup plus ctx wiring.
  ## The session owns the default device, one command queue and one library ingest of the kernel sandbox source, idempotent.
  objc.withMemPool:
    if sessionDevice.isNil:
      sessionDevice = objc.MTLCreateSystemDefaultDevice()
      if objc.isNil(sessionDevice):
        failLoud("no Metal device")
      sessionQueue = objc.msgSend(sessionDevice, objc.`$$`("newCommandQueue"))
      if objc.isNil(sessionQueue):
        failLoud("newCommandQueue returned nil")
      let opts = compileOptions()
      sessionLibrary = compileLibrary(sessionDevice, metalSource, opts)
      objc.release(opts)
    ctx.drains[bkMetal] = drainMetal

proc elementwiseCfg(n: int): LaunchCfg =
  LaunchCfg(
    gridX: int32((n + workgroupSize - 1) div workgroupSize),
    gridY: 1, gridZ: 1,
    blkX: workgroupSize, blkY: 1, blkZ: 1)

proc padTo4(v: SmallSeq[4, int32], pad: int32): array[4, int32] =
  ## Pads to 4 axes with `pad`: shapes pad with 1s (the divmod chain stays safe), strides with 0s (padded axes contribute nothing).
  for i in 0 ..< 4:
    result[i] = if i < int(v.len): v[i] else: pad

proc toMetalTensorImpl*[T](ctx: Ctx, data: openArray[T], shape: openArray[int], requiresGrad = false): TensorImpl[T, bkMetal] =
  ## Builds a Metal tensor from host data, contiguous, uploaded at construction.
  ## The upload memcpys into the shared buffer, it never submits.
  let meta = contiguousMeta(shape)
  result = TensorImpl[T, bkMetal](
    ctx: ctx,
    tid: ctx.newTid(),
    requiresGrad: requiresGrad,
    shape: meta.shape,
    strides: meta.strides,
    mem: allocDeviceMem(data.len * sizeof(T)))
  uploadDeviceMem(result.mem, unsafeAddr data[0], data.len * sizeof(T))

proc add*[T](s: TensorImpl[T, bkMetal], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased add: packs the op's args into one blob at enqueue, appends one record entry, returns the erased output tensor.
  ## The output requires grad when either input does.
  let o = other.to(TensorImpl[T, bkMetal])
  let n = s.getNumel()
  let outMem = allocDeviceMem(n * sizeof(T))
  let blob = packAddArgs(AddArgs(
    outAddr: cast[uint64](outMem),
    aAddr: cast[uint64](s.mem),
    bAddr: cast[uint64](o.mem),
    scalars: AddScalars(
      aOff: int32(s.offset),
      bOff: int32(o.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  let outTid = s.ctx.newTid()
  let opId = s.ctx.enqueue(opAdd, blob, elementwiseCfg(n), bkMetal, outTid, s.tid, o.tid)
  TensorImpl[T, bkMetal](
    ctx: s.ctx,
    tid: outTid,
    seqNo: opId,
    requiresGrad: s.requiresGrad or o.requiresGrad,
    shape: s.shape,
    strides: contiguousStrides(s.shape),
    mem: outMem).to(AnyTensorImpl)

proc mul*[T](s: TensorImpl[T, bkMetal], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased tensor product: packs the op's args into one blob at enqueue,
  ## appends one record entry, returns the erased output tensor.
  ## The output requires grad when either input does.
  let o = other.to(TensorImpl[T, bkMetal])
  let n = s.getNumel()
  let outMem = allocDeviceMem(n * sizeof(T))
  let blob = packMul2Args(Mul2Args(
    outAddr: cast[uint64](outMem),
    aAddr: cast[uint64](s.mem),
    bAddr: cast[uint64](o.mem),
    scalars: Mul2Scalars(
      aOff: int32(s.offset),
      bOff: int32(o.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  let outTid = s.ctx.newTid()
  let opId = s.ctx.enqueue(opMul2, blob, elementwiseCfg(n), bkMetal, outTid, s.tid, o.tid)
  TensorImpl[T, bkMetal](
    ctx: s.ctx,
    tid: outTid,
    seqNo: opId,
    requiresGrad: s.requiresGrad or o.requiresGrad,
    shape: s.shape,
    strides: contiguousStrides(s.shape),
    mem: outMem).to(AnyTensorImpl)

proc mulScalar*[T](s: TensorImpl[T, bkMetal], scalar: float64): AnyTensorImpl =
  ## Erased scalar multiply: the scalar narrows to the storage element type at pack time.
  ## The output requires grad when the input does.
  let n = s.getNumel()
  let outMem = allocDeviceMem(n * sizeof(T))
  let blob = packMulArgs(MulArgs(
    outAddr: cast[uint64](outMem),
    aAddr: cast[uint64](s.mem),
    scalars: MulScalars(
      aOff: int32(s.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      scalar: float32(scalar),
      n: int32(n))))
  let outTid = s.ctx.newTid()
  let opId = s.ctx.enqueue(opMul, blob, elementwiseCfg(n), bkMetal, outTid, s.tid)
  TensorImpl[T, bkMetal](
    ctx: s.ctx,
    tid: outTid,
    seqNo: opId,
    requiresGrad: s.requiresGrad,
    shape: s.shape,
    strides: contiguousStrides(s.shape),
    mem: outMem).to(AnyTensorImpl)

proc addNoGrad*[T](s: TensorImpl[T, bkMetal], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased grad-chain add: runs the k_add record, grad flag off.
  ## backward() adds fan-out accumulation through it.
  let o = other.to(TensorImpl[T, bkMetal])
  let n = s.getNumel()
  let outMem = allocDeviceMem(n * sizeof(T))
  let blob = packAddArgs(AddArgs(
    outAddr: cast[uint64](outMem),
    aAddr: cast[uint64](s.mem),
    bAddr: cast[uint64](o.mem),
    scalars: AddScalars(
      aOff: int32(s.offset),
      bOff: int32(o.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  let outTid = s.ctx.newTid()
  let opId = s.ctx.enqueue(opAdd, blob, elementwiseCfg(n), bkMetal, outTid, s.tid, o.tid, isGrad = true)
  TensorImpl[T, bkMetal](
    ctx: s.ctx,
    tid: outTid,
    seqNo: opId,
    shape: s.shape,
    strides: contiguousStrides(s.shape),
    mem: outMem).to(AnyTensorImpl)

proc mulNoGrad*[T](s: TensorImpl[T, bkMetal], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased grad-chain tensor product: runs the k_mul2 record, grad flag off.
  ## backward() builds the mul vjp from it.
  let o = other.to(TensorImpl[T, bkMetal])
  let n = s.getNumel()
  let outMem = allocDeviceMem(n * sizeof(T))
  let blob = packMul2Args(Mul2Args(
    outAddr: cast[uint64](outMem),
    aAddr: cast[uint64](s.mem),
    bAddr: cast[uint64](o.mem),
    scalars: Mul2Scalars(
      aOff: int32(s.offset),
      bOff: int32(o.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  let outTid = s.ctx.newTid()
  let opId = s.ctx.enqueue(opMul2, blob, elementwiseCfg(n), bkMetal, outTid, s.tid, o.tid, isGrad = true)
  TensorImpl[T, bkMetal](
    ctx: s.ctx,
    tid: outTid,
    seqNo: opId,
    shape: s.shape,
    strides: contiguousStrides(s.shape),
    mem: outMem).to(AnyTensorImpl)

proc mulScalarNoGrad*[T](s: TensorImpl[T, bkMetal], scalar: float64): AnyTensorImpl =
  ## Erased grad-chain scalar multiply: runs the k_mul record, grad flag off.
  ## backward() builds the mulScalar vjp from it.
  let n = s.getNumel()
  let outMem = allocDeviceMem(n * sizeof(T))
  let blob = packMulArgs(MulArgs(
    outAddr: cast[uint64](outMem),
    aAddr: cast[uint64](s.mem),
    scalars: MulScalars(
      aOff: int32(s.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      scalar: float32(scalar),
      n: int32(n))))
  let outTid = s.ctx.newTid()
  let opId = s.ctx.enqueue(opMul, blob, elementwiseCfg(n), bkMetal, outTid, s.tid, isGrad = true)
  TensorImpl[T, bkMetal](
    ctx: s.ctx,
    tid: outTid,
    seqNo: opId,
    shape: s.shape,
    strides: contiguousStrides(s.shape),
    mem: outMem).to(AnyTensorImpl)

proc onesNoGrad*[T](s: TensorImpl[T, bkMetal]): AnyTensorImpl =
  ## Erased grad-chain ones fill: runs the k_fill record, grad flag off.
  ## backward() seeds the cotangent map from it.
  let n = s.getNumel()
  let outMem = allocDeviceMem(n * sizeof(T))
  let blob = packFillArgs(FillArgs(
    outAddr: cast[uint64](outMem),
    scalars: FillScalars(n: int32(n))))
  let outTid = s.ctx.newTid()
  let opId = s.ctx.enqueue(opFill, blob, elementwiseCfg(n), bkMetal, outTid, isGrad = true)
  TensorImpl[T, bkMetal](
    ctx: s.ctx,
    tid: outTid,
    seqNo: opId,
    shape: s.shape,
    strides: contiguousStrides(s.shape),
    mem: outMem).to(AnyTensorImpl)

proc readbackBytes*[T](s: TensorImpl[T, bkMetal]): seq[byte] =
  ## Copies the tensor to host bytes in logical order, last axis fastest,
  ## materializing pending ops first. A view gathers through its strides and offset.
  ## Reading a pending tensor evals the context (the MLX rule).
  if s.ctx.isPending(s.seqNo):
    s.ctx.eval()
  if s.getNumel() == 0:
    return newSeq[byte](0)
  let base = cast[ptr T](cast[int](contents(cast[objc.ID](s.mem))) + s.offset * sizeof(T))
  stridedBytes[T](base, s.shape, s.strides)

proc toErased*[T](s: TensorImpl[T, bkMetal]): AnyTensorImpl =
  ## Returns the erased face over `s`.
  ##
  ## Contract: the vtable's mixins resolve at the outermost instantiation scope.
  ## tensors.nim re-exports the typed conformant procs, so module users satisfy
  ## the interface conformance check.
  s.to(AnyTensorImpl)
