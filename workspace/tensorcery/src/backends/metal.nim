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
  workspace/tensorcery/src/kernels/k_fill,
  workspace/tensorcery/src/kernels/k_gather,
  workspace/tensorcery/src/kernels/k_layernorm,
  workspace/tensorcery/src/kernels/k_gelu,
  workspace/tensorcery/src/kernels/k_softmax,
  workspace/tensorcery/src/kernels/k_crossentropy,
  workspace/tensorcery/src/kernels/k_where,
  workspace/tensorcery/src/kernels/k_sgd_adam

const
  workgroupSize = 256
    ## Threads per workgroup of the elementwise dispatches.
  cmdBufCompleted = 4
    ## MTLCommandBufferStatus: the command buffer ran to completion.
  cmdBufError = 5
    ## MTLCommandBufferStatus: the command buffer aborted with an error.

const metalSource* = "#include <metal_stdlib>\nusing namespace metal;\n" &
    kAddMsl & kMulMsl & kMul2Msl & kFillMsl &
    kGatherMsl & kLayerNormMsl & kGeluMsl & kSoftmaxMsl & kCrossEntropyMsl &
    kWhereMsl & kSgdMsl & kAdamMsl

const kernelNames: array[KernelId, string] =
  [opAdd: "k_add", opMul: "k_mul", opMul2: "k_mul2", opFill: "k_fill",
   opReshape: "", opPermute: "", opSlice: "",
   opGather: "k_gather", opLayerNorm: "k_layernorm", opGelu: "k_gelu",
   opSoftmax: "k_softmax", opCrossEntropy: "k_crossentropy", opWhere: "k_where",
   opSgd: "k_sgd", opAdam: "k_adam"]

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

proc allocSharedMem*(nbytes, numel: int): SharedMemory =
  ## Allocates a shared Metal buffer as a SharedMemory handle, contents uninitialized.
  SharedMemory(handle: allocDeviceMem(nbytes), seqNo: 0, numel: numel)

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
      of opGather:
        bindBlob(encoder, entry.blob, gatherBufferSlots, sizeof(GatherScalars))
      of opLayerNorm:
        bindBlob(encoder, entry.blob, layerNormBufferSlots, sizeof(LayerNormScalars))
      of opGelu:
        bindBlob(encoder, entry.blob, geluBufferSlots, sizeof(GeluScalars))
      of opSoftmax:
        bindBlob(encoder, entry.blob, softmaxBufferSlots, sizeof(SoftmaxScalars))
      of opCrossEntropy:
        bindBlob(encoder, entry.blob, crossEntropyBufferSlots, sizeof(CrossEntropyScalars))
      of opWhere:
        bindBlob(encoder, entry.blob, whereBufferSlots, sizeof(WhereScalars))
      of opSgd:
        bindBlob(encoder, entry.blob, sgdBufferSlots, sizeof(SgdScalars))
      of opAdam:
        bindBlob(encoder, entry.blob, adamBufferSlots, sizeof(AdamScalars))
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
    mem: SharedMemory(handle: allocDeviceMem(data.len * sizeof(T)),
        seqNo: 0, numel: data.len))
  uploadDeviceMem(result.mem.handle, unsafeAddr data[0], data.len * sizeof(T))

proc record[T](s: TensorImpl[T, bkMetal], kernel: KernelId, blob: seq[byte], cfg: LaunchCfg,
    outMem: SharedMemory, outShape: SmallSeq[4, int32], requiresGrad, isGrad: bool,
    inA, inB, inC: int): TensorImpl[T, bkMetal] =
  ## Enqueues one record entry and wraps the freshly allocated output storage.
  ## The storage epoch advances to the entry, so reads of the output materialize it first.
  let outTid = s.ctx.newTid()
  let opId = s.ctx.enqueue(kernel, blob, cfg, bkMetal, outTid, inA, inB, inC, 0, isGrad)
  outMem.seqNo = opId
  TensorImpl[T, bkMetal](
    ctx: s.ctx,
    tid: outTid,
    requiresGrad: requiresGrad,
    shape: outShape,
    strides: contiguousStrides(outShape),
    mem: outMem)

proc add*[T](s: TensorImpl[T, bkMetal], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased add: packs the op's args into one blob at enqueue, appends one record entry, returns the erased output tensor.
  ## The output requires grad when either input does.
  let o = other.to(TensorImpl[T, bkMetal])
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let blob = packAddArgs(AddArgs(
    outAddr: cast[uint64](outMem.handle),
    aAddr: cast[uint64](s.mem.handle),
    bAddr: cast[uint64](o.mem.handle),
    scalars: AddScalars(
      aOff: int32(s.offset),
      bOff: int32(o.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  s.record(opAdd, blob, elementwiseCfg(n), outMem, s.shape,
      s.requiresGrad or o.requiresGrad, false, s.tid, o.tid, 0).to(AnyTensorImpl)

proc mul*[T](s: TensorImpl[T, bkMetal], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased tensor product: packs the op's args into one blob at enqueue,
  ## appends one record entry, returns the erased output tensor.
  ## The output requires grad when either input does.
  let o = other.to(TensorImpl[T, bkMetal])
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let blob = packMul2Args(Mul2Args(
    outAddr: cast[uint64](outMem.handle),
    aAddr: cast[uint64](s.mem.handle),
    bAddr: cast[uint64](o.mem.handle),
    scalars: Mul2Scalars(
      aOff: int32(s.offset),
      bOff: int32(o.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  s.record(opMul2, blob, elementwiseCfg(n), outMem, s.shape,
      s.requiresGrad or o.requiresGrad, false, s.tid, o.tid, 0).to(AnyTensorImpl)

proc mulScalar*[T](s: TensorImpl[T, bkMetal], scalar: float64): AnyTensorImpl =
  ## Erased scalar multiply: the scalar narrows to the storage element type at pack time.
  ## The output requires grad when the input does.
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let blob = packMulArgs(MulArgs(
    outAddr: cast[uint64](outMem.handle),
    aAddr: cast[uint64](s.mem.handle),
    scalars: MulScalars(
      aOff: int32(s.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      scalar: float32(scalar),
      n: int32(n))))
  s.record(opMul, blob, elementwiseCfg(n), outMem, s.shape,
      s.requiresGrad, false, s.tid, 0, 0).to(AnyTensorImpl)

proc addNoGrad*[T](s: TensorImpl[T, bkMetal], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased grad-chain add: runs the k_add record, grad flag off.
  ## backward() adds fan-out accumulation through it.
  let o = other.to(TensorImpl[T, bkMetal])
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let blob = packAddArgs(AddArgs(
    outAddr: cast[uint64](outMem.handle),
    aAddr: cast[uint64](s.mem.handle),
    bAddr: cast[uint64](o.mem.handle),
    scalars: AddScalars(
      aOff: int32(s.offset),
      bOff: int32(o.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  s.record(opAdd, blob, elementwiseCfg(n), outMem, s.shape,
      false, true, s.tid, o.tid, 0).to(AnyTensorImpl)

proc mulNoGrad*[T](s: TensorImpl[T, bkMetal], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased grad-chain tensor product: runs the k_mul2 record, grad flag off.
  ## backward() builds the mul vjp from it.
  let o = other.to(TensorImpl[T, bkMetal])
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let blob = packMul2Args(Mul2Args(
    outAddr: cast[uint64](outMem.handle),
    aAddr: cast[uint64](s.mem.handle),
    bAddr: cast[uint64](o.mem.handle),
    scalars: Mul2Scalars(
      aOff: int32(s.offset),
      bOff: int32(o.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  s.record(opMul2, blob, elementwiseCfg(n), outMem, s.shape,
      false, true, s.tid, o.tid, 0).to(AnyTensorImpl)

proc mulScalarNoGrad*[T](s: TensorImpl[T, bkMetal], scalar: float64): AnyTensorImpl =
  ## Erased grad-chain scalar multiply: runs the k_mul record, grad flag off.
  ## backward() builds the mulScalar vjp from it.
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let blob = packMulArgs(MulArgs(
    outAddr: cast[uint64](outMem.handle),
    aAddr: cast[uint64](s.mem.handle),
    scalars: MulScalars(
      aOff: int32(s.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      scalar: float32(scalar),
      n: int32(n))))
  s.record(opMul, blob, elementwiseCfg(n), outMem, s.shape,
      false, true, s.tid, 0, 0).to(AnyTensorImpl)

proc onesNoGrad*[T](s: TensorImpl[T, bkMetal]): AnyTensorImpl =
  ## Erased grad-chain ones fill: runs the k_fill record, grad flag off.
  ## backward() seeds the cotangent map from it.
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let blob = packFillArgs(FillArgs(
    outAddr: cast[uint64](outMem.handle),
    scalars: FillScalars(n: int32(n))))
  s.record(opFill, blob, elementwiseCfg(n), outMem, s.shape,
      false, true, 0, 0, 0).to(AnyTensorImpl)

proc readbackBytes*[T](s: TensorImpl[T, bkMetal]): seq[byte] =
  ## Returns:
  ## - the tensor as host bytes in logical order, last axis fastest
  ##
  ## A view gathers through its strides and offset. Reading a pending tensor
  ## evals the context first (the MLX rule).
  if s.ctx.isPending(s.mem.seqNo):
    s.ctx.eval()
  if s.getNumel() == 0:
    return newSeq[byte](0)
  let base = cast[ptr T](cast[int](contents(cast[objc.ID](s.mem.handle))) + s.offset * sizeof(T))
  stridedBytes[T](base, s.shape, s.strides)

# Microgpt op set, Metal leg
# ------------------------------------------------------------

proc gather*[T](s: TensorImpl[T, bkMetal], indices: AnyTensorImpl): AnyTensorImpl =
  ## Erased embedding gather: one embedding vector per index over the rank-2 table.
  ## The output requires grad when the table does. The int32 indices never do.
  ##
  ## Raises AssertionDefect for a non rank-2 table or any non-contiguous operand.
  let idx = indices.to(TensorImpl[int32, bkMetal])
  doAssert int(s.shape.len) == 2,
      "gather: the table must be rank 2 (vocabulary, embedding dim), got rank " & $s.shape.len
  doAssert isContiguous(s.shape, s.strides), "gather: non-contiguous table refused"
  doAssert isContiguous(idx.shape, idx.strides), "gather: non-contiguous index tensor refused"
  let e = int32(s.shape[int(s.shape.len) - 1])
  let n = idx.getNumel() * int(e)
  let outMem = allocSharedMem(n * sizeof(T), n)
  let blob = packGatherArgs(GatherArgs(
    outAddr: cast[uint64](outMem.handle),
    wAddr: cast[uint64](s.mem.handle),
    idxAddr: cast[uint64](idx.mem.handle),
    scalars: GatherScalars(
      wOff: int32(s.offset), idxOff: int32(idx.offset), e: e, n: int32(n))))
  s.record(opGather, blob, elementwiseCfg(n), outMem, appendDim(idx.shape, e),
      s.requiresGrad, false, s.tid, idx.tid, 0).to(AnyTensorImpl)

proc layernorm*[T](s: TensorImpl[T, bkMetal], w, b: AnyTensorImpl, eps: float64): AnyTensorImpl =
  ## Erased last-axis layernorm over `s` with the learned scale `w` and shift `b`.
  ## The output requires grad when any operand does.
  ##
  ## Raises AssertionDefect when the scale or shift does not hold one element per
  ## normalized axis or is non-contiguous.
  let wImpl = w.to(TensorImpl[T, bkMetal])
  let bImpl = b.to(TensorImpl[T, bkMetal])
  let d = int(s.shape[int(s.shape.len) - 1])
  let rows = s.getNumel() div d
  doAssert wImpl.getNumel() == d and bImpl.getNumel() == d,
      "layernorm: scale and shift must hold one element per normalized axis, " &
      $wImpl.getNumel() & " and " & $bImpl.getNumel() & " for axis size " & $d
  doAssert isContiguous(wImpl.shape, wImpl.strides) and isContiguous(bImpl.shape, bImpl.strides),
      "layernorm: non-contiguous scale or shift refused"
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let blob = packLayerNormArgs(LayerNormArgs(
    outAddr: cast[uint64](outMem.handle),
    xAddr: cast[uint64](s.mem.handle),
    wAddr: cast[uint64](wImpl.mem.handle),
    bAddr: cast[uint64](bImpl.mem.handle),
    scalars: LayerNormScalars(
      xOff: int32(s.offset), wOff: int32(wImpl.offset), bOff: int32(bImpl.offset),
      xShape: padTo4(s.shape, 1), xs: padTo4(s.strides, 0),
      d: int32(d), rows: int32(rows), xStep: int32(s.strides[int(s.shape.len) - 1]),
      eps: float32(eps))))
  s.record(opLayerNorm, blob, elementwiseCfg(rows), outMem, s.shape,
      s.requiresGrad or wImpl.requiresGrad or bImpl.requiresGrad, false,
      s.tid, wImpl.tid, bImpl.tid).to(AnyTensorImpl)

proc gelu*[T](s: TensorImpl[T, bkMetal]): AnyTensorImpl =
  ## Erased tanh-approximate gelu over `s`. The output requires grad when `s` does.
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let blob = packGeluArgs(GeluArgs(
    outAddr: cast[uint64](outMem.handle),
    aAddr: cast[uint64](s.mem.handle),
    scalars: GeluScalars(aOff: int32(s.offset), n: int32(n))))
  s.record(opGelu, blob, elementwiseCfg(n), outMem, s.shape,
      s.requiresGrad, false, s.tid, 0, 0).to(AnyTensorImpl)

proc softmax*[T](s: TensorImpl[T, bkMetal]): AnyTensorImpl =
  ## Erased last-axis softmax over `s`. The output requires grad when `s` does.
  let d = s.getNumel() div int(s.shape[int(s.shape.len) - 1])
  let dAxis = int(s.shape[int(s.shape.len) - 1])
  let rows = s.getNumel() div dAxis
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let blob = packSoftmaxArgs(SoftmaxArgs(
    outAddr: cast[uint64](outMem.handle),
    xAddr: cast[uint64](s.mem.handle),
    scalars: SoftmaxScalars(
      xOff: int32(s.offset),
      xShape: padTo4(s.shape, 1), xs: padTo4(s.strides, 0),
      d: int32(dAxis), rows: int32(rows), xStep: int32(s.strides[int(s.shape.len) - 1]))))
  s.record(opSoftmax, blob, elementwiseCfg(rows), outMem, s.shape,
      s.requiresGrad, false, s.tid, 0, 0).to(AnyTensorImpl)

proc crossEntropy*[T](s: TensorImpl[T, bkMetal], targets: AnyTensorImpl): AnyTensorImpl =
  ## Erased cross entropy: one loss per set of `v` logits over the last axis of `s`,
  ## the int32 `targets` name the class per set. The output requires grad when the logits do.
  ##
  ## Raises AssertionDefect for a non-contiguous target view.
  let tgt = targets.to(TensorImpl[int32, bkMetal])
  doAssert isContiguous(tgt.shape, tgt.strides), "cross entropy: non-contiguous target view refused"
  let v = int(s.shape[int(s.shape.len) - 1])
  let rows = s.getNumel() div v
  let outMem = allocSharedMem(rows * sizeof(T), rows)
  let blob = packCrossEntropyArgs(CrossEntropyArgs(
    lossAddr: cast[uint64](outMem.handle),
    logitsAddr: cast[uint64](s.mem.handle),
    targetAddr: cast[uint64](tgt.mem.handle),
    scalars: CrossEntropyScalars(
      logitsOff: int32(s.offset), targetOff: int32(tgt.offset),
      logitsShape: padTo4(s.shape, 1), logitsStrides: padTo4(s.strides, 0),
      v: int32(v), rows: int32(rows), logitStep: int32(s.strides[int(s.shape.len) - 1]))))
  s.record(opCrossEntropy, blob, elementwiseCfg(rows), outMem, dropLastDim(s.shape),
      s.requiresGrad, false, s.tid, tgt.tid, 0).to(AnyTensorImpl)

proc where*[T](s: TensorImpl[T, bkMetal], cond, other: AnyTensorImpl): AnyTensorImpl =
  ## Erased elementwise select with `s` on the true branch and `other` on the false branch.
  ## The output requires grad when any operand does.
  ##
  ## Raises AssertionDefect on a shape mismatch, no broadcasting.
  let c = cond.to(TensorImpl[T, bkMetal])
  let o = other.to(TensorImpl[T, bkMetal])
  doAssert s.shape == c.shape and s.shape == o.shape,
      "where: shape mismatch " & $s.shape & " vs " & $c.shape & " vs " & $o.shape &
      " refused, no broadcasting"
  doAssert isContiguous(c.shape, c.strides) and isContiguous(s.shape, s.strides) and
      isContiguous(o.shape, o.strides),
      "where: non-contiguous operands refused, the kernel reads them linearly from their offsets"
  let n = s.getNumel()
  let outMem = allocSharedMem(n * sizeof(T), n)
  let blob = packWhereArgs(WhereArgs(
    outAddr: cast[uint64](outMem.handle),
    condAddr: cast[uint64](c.mem.handle),
    aAddr: cast[uint64](s.mem.handle),
    bAddr: cast[uint64](o.mem.handle),
    scalars: WhereScalars(
      condOff: int32(c.offset), aOff: int32(s.offset), bOff: int32(o.offset),
      n: int32(n))))
  s.record(opWhere, blob, elementwiseCfg(n), outMem, s.shape,
      s.requiresGrad or c.requiresGrad or o.requiresGrad, false,
      c.tid, s.tid, o.tid).to(AnyTensorImpl)

proc sgdUpdate*[T](s: TensorImpl[T, bkMetal], grad: AnyTensorImpl, lr: float64): void =
  ## Erased in-place sgd update: `s` moves by `-lr * grad`.
  ## The entry's output id is `s`'s tensor id and the storage epoch advances to the entry,
  ## so reading the parameters or any view over them materializes the update first (the MLX rule).
  ##
  ## Raises AssertionDefect for a non-contiguous parameter or gradient view.
  let g = grad.to(TensorImpl[T, bkMetal])
  doAssert isContiguous(s.shape, s.strides) and isContiguous(g.shape, g.strides),
      "sgd: non-contiguous parameter or gradient view refused"
  doAssert s.shape == g.shape,
      "sgd: shape mismatch " & $s.shape & " vs " & $g.shape & " refused"
  let n = s.getNumel()
  let blob = packSgdArgs(SgdArgs(
    pAddr: cast[uint64](s.mem.handle),
    gAddr: cast[uint64](g.mem.handle),
    scalars: SgdScalars(lr: float32(lr), pOff: int32(s.offset), gOff: int32(g.offset), n: int32(n))))
  let opId = s.ctx.enqueue(opSgd, blob, elementwiseCfg(n), bkMetal, s.tid, s.tid, g.tid)
  s.mem.seqNo = opId

proc adamUpdate*[T](s: TensorImpl[T, bkMetal], grad, m, v: AnyTensorImpl, pack: AdamScalars): void =
  ## Erased in-place adam update over `s` with gradient `grad` and the persistent moments `m` and `v`.
  ##
  ## The pack lists the learning rate, decay rates, epsilon and bias corrections for the current step, all folded in by the caller.
  ##
  ## The entry's output id is `s`'s tensor id and the storage epoch advances to the entry,
  ## so reading the parameters or any view over them materializes the update first (the MLX rule).
  ##
  ## Raises AssertionDefect for a shape mismatch or a non-contiguous operand.
  let g = grad.to(TensorImpl[T, bkMetal])
  let mImpl = m.to(TensorImpl[T, bkMetal])
  let vImpl = v.to(TensorImpl[T, bkMetal])
  doAssert s.shape == g.shape and s.shape == mImpl.shape and
      s.shape == vImpl.shape,
      "adam: shape mismatch " & $s.shape & " vs " & $g.shape & " vs " & $mImpl.shape &
      " vs " & $vImpl.shape & " refused"
  doAssert isContiguous(s.shape, s.strides) and isContiguous(g.shape, g.strides) and
      isContiguous(mImpl.shape, mImpl.strides) and isContiguous(vImpl.shape, vImpl.strides),
      "adam: non-contiguous operand refused"
  let n = s.getNumel()
  var scalars = pack
  scalars.pOff = int32(s.offset)
  scalars.gOff = int32(g.offset)
  scalars.mOff = int32(mImpl.offset)
  scalars.vOff = int32(vImpl.offset)
  scalars.n = int32(n)
  let blob = packAdamArgs(AdamArgs(
    pAddr: cast[uint64](s.mem.handle),
    gAddr: cast[uint64](g.mem.handle),
    mAddr: cast[uint64](mImpl.mem.handle),
    vAddr: cast[uint64](vImpl.mem.handle),
    scalars: scalars))
  let opId = s.ctx.enqueue(opAdam, blob, elementwiseCfg(n), bkMetal, s.tid, s.tid, g.tid, mImpl.tid, vImpl.tid)
  s.mem.seqNo = opId


proc toErased*[T](s: TensorImpl[T, bkMetal]): AnyTensorImpl =
  ## Returns the erased face over `s`.
  ##
  ## Contract: the vtable's mixins resolve at the outermost instantiation scope.
  ## tensors.nim re-exports the typed conformant procs, so module users satisfy
  ## the interface conformance check.
  s.to(AnyTensorImpl)
