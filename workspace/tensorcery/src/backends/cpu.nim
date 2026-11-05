# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU backend: host storage for TensorImpl[T, bkCpu] and the flow-events drain.
##
## One session owns one threadpool, worker 0 is the calling thread.
## eval() chains one task per pending entry through flow events in record order:
##
## - each task runs its kernel through the parallelFor wrapper
## - the task fires its event, chaining the next op task, and counts one completion
## - the root thread waits inside the drain scope, so eval returns with every output readable
##
## Recorded blobs reuse the k_add/k_mul/k_mul2 record format: 8-byte address slots.
## For CPU the addresses are host pointers into the impl's seq storage.
## The scalar image of a multiply narrows the multiplier to float32, matching the k_mul record.

import std/tables
import std/cpuinfo
import std/atomics
import pkg/iface

# Relative import: the repo-root config puts the main checkout on the Nim path.
# Its older threadpool_compute wrapper lacks FlowEvent and would shadow the worktree wrapper.
import
  ../../../threadpool_compute/threadpool_compute,
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

# Session state
# ------------------------------------------------------------

type
  CpuOpRunner* = proc(work: ptr CpuOpWork) {.cdecl.}
    ## Runs one recorded op's kernel, instantiated per element type at enqueue.

  CpuOpWork = object
    ## Args of one recorded op plus its completion event.
    ## The blob is the args object's flat image, so the drain heap-copies it once
    ## and the runner casts it to the op's args type. Heap allocated at drain,
    ## the args copy and the work block are freed by the op task after it fires the event.
    run: CpuOpRunner
    args: pointer
    event: FlowEvent
    completions: ptr Atomic[int32]

  CpuOpTaskEnv = object
    work: ptr CpuOpWork

var
  sessionPool: ptr Threadpool
    ## Pool of the CPU session, created once, worker 0 is the calling thread.
  opRunners: Table[pointer, Table[int, CpuOpRunner]]
    ## Typed kernel runner per recorded op id, keyed by owning context.
    ## Registered at enqueue, dropped after the drain. The record stays data-only.
  completionCount*: Atomic[int32]
    ## Completed CPU ops since the last reset, one increment per executed op task.
    ## Tests reset it and assert the count after eval.

proc resetCompletionCount*() =
  completionCount.store(0)

# Kernels
# ------------------------------------------------------------

type StridedAddEnv[T] = object
  ## Per-iteration env of the CPU add kernel.
  outp, a, b: ptr UncheckedArray[T]
  scalars: AddScalars

proc stridedAddBody[T](idx: int32, env: ptr StridedAddEnv[T]) {.cdecl.} =
  ## CPU port of the k_add divmod chain:
  ## output[i] = a[aOff + strided(i)] + b[bOff + strided(i)]
  let sc = env.scalars
  let c3 = idx mod sc.shape[3]
  let t2 = idx div sc.shape[3]
  let c2 = t2 mod sc.shape[2]
  let t1 = t2 div sc.shape[2]
  let c1 = t1 mod sc.shape[1]
  let c0 = t1 div sc.shape[1]
  let ia = sc.aOff + (c0 * sc.sa[0] + c1 * sc.sa[1]) + c2 * sc.sa[2] + c3 * sc.sa[3]
  let ib = sc.bOff + (c0 * sc.sb[0] + c1 * sc.sb[1]) + c2 * sc.sb[2] + c3 * sc.sb[3]
  env.outp[][idx] = env.a[][ia] + env.b[][ib]

type StridedMulEnv[T] = object
  ## Per-iteration env of the CPU scalar-multiply kernel.
  outp, a: ptr UncheckedArray[T]
  scalars: MulScalars

proc stridedMulBody[T](idx: int32, env: ptr StridedMulEnv[T]) {.cdecl.} =
  ## CPU port of the k_mul divmod chain:
  ## output[i] = a[aOff + strided(i)] * scalar
  let sc = env.scalars
  let c3 = idx mod sc.shape[3]
  let t2 = idx div sc.shape[3]
  let c2 = t2 mod sc.shape[2]
  let t1 = t2 div sc.shape[2]
  let c1 = t1 mod sc.shape[1]
  let c0 = t1 div sc.shape[1]
  let ia = sc.aOff + (c0 * sc.sa[0] + c1 * sc.sa[1]) + c2 * sc.sa[2] + c3 * sc.sa[3]
  env.outp[][idx] = env.a[][ia] * T(sc.scalar)

type StridedMul2Env[T] = object
  ## Per-iteration env of the CPU tensor-product kernel.
  outp, a, b: ptr UncheckedArray[T]
  scalars: Mul2Scalars

proc stridedMul2Body[T](idx: int32, env: ptr StridedMul2Env[T]) {.cdecl.} =
  ## CPU port of the k_mul2 divmod chain:
  ## output[i] = a[aOff + strided(i)] * b[bOff + strided(i)]
  let sc = env.scalars
  let c3 = idx mod sc.shape[3]
  let t2 = idx div sc.shape[3]
  let c2 = t2 mod sc.shape[2]
  let t1 = t2 div sc.shape[2]
  let c1 = t1 mod sc.shape[1]
  let c0 = t1 div sc.shape[1]
  let ia = sc.aOff + (c0 * sc.sa[0] + c1 * sc.sa[1]) + c2 * sc.sa[2] + c3 * sc.sa[3]
  let ib = sc.bOff + (c0 * sc.sb[0] + c1 * sc.sb[1]) + c2 * sc.sb[2] + c3 * sc.sb[3]
  env.outp[][idx] = env.a[][ia] * env.b[][ib]

type StridedFillEnv[T] = object
  ## Per-iteration env of the CPU ones-fill kernel.
  outp: ptr UncheckedArray[T]
  scalars: FillScalars

proc stridedFillBody[T](idx: int32, env: ptr StridedFillEnv[T]) {.cdecl.} =
  ## CPU port of the k_fill kernel: output[i] = 1
  env.outp[][idx] = T(1)

# Op task and drain
# ------------------------------------------------------------

proc scheduleFromEvent(queue: pointer, task: ptr Task) {.cdecl.} =
  ## Enqueue callback of flow-event triggers: schedules the drained task
  ## on the triggering worker's deque.
  schedule(cast[ptr WorkerContext](queue), task, false)

proc cpuOpTask(env: pointer) {.cdecl.} =
  ## Body of one recorded op on a pool worker.
  ##
  ## - runs the kernel through the parallelFor wrapper
  ## - fires the op's flow event, counts one completion
  # task->env layout is [task self-pointer][payload]. The fn receives the header included
  # CpuOpTaskEnv payload sits behind the header
  let work = cast[ptr CpuOpTaskEnv](cast[int](env) + FlowvarHeaderSize)[].work
  work.run(work)
  flowEvent_trigger(work.event, workerContext(), scheduleFromEvent)
  discard work.completions[].fetchAdd(1)
  dealloc(work.args)
  dealloc(work)

proc runAddOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## Add kernel execution through the parallelFor wrapper.
  let args = cast[ptr AddArgs](work.args)
  var env = StridedAddEnv[T](
    outp: cast[ptr UncheckedArray[T]](args.outAddr),
    a: cast[ptr UncheckedArray[T]](args.aAddr),
    b: cast[ptr UncheckedArray[T]](args.bAddr),
    scalars: args.scalars)
  syncScope:
    parallelFor(0, args.scalars.n, env, stridedAddBody[T])

proc runMulOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## Scalar-multiply kernel execution through the parallelFor wrapper.
  let args = cast[ptr MulArgs](work.args)
  var env = StridedMulEnv[T](
    outp: cast[ptr UncheckedArray[T]](args.outAddr),
    a: cast[ptr UncheckedArray[T]](args.aAddr),
    scalars: args.scalars)
  syncScope:
    parallelFor(0, args.scalars.n, env, stridedMulBody[T])

proc runMul2Op[T](work: ptr CpuOpWork) {.cdecl.} =
  ## Tensor-product kernel execution through the parallelFor wrapper.
  let args = cast[ptr Mul2Args](work.args)
  var env = StridedMul2Env[T](
    outp: cast[ptr UncheckedArray[T]](args.outAddr),
    a: cast[ptr UncheckedArray[T]](args.aAddr),
    b: cast[ptr UncheckedArray[T]](args.bAddr),
    scalars: args.scalars)
  syncScope:
    parallelFor(0, args.scalars.n, env, stridedMul2Body[T])

proc runFillOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## Ones-fill kernel execution through the parallelFor wrapper.
  let args = cast[ptr FillArgs](work.args)
  var env = StridedFillEnv[T](
    outp: cast[ptr UncheckedArray[T]](args.outAddr),
    scalars: args.scalars)
  syncScope:
    parallelFor(0, args.scalars.n, env, stridedFillBody[T])

proc runGatherOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## Gather kernel execution through the parallelFor wrapper.
  let args = cast[ptr GatherArgs](work.args)
  var env = GatherEnv[T](
    outp: cast[ptr UncheckedArray[T]](args.outAddr),
    w: cast[ptr UncheckedArray[T]](args.wAddr),
    idx: cast[ptr UncheckedArray[int32]](args.idxAddr),
    scalars: args.scalars)
  syncScope:
    parallelFor(0, args.scalars.n, env, gatherBody[T])

proc runLayerNormOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## Layernorm kernel execution through the parallelFor wrapper, one d-element set per iteration.
  let args = cast[ptr LayerNormArgs](work.args)
  var env = LayerNormEnv[T](
    outp: cast[ptr UncheckedArray[T]](args.outAddr),
    x: cast[ptr UncheckedArray[T]](args.xAddr),
    w: cast[ptr UncheckedArray[T]](args.wAddr),
    b: cast[ptr UncheckedArray[T]](args.bAddr),
    scalars: args.scalars)
  syncScope:
    parallelFor(0, args.scalars.rows, env, layernormBody[T])

proc runGeluOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## Gelu kernel execution through the parallelFor wrapper.
  let args = cast[ptr GeluArgs](work.args)
  var env = GeluEnv[T](
    outp: cast[ptr UncheckedArray[T]](args.outAddr),
    a: cast[ptr UncheckedArray[T]](args.aAddr),
    scalars: args.scalars)
  syncScope:
    parallelFor(0, args.scalars.n, env, geluBody[T])

proc runSoftmaxOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## Softmax kernel execution through the parallelFor wrapper, one d-element set per iteration.
  let args = cast[ptr SoftmaxArgs](work.args)
  var env = SoftmaxEnv[T](
    outp: cast[ptr UncheckedArray[T]](args.outAddr),
    x: cast[ptr UncheckedArray[T]](args.xAddr),
    scalars: args.scalars)
  syncScope:
    parallelFor(0, args.scalars.rows, env, softmaxBody[T])

proc runCrossEntropyOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## Cross entropy kernel execution through the parallelFor wrapper, one logit set per iteration.
  let args = cast[ptr CrossEntropyArgs](work.args)
  var env = CrossEntropyEnv[T](
    loss: cast[ptr UncheckedArray[T]](args.lossAddr),
    logits: cast[ptr UncheckedArray[T]](args.logitsAddr),
    target: cast[ptr UncheckedArray[int32]](args.targetAddr),
    scalars: args.scalars)
  syncScope:
    parallelFor(0, args.scalars.rows, env, crossEntropyBody[T])

proc runWhereOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## Where kernel execution through the parallelFor wrapper.
  let args = cast[ptr WhereArgs](work.args)
  var env = WhereEnv[T](
    outp: cast[ptr UncheckedArray[T]](args.outAddr),
    cond: cast[ptr UncheckedArray[T]](args.condAddr),
    a: cast[ptr UncheckedArray[T]](args.aAddr),
    b: cast[ptr UncheckedArray[T]](args.bAddr),
    scalars: args.scalars)
  syncScope:
    parallelFor(0, args.scalars.n, env, whereBody[T])

proc runSgdOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## In-place sgd update execution through the parallelFor wrapper.
  let args = cast[ptr SgdArgs](work.args)
  var env = SgdEnv[T](
    p: cast[ptr UncheckedArray[T]](args.pAddr),
    g: cast[ptr UncheckedArray[T]](args.gAddr),
    scalars: args.scalars)
  syncScope:
    parallelFor(0, args.scalars.n, env, sgdBody[T])

proc runAdamOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## In-place adam update execution through the parallelFor wrapper.
  let args = cast[ptr AdamArgs](work.args)
  var env = AdamEnv[T](
    p: cast[ptr UncheckedArray[T]](args.pAddr),
    g: cast[ptr UncheckedArray[T]](args.gAddr),
    m: cast[ptr UncheckedArray[T]](args.mAddr),
    v: cast[ptr UncheckedArray[T]](args.vAddr),
    scalars: args.scalars)
  syncScope:
    parallelFor(0, args.scalars.n, env, adamBody[T])

proc cpuRunner[T](kernel: KernelId): CpuOpRunner =
  ## Typed kernel runner per recorded op, instantiated at enqueue.
  case kernel
  of opAdd: result = runAddOp[T]
  of opMul: result = runMulOp[T]
  of opMul2: result = runMul2Op[T]
  of opFill: result = runFillOp[T]
  of opGather: result = runGatherOp[T]
  of opLayerNorm:
    when T is SomeFloat:
      result = runLayerNormOp[T]
    else:
      # int32 tensors carry indices and targets only. The float math never runs
      result = nil
  of opGelu:
    when T is SomeFloat:
      result = runGeluOp[T]
    else:
      result = nil
  of opSoftmax:
    when T is SomeFloat:
      result = runSoftmaxOp[T]
    else:
      result = nil
  of opCrossEntropy:
    when T is SomeFloat:
      result = runCrossEntropyOp[T]
    else:
      result = nil
  of opWhere: result = runWhereOp[T]
  of opSgd:
    when T is SomeFloat:
      result = runSgdOp[T]
    else:
      result = nil
  of opAdam:
    when T is SomeFloat:
      result = runAdamOp[T]
    else:
      result = nil
  of opReshape, opPermute, opSlice:
    # pure metadata, never dispatches
    result = nil

proc drainCpu(ctx: Ctx) {.nimcall.} =
  ## One synchronous drain of the record: one op task per pending entry.
  ## Flow events chain the tasks in record order.
  ## The root thread waits inside the drain scope until the last op completes,
  ## then drops the events and the runners.
  var events: seq[FlowEvent]
  syncScope:
    let wctx = workerContext()
    for entry in ctx.record:
      if entry.backend != bkCpu or not ctx.isPending(entry.id):
        continue
      if isShapeOp(entry.kernel):
        # pure metadata, never dispatches
        continue
      var ev: FlowEvent
      flowEvent_initSingle(ev)
      events.add(ev)

      let work = cast[ptr CpuOpWork](alloc0(sizeof(CpuOpWork)))
      work[].event = ev
      work[].completions = completionCount.addr
      work[].run = opRunners[cast[pointer](ctx)][entry.id]
      # entry.blob is the args object's flat image and the runner casts it back
      work[].args = alloc0(entry.blob.len)
      copyMem(work[].args, entry.blob[0].unsafeAddr, entry.blob.len)

      var taskEnv = CpuOpTaskEnv(work: work)
      let task = task_newSpawnEnv(wctx.currentTask, wctx.currentScope,
                                  cpuOpTask, taskEnv.addr, int32 sizeof CpuOpTaskEnv)
      if events.len == 1:
        schedule(wctx, task, false)
      elif not flowEvent_delayedUntil(task, events[^2]):
        # previous op already fired, schedule directly
        schedule(wctx, task, false)

  for ev in events.mitems:
    flowEvent_unref(ev)
  opRunners.del(cast[pointer](ctx))

proc ensureCpu*(ctx: Ctx) =
  ## One-time pool setup plus ctx wiring, idempotent.
  ## The pool persists for the process lifetime.
  if sessionPool.isNil:
    sessionPool = threadpool_new(int32 countProcessors())
  ctx.drains[bkCpu] = drainCpu

# Erased face
# ------------------------------------------------------------

proc record[T](s: TensorImpl[T, bkCpu], kernel: KernelId, blob: seq[byte],
    outStorage: sink seq[T], outShape: SmallSeq[4, int32], requiresGrad, isGrad: bool,
    inA, inB, inC: int): TensorImpl[T, bkCpu] =
  ## Enqueues one record entry, registers the typed kernel runner and wraps the freshly
  ## allocated output storage.
  ##
  ## `outStorage` is a sink: the impl takes ownership of the caller's buffer, so the addresses
  ## that the blob packed stay the addresses that the impl owns. A borrowed parameter would
  ## deep-copy the seq into the impl and leave every packed address dangling
  ## once the caller's buffer is freed.
  let outTid = s.ctx.newTid()
  let opId = s.ctx.enqueue(kernel, blob, LaunchCfg(), bkCpu, outTid, inA, inB, inC, 0, isGrad)
  opRunners.mgetOrPut(cast[pointer](s.ctx), initTable[int, CpuOpRunner]())[opId] =
      cpuRunner[T](kernel)
  result = TensorImpl[T, bkCpu](
    ctx: s.ctx,
    tid: outTid,
    seqNo: opId,
    requiresGrad: requiresGrad,
    shape: outShape,
    strides: contiguousStrides(outShape),
    storage: outStorage)

proc padTo4(v: SmallSeq[4, int32], pad: int32): array[4, int32] =
  ## Pads to 4 axes with `pad`: shapes pad with 1s (the divmod chain stays safe), strides with 0s (padded axes contribute nothing).
  for i in 0 ..< 4:
    result[i] = if i < int(v.len): v[i] else: pad

proc add*[T](s: TensorImpl[T, bkCpu], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased add: packs the op's args into one blob at enqueue, appends one record entry, returns the erased output tensor.
  ## The output requires grad when either input does.
  let o = other.to(TensorImpl[T, bkCpu])
  let n = s.getNumel()
  var outStorage = newSeq[T](n)
  let blob = packAddArgs(AddArgs(
    outAddr: cast[uint64](outStorage[0].addr),
    aAddr: cast[uint64](s.storage[0].unsafeAddr),
    bAddr: cast[uint64](o.storage[0].unsafeAddr),
    scalars: AddScalars(
      aOff: int32(s.offset),
      bOff: int32(o.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  s.record(opAdd, blob, outStorage, s.shape,
      s.requiresGrad or o.requiresGrad, false, s.tid, o.tid, 0).to(AnyTensorImpl)

proc mul*[T](s: TensorImpl[T, bkCpu], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased tensor product: packs the op's args into one blob at enqueue,
  ## appends one record entry, returns the erased output tensor.
  ## The output requires grad when either input does.
  let o = other.to(TensorImpl[T, bkCpu])
  let n = s.getNumel()
  var outStorage = newSeq[T](n)
  let blob = packMul2Args(Mul2Args(
    outAddr: cast[uint64](outStorage[0].addr),
    aAddr: cast[uint64](s.storage[0].unsafeAddr),
    bAddr: cast[uint64](o.storage[0].unsafeAddr),
    scalars: Mul2Scalars(
      aOff: int32(s.offset),
      bOff: int32(o.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  s.record(opMul2, blob, outStorage, s.shape,
      s.requiresGrad or o.requiresGrad, false, s.tid, o.tid, 0).to(AnyTensorImpl)

proc mulScalar*[T](s: TensorImpl[T, bkCpu], scalar: float64): AnyTensorImpl =
  ## Erased scalar multiply: the scalar narrows to float32 at pack time, matching the k_mul record.
  ## The output requires grad when the input does.
  let n = s.getNumel()
  var outStorage = newSeq[T](n)
  let blob = packMulArgs(MulArgs(
    outAddr: cast[uint64](outStorage[0].addr),
    aAddr: cast[uint64](s.storage[0].unsafeAddr),
    scalars: MulScalars(
      aOff: int32(s.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      scalar: float32(scalar),
      n: int32(n))))
  s.record(opMul, blob, outStorage, s.shape,
      s.requiresGrad, false, s.tid, 0, 0).to(AnyTensorImpl)

proc addNoGrad*[T](s: TensorImpl[T, bkCpu], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased grad-chain add: runs the k_add record, grad flag off.
  ## backward() adds fan-out accumulation through it.
  let o = other.to(TensorImpl[T, bkCpu])
  let n = s.getNumel()
  var outStorage = newSeq[T](n)
  let blob = packAddArgs(AddArgs(
    outAddr: cast[uint64](outStorage[0].addr),
    aAddr: cast[uint64](s.storage[0].unsafeAddr),
    bAddr: cast[uint64](o.storage[0].unsafeAddr),
    scalars: AddScalars(
      aOff: int32(s.offset),
      bOff: int32(o.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  s.record(opAdd, blob, outStorage, s.shape,
      false, true, s.tid, o.tid, 0).to(AnyTensorImpl)

proc mulNoGrad*[T](s: TensorImpl[T, bkCpu], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased grad-chain tensor product: runs the k_mul2 record, grad flag off.
  ## backward() builds the mul vjp from it.
  let o = other.to(TensorImpl[T, bkCpu])
  let n = s.getNumel()
  var outStorage = newSeq[T](n)
  let blob = packMul2Args(Mul2Args(
    outAddr: cast[uint64](outStorage[0].addr),
    aAddr: cast[uint64](s.storage[0].unsafeAddr),
    bAddr: cast[uint64](o.storage[0].unsafeAddr),
    scalars: Mul2Scalars(
      aOff: int32(s.offset),
      bOff: int32(o.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  s.record(opMul2, blob, outStorage, s.shape,
      false, true, s.tid, o.tid, 0).to(AnyTensorImpl)

proc mulScalarNoGrad*[T](s: TensorImpl[T, bkCpu], scalar: float64): AnyTensorImpl =
  ## Erased grad-chain scalar multiply: runs the k_mul record, grad flag off.
  ## backward() builds the mulScalar vjp from it.
  let n = s.getNumel()
  var outStorage = newSeq[T](n)
  let blob = packMulArgs(MulArgs(
    outAddr: cast[uint64](outStorage[0].addr),
    aAddr: cast[uint64](s.storage[0].unsafeAddr),
    scalars: MulScalars(
      aOff: int32(s.offset),
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      scalar: float32(scalar),
      n: int32(n))))
  s.record(opMul, blob, outStorage, s.shape,
      false, true, s.tid, 0, 0).to(AnyTensorImpl)

proc onesNoGrad*[T](s: TensorImpl[T, bkCpu]): AnyTensorImpl =
  ## Erased grad-chain ones fill: runs the k_fill record, grad flag off.
  ## backward() seeds the cotangent map from it.
  let n = s.getNumel()
  var outStorage = newSeq[T](n)
  let blob = packFillArgs(FillArgs(
    outAddr: cast[uint64](outStorage[0].addr),
    scalars: FillScalars(n: int32(n))))
  s.record(opFill, blob, outStorage, s.shape,
      false, true, 0, 0, 0).to(AnyTensorImpl)

proc readbackBytes*[T](s: TensorImpl[T, bkCpu]): seq[byte] =
  ## Returns:
  ## - the tensor as host bytes in logical order, last axis fastest
  ##
  ## A view gathers through its strides and offset. Reading a pending tensor
  ## evals the context first (the MLX rule).
  if s.ctx.isPending(s.seqNo):
    s.ctx.eval()
  if s.getNumel() == 0:
    return newSeq[byte](0)
  let base = cast[ptr T](cast[int](s.storage[0].unsafeAddr) + s.offset * sizeof(T))
  stridedBytes[T](base, s.shape, s.strides)

# Microgpt op set, CPU leg
# ------------------------------------------------------------

proc gather*[T](s: TensorImpl[T, bkCpu], indices: AnyTensorImpl): AnyTensorImpl =
  ## Erased embedding gather: one embedding vector per index over the rank-2 table.
  ## The output requires grad when the table does. The int32 indices never do.
  ##
  ## Raises AssertionDefect for a non rank-2 table or any non-contiguous operand.
  let idx = indices.to(TensorImpl[int32, bkCpu])
  doAssert int(s.shape.len) == 2,
      "gather: the table must be rank 2 (vocabulary, embedding dim), got rank " & $s.shape.len
  doAssert isContiguous(s.shape, s.strides), "gather: non-contiguous table refused"
  doAssert isContiguous(idx.shape, idx.strides), "gather: non-contiguous index tensor refused"
  let e = int32(s.shape[int(s.shape.len) - 1])
  let n = idx.getNumel() * int(e)
  var outStorage = newSeq[T](n)
  let blob = packGatherArgs(GatherArgs(
    outAddr: cast[uint64](outStorage[0].addr),
    wAddr: cast[uint64](s.storage[0].unsafeAddr),
    idxAddr: cast[uint64](idx.storage[0].unsafeAddr),
    scalars: GatherScalars(
      wOff: int32(s.offset), idxOff: int32(idx.offset), e: e, n: int32(n))))
  s.record(opGather, blob, outStorage, appendDim(idx.shape, e),
      s.requiresGrad, false, s.tid, idx.tid, 0).to(AnyTensorImpl)

proc layernorm*[T](s: TensorImpl[T, bkCpu], w, b: AnyTensorImpl, eps: float64): AnyTensorImpl =
  ## Erased last-axis layernorm over `s` with the learned scale `w` and shift `b`.
  ## The output requires grad when any operand does.
  ##
  ## Raises AssertionDefect when the scale or shift does not hold one element per
  ## normalized axis or is non-contiguous.
  let wImpl = w.to(TensorImpl[T, bkCpu])
  let bImpl = b.to(TensorImpl[T, bkCpu])
  let d = int(s.shape[int(s.shape.len) - 1])
  let rows = s.getNumel() div d
  doAssert wImpl.getNumel() == d and bImpl.getNumel() == d,
      "layernorm: scale and shift must hold one element per normalized axis, " &
      $wImpl.getNumel() & " and " & $bImpl.getNumel() & " for axis size " & $d
  doAssert isContiguous(wImpl.shape, wImpl.strides) and isContiguous(bImpl.shape, bImpl.strides),
      "layernorm: non-contiguous scale or shift refused"
  var outStorage = newSeq[T](s.getNumel())
  let blob = packLayerNormArgs(LayerNormArgs(
    outAddr: cast[uint64](outStorage[0].addr),
    xAddr: cast[uint64](s.storage[0].unsafeAddr),
    wAddr: cast[uint64](wImpl.storage[0].unsafeAddr),
    bAddr: cast[uint64](bImpl.storage[0].unsafeAddr),
    scalars: LayerNormScalars(
      xOff: int32(s.offset), wOff: int32(wImpl.offset), bOff: int32(bImpl.offset),
      xShape: padTo4(s.shape, 1), xs: padTo4(s.strides, 0),
      d: int32(d), rows: int32(rows), xStep: int32(s.strides[int(s.shape.len) - 1]),
      eps: float32(eps))))
  s.record(opLayerNorm, blob, outStorage, s.shape,
      s.requiresGrad or wImpl.requiresGrad or bImpl.requiresGrad, false,
      s.tid, wImpl.tid, bImpl.tid).to(AnyTensorImpl)

proc gelu*[T](s: TensorImpl[T, bkCpu]): AnyTensorImpl =
  ## Erased tanh-approximate gelu over `s`. The output requires grad when `s` does.
  let n = s.getNumel()
  var outStorage = newSeq[T](n)
  let blob = packGeluArgs(GeluArgs(
    outAddr: cast[uint64](outStorage[0].addr),
    aAddr: cast[uint64](s.storage[0].unsafeAddr),
    scalars: GeluScalars(aOff: int32(s.offset), n: int32(n))))
  s.record(opGelu, blob, outStorage, s.shape,
      s.requiresGrad, false, s.tid, 0, 0).to(AnyTensorImpl)

proc softmax*[T](s: TensorImpl[T, bkCpu]): AnyTensorImpl =
  ## Erased last-axis softmax over `s`. The output requires grad when `s` does.
  let dAxis = int(s.shape[int(s.shape.len) - 1])
  let rows = s.getNumel() div dAxis
  var outStorage = newSeq[T](s.getNumel())
  let blob = packSoftmaxArgs(SoftmaxArgs(
    outAddr: cast[uint64](outStorage[0].addr),
    xAddr: cast[uint64](s.storage[0].unsafeAddr),
    scalars: SoftmaxScalars(
      xOff: int32(s.offset),
      xShape: padTo4(s.shape, 1), xs: padTo4(s.strides, 0),
      d: int32(dAxis), rows: int32(rows), xStep: int32(s.strides[int(s.shape.len) - 1]))))
  s.record(opSoftmax, blob, outStorage, s.shape,
      s.requiresGrad, false, s.tid, 0, 0).to(AnyTensorImpl)

proc crossEntropy*[T](s: TensorImpl[T, bkCpu], targets: AnyTensorImpl): AnyTensorImpl =
  ## Erased cross entropy: one loss per set of `v` logits over the last axis of `s`,
  ## the int32 `targets` name the class per set. The output requires grad when the logits do.
  ##
  ## Raises AssertionDefect for a non-contiguous target view.
  let tgt = targets.to(TensorImpl[int32, bkCpu])
  doAssert isContiguous(tgt.shape, tgt.strides), "cross entropy: non-contiguous target view refused"
  let v = int(s.shape[int(s.shape.len) - 1])
  let rows = s.getNumel() div v
  var outStorage = newSeq[T](rows)
  let blob = packCrossEntropyArgs(CrossEntropyArgs(
    lossAddr: cast[uint64](outStorage[0].addr),
    logitsAddr: cast[uint64](s.storage[0].unsafeAddr),
    targetAddr: cast[uint64](tgt.storage[0].unsafeAddr),
    scalars: CrossEntropyScalars(
      logitsOff: int32(s.offset), targetOff: int32(tgt.offset),
      logitsShape: padTo4(s.shape, 1), logitsStrides: padTo4(s.strides, 0),
      v: int32(v), rows: int32(rows), logitStep: int32(s.strides[int(s.shape.len) - 1]))))
  s.record(opCrossEntropy, blob, outStorage, dropLastDim(s.shape),
      s.requiresGrad, false, s.tid, tgt.tid, 0).to(AnyTensorImpl)

proc where*[T](s: TensorImpl[T, bkCpu], cond, other: AnyTensorImpl): AnyTensorImpl =
  ## Erased elementwise select with `s` on the true branch and `other` on the false branch.
  ## The output requires grad when any operand does.
  ##
  ## Raises AssertionDefect on a shape mismatch, no broadcasting.
  let c = cond.to(TensorImpl[T, bkCpu])
  let o = other.to(TensorImpl[T, bkCpu])
  doAssert sameMeta(s.shape, c.shape) and sameMeta(s.shape, o.shape),
      "where: shape mismatch " & $s.shape & " vs " & $c.shape & " vs " & $o.shape &
      " refused, no broadcasting"
  let n = s.getNumel()
  var outStorage = newSeq[T](n)
  let blob = packWhereArgs(WhereArgs(
    outAddr: cast[uint64](outStorage[0].addr),
    condAddr: cast[uint64](c.storage[0].unsafeAddr),
    aAddr: cast[uint64](s.storage[0].unsafeAddr),
    bAddr: cast[uint64](o.storage[0].unsafeAddr),
    scalars: WhereScalars(
      condOff: int32(c.offset), aOff: int32(s.offset), bOff: int32(o.offset),
      n: int32(n))))
  s.record(opWhere, blob, outStorage, s.shape,
      s.requiresGrad or c.requiresGrad or o.requiresGrad, false,
      c.tid, s.tid, o.tid).to(AnyTensorImpl)

proc sgdUpdate*[T](s: TensorImpl[T, bkCpu], grad: AnyTensorImpl, lr: float64): void =
  ## Erased in-place sgd update: `s` moves by `-lr * grad`.
  ## The entry's output id is `s`'s tensor id and `s`'s seqNo advances to the entry,
  ## so reading the parameters materializes the update first (the MLX rule).
  ##
  ## Raises AssertionDefect for a shape mismatch or a non-contiguous operand.
  let g = grad.to(TensorImpl[T, bkCpu])
  doAssert sameMeta(s.shape, g.shape),
      "sgd: shape mismatch " & $s.shape & " vs " & $g.shape & " refused"
  doAssert isContiguous(s.shape, s.strides) and isContiguous(g.shape, g.strides),
      "sgd: non-contiguous parameter or gradient view refused"
  let n = s.getNumel()
  let blob = packSgdArgs(SgdArgs(
    pAddr: cast[uint64](s.storage[0].unsafeAddr),
    gAddr: cast[uint64](g.storage[0].unsafeAddr),
    scalars: SgdScalars(lr: float32(lr), pOff: int32(s.offset), gOff: int32(g.offset), n: int32(n))))
  let opId = s.ctx.enqueue(opSgd, blob, LaunchCfg(), bkCpu, s.tid, s.tid, g.tid)
  opRunners.mgetOrPut(cast[pointer](s.ctx), initTable[int, CpuOpRunner]())[opId] =
      cpuRunner[T](opSgd)
  s.seqNo = opId

proc adamUpdate*[T](s: TensorImpl[T, bkCpu], grad, m, v: AnyTensorImpl, pack: AdamScalars): void =
  ## Erased in-place adam update over `s` with gradient `grad` and the persistent moments `m` and `v`.
  ##
  ## The pack lists the learning rate, decay rates, epsilon and bias corrections for the current step, all folded in by the caller.
  ##
  ## The entry's output id is `s`'s tensor id and `s`'s seqNo advances to the entry,
  ## so reading the parameters materializes the update first (the MLX rule).
  ##
  ## Raises AssertionDefect for a shape mismatch or a non-contiguous operand.
  let g = grad.to(TensorImpl[T, bkCpu])
  let mImpl = m.to(TensorImpl[T, bkCpu])
  let vImpl = v.to(TensorImpl[T, bkCpu])
  doAssert sameMeta(s.shape, g.shape) and sameMeta(s.shape, mImpl.shape) and
      sameMeta(s.shape, vImpl.shape),
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
    pAddr: cast[uint64](s.storage[0].unsafeAddr),
    gAddr: cast[uint64](g.storage[0].unsafeAddr),
    mAddr: cast[uint64](mImpl.storage[0].unsafeAddr),
    vAddr: cast[uint64](vImpl.storage[0].unsafeAddr),
    scalars: scalars))
  let opId = s.ctx.enqueue(opAdam, blob, LaunchCfg(), bkCpu, s.tid, s.tid, g.tid, mImpl.tid, vImpl.tid)
  opRunners.mgetOrPut(cast[pointer](s.ctx), initTable[int, CpuOpRunner]())[opId] =
      cpuRunner[T](opAdam)
  s.seqNo = opId


proc toErased*[T](s: TensorImpl[T, bkCpu]): AnyTensorImpl =
  ## Returns the erased face over `s`.
  ##
  ## Contract: the vtable's mixins resolve at the outermost instantiation scope.
  ## tensors.nim re-exports the typed conformant procs, so module users satisfy
  ## the interface conformance check.
  s.to(AnyTensorImpl)
