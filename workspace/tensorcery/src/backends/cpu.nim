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
## Recorded blobs reuse the k_add/k_mul record format: 8-byte address slots.
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
  workspace/tensorcery/src/kernels/k_mul

# Session state
# ------------------------------------------------------------

type
  CpuOpRunner* = proc(work: ptr CpuOpWork) {.cdecl.}
    ## Runs one recorded op's kernel, instantiated per element type at enqueue.

  CpuOpWork = object
    ## Decoded args of one recorded op plus its completion event.
    ## Heap allocated at drain, freed by the op task after it fires the event.
    run: CpuOpRunner
    outp, a, b: pointer
    addScalars: AddScalars
    mulScalars: MulScalars
    event: FlowEvent
    completions: ptr Atomic[int32]

  CpuOpTaskEnv = object
    work: ptr CpuOpWork

var
  sessionPool: ptr Threadpool
    ## Pool of the CPU session, created once, worker 0 is the calling thread.
  opRunners: Table[pointer, Table[int, CpuOpRunner]]
    ## Typed kernel runner per recorded op id, keyed by owning context.
    ## Registered at enqueue, dropped after the drain, the record stays data-only.
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
  # task->env layout is [task self-pointer][payload], the fn receives the header included
  # CpuOpTaskEnv payload sits behind the header
  let work = cast[ptr CpuOpTaskEnv](cast[int](env) + FlowvarHeaderSize)[].work
  work.run(work)
  flowEvent_trigger(work.event, workerContext(), scheduleFromEvent)
  discard work.completions[].fetchAdd(1)
  dealloc(work)

proc runAddOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## Add kernel execution through the parallelFor wrapper.
  var env = StridedAddEnv[T](
    outp: cast[ptr UncheckedArray[T]](work.outp),
    a: cast[ptr UncheckedArray[T]](work.a),
    b: cast[ptr UncheckedArray[T]](work.b),
    scalars: work.addScalars)
  syncScope:
    parallelFor(0, work.addScalars.n, env, stridedAddBody[T])

proc runMulOp[T](work: ptr CpuOpWork) {.cdecl.} =
  ## Scalar-multiply kernel execution through the parallelFor wrapper.
  var env = StridedMulEnv[T](
    outp: cast[ptr UncheckedArray[T]](work.outp),
    a: cast[ptr UncheckedArray[T]](work.a),
    scalars: work.mulScalars)
  syncScope:
    parallelFor(0, work.mulScalars.n, env, stridedMulBody[T])

proc decodeAddArgs(blob: openArray[byte]): AddArgs =
  ## Inverse of packAddArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(AddArgs))

proc decodeMulArgs(blob: openArray[byte]): MulArgs =
  ## Inverse of packMulArgs: rebuilds the args object from the recorded blob.
  copyMem(result.addr, blob[0].unsafeAddr, sizeof(MulArgs))

proc drainCpu(ctx: Ctx) {.nimcall.} =
  ## One synchronous drain of the record: one op task per entry.
  ## Flow events chain the tasks in record order.
  ## The root thread waits inside the drain scope until the last op completes,
  ## then drops the events and the runners.
  let wctx = workerContext()
  var scopeBarrier: ScopedBarrier
  scoped_barrier_initialize(scopeBarrier.addr)
  let suspendedScope = wctx.currentScope
  wctx.currentScope = scopeBarrier.addr

  var events: seq[FlowEvent]
  for entry in ctx.record:
    if entry.backend != bkCpu:
      continue
    var ev: FlowEvent
    flowEvent_initSingle(ev)
    events.add(ev)

    let work = cast[ptr CpuOpWork](alloc0(sizeof(CpuOpWork)))
    work[].event = ev
    work[].completions = completionCount.addr
    work[].run = opRunners[cast[pointer](ctx)][entry.output]
    case entry.kernel
    of opAdd:
      let args = decodeAddArgs(entry.blob)
      work[].outp = cast[pointer](args.outAddr)
      work[].a = cast[pointer](args.aAddr)
      work[].b = cast[pointer](args.bAddr)
      work[].addScalars = args.scalars
    of opMul:
      let args = decodeMulArgs(entry.blob)
      work[].outp = cast[pointer](args.outAddr)
      work[].a = cast[pointer](args.aAddr)
      work[].mulScalars = args.scalars

    var taskEnv = CpuOpTaskEnv(work: work)
    let task = task_newSpawnEnv(wctx.currentTask, scopeBarrier.addr,
                                cpuOpTask, taskEnv.addr, int32 sizeof CpuOpTaskEnv)
    if events.len == 1:
      schedule(wctx, task, false)
    elif not flowEvent_delayedUntil(task, events[^2]):
      # previous op already fired, schedule directly
      schedule(wctx, task, false)

  syncScopeWait(scopeBarrier.addr)
  wctx.currentScope = suspendedScope

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

proc padTo4(v: SmallSeq[4, int32], pad: int32): array[4, int32] =
  ## Pads to 4 axes with `pad`: shapes pad with 1s (the divmod chain stays safe), strides with 0s (padded axes contribute nothing).
  for i in 0 ..< 4:
    result[i] = if i < int(v.len): v[i] else: pad

proc add*[T](s: TensorImpl[T, bkCpu], other: AnyTensorImpl): AnyTensorImpl =
  ## Erased add: packs the op's args into one blob at enqueue, appends one record entry, returns the erased output tensor.
  let o = other.to(TensorImpl[T, bkCpu])
  let n = s.getNumel()
  var outStorage = newSeq[T](n)
  let blob = packAddArgs(AddArgs(
    outAddr: cast[uint64](outStorage[0].addr),
    aAddr: cast[uint64](s.storage[0].unsafeAddr),
    bAddr: cast[uint64](o.storage[0].unsafeAddr),
    scalars: AddScalars(
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      sb: padTo4(o.strides, 0),
      n: int32(n))))
  let opId = s.ctx.enqueue(opAdd, blob, LaunchCfg(), bkCpu)
  opRunners.mgetOrPut(cast[pointer](s.ctx), initTable[int, CpuOpRunner]())[opId] = runAddOp[T]
  TensorImpl[T, bkCpu](
    ctx: s.ctx,
    seqNo: opId,
    shape: s.shape,
    strides: contiguousStrides(s.shape),
    storage: outStorage).to(AnyTensorImpl)

proc mulScalar*[T](s: TensorImpl[T, bkCpu], scalar: float64): AnyTensorImpl =
  ## Erased scalar multiply: the scalar narrows to float32 at pack time,
  ## matching the k_mul record.
  let n = s.getNumel()
  var outStorage = newSeq[T](n)
  let blob = packMulArgs(MulArgs(
    outAddr: cast[uint64](outStorage[0].addr),
    aAddr: cast[uint64](s.storage[0].unsafeAddr),
    scalars: MulScalars(
      shape: padTo4(s.shape, 1),
      sa: padTo4(s.strides, 0),
      scalar: float32(scalar),
      n: int32(n))))
  let opId = s.ctx.enqueue(opMul, blob, LaunchCfg(), bkCpu)
  opRunners.mgetOrPut(cast[pointer](s.ctx), initTable[int, CpuOpRunner]())[opId] = runMulOp[T]
  TensorImpl[T, bkCpu](
    ctx: s.ctx,
    seqNo: opId,
    shape: s.shape,
    strides: contiguousStrides(s.shape),
    storage: outStorage).to(AnyTensorImpl)

proc readbackBytes*[T](s: TensorImpl[T, bkCpu]): seq[byte] =
  ## Copies the tensor's storage to host bytes, materializing pending ops first.
  ## Reading a pending tensor evals the context (the MLX rule).
  if s.ctx.isPending(s.seqNo):
    s.ctx.eval()
  result = newSeq[byte](s.getNumel() * sizeof(T))
  if result.len > 0:
    copyMem(result[0].addr, s.storage[0].unsafeAddr, result.len)

proc toErased*[T](s: TensorImpl[T, bkCpu]): AnyTensorImpl =
  ## Returns the erased face over `s`.
  ##
  ## Contract: the vtable's mixins resolve at the outermost instantiation scope.
  ## tensors.nim re-exports the typed conformant procs, so module users satisfy
  ## the interface conformance check.
  s.to(AnyTensorImpl)
