# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/os

const tpcRoot = currentSourcePath.parentDir

{.emit: "/*TYPESECTION*/\n#include \"" & tpcRoot / "src" / "threadpool_runtime.c" & "\"".}

{.passc: "-I\"" & tpcRoot & "\"".}
{.passc: "-I\"" & tpcRoot / "src" / "threadpool" & "\"".}
{.passc: "-I\"" & tpcRoot / "include" & "\"".}

# Every generated C file sees the C API declarations, emitted-C launch sites
# call the runtime directly
{.passc: "-include \"" & tpcRoot / "include" / "threadpool_compute.h" & "\"".}

when not defined(windows):
  {.passL: "-lpthread".}

const tpcHeader = "include/threadpool_compute.h"
{.pragma: tpc, importc, cdecl, header: tpcHeader.}

# ################################################## #
#                 threadpool_compute API             #
# ################################################## #

const FlowvarHeaderSize* = sizeof(pointer)
  ## Bytes of task self-pointer prepended to the environment block,
  ## TPC_FLOWVAR_HEADER_SIZE, a task body reads its payload after it

type
  Threadpool* {.importc: "tpc_threadpool", header: tpcHeader.} = object
  Task* {.importc: "tpc_task", header: tpcHeader.} = object
  Taskqueue* {.importc: "tpc_taskqueue", header: tpcHeader.} = object

  WorkerContext* {.importc: "tpc_worker_context", header: tpcHeader.} = object
    id*: int32
    threadpool*: ptr Threadpool
    taskqueue*: ptr Taskqueue
    currentTask*: ptr Task
    currentScope*: ptr ScopedBarrier

  ScopedBarrier* {.importc: "tpc_scoped_barrier", header: tpcHeader.} = object

  Flowvar* {.importc: "tpc_flowvar", header: tpcHeader.} = object
    task*: ptr Task

  TaskFn* = proc(env: pointer) {.cdecl.}
    ## A task body with the C `tpc_task_fn` signature

  RangeFn* = proc(idx: int32, env: pointer) {.cdecl.}
    ## A parallel-for iteration body with the C `tpc_range_fn` signature

const RootTask* = cast[ptr Task](0xEFFACED0)
  ## Root sentinel task, TPC_ROOT_TASK

# Pool lifecycle
# ------------------------------------------------------------

proc threadpool_new*(numThreads: int32): ptr Threadpool {.tpc, importc: "tpc_threadpool_new".}
  ## Creates a pool where the calling thread becomes worker 0

proc threadpool_shutdown*(tp: ptr Threadpool) {.tpc, importc: "tpc_threadpool_shutdown".}
  ## Finishes pending work, signals termination and joins the workers

proc threadpool_current*(): ptr Threadpool {.tpc, importc: "tpc_threadpool_current".}
  ## Pool of the calling thread

proc syncAll*(tp: ptr Threadpool) {.tpc, importc: "tpc_syncAll".}
  ## Runs pending tasks, then waits until every other worker is parked

proc workerContext*(): ptr WorkerContext {.tpc, importc: "tpc_workerContext".}
  ## Worker context of the calling thread, worker 0 is the root thread

# Task and flowvar primitives
# ------------------------------------------------------------

proc task_newSpawnEnv*(parent: ptr Task, barrier: ptr ScopedBarrier, fn: TaskFn, env: pointer, envSize: int32): ptr Task {.tpc, importc: "tpc_task_newSpawnEnv".}
  ## Allocates a spawn task with an envSize-byte environment block,
  ## copied from env with the self-pointer prepended

proc task_newLoopEnv*(parent: ptr Task, barrier: ptr ScopedBarrier, fn: TaskFn, env: pointer, envSize: int32, start, stop, stride: int32, isFirstIter: bool): ptr Task {.tpc, importc: "tpc_task_newLoopEnv".}
  ## Allocates a loop task over [start, stop) with the given stride

proc task_readyWith*(task: ptr Task, result: pointer, resultSize: csize_t) {.tpc, importc: "tpc_task_readyWith".}
  ## Sends the result from the task body to its waiter

proc flowvar_new*(task: ptr Task): Flowvar {.tpc, importc: "tpc_flowvar_new".}
  ## Turns a freshly spawned task into its flowvar view

proc flowvar_sync*(dst: pointer, fv: ptr Flowvar, resultSize: csize_t) {.tpc, importc: "tpc_flowvar_sync".}
  ## Waits for the flowvar and copies its result into dst

proc flowvar_isReady*(fv: Flowvar): bool {.tpc, importc: "tpc_flowvar_isReady".}
  ## True when sync would not block

proc schedule*(ctx: ptr WorkerContext, task: ptr Task, forceWake: bool) {.tpc, importc: "tpc_schedule".}
  ## Publishes the task on the calling worker's deque

proc scoped_barrier_initialize*(sb: ptr ScopedBarrier) {.tpc, importc: "tpc_scoped_barrier_initialize".}
  ## Prepares a barrier for the current scope

proc syncScopeWait*(sb: ptr ScopedBarrier) {.tpc, importc: "tpc_syncScopeWait".}
  ## Waits until the scope's descendant tasks complete

proc completeFuture*(fv: ptr Flowvar, parentResult: pointer, resultSize: csize_t) {.tpc, importc: "tpc_completeFuture".}
  ## Waits until the task behind the flowvar completes and copies
  ## resultSize bytes from its result slot into parentResult.
  ## Task memory outlives the call and flowvar_sync frees the task
  ## after the copy, calling this alone leaks the allocation.

# Spawn helpers
# ------------------------------------------------------------

proc spawnVoid*[T](fn: TaskFn, env: var T) =
  ## Spawns a fire-and-forget task, copying the env block
  let ctx = workerContext()
  let task = task_newSpawnEnv(ctx.currentTask, ctx.currentScope, fn,
                              env.addr, int32 sizeof T)
  schedule(ctx, task, false)

proc spawn*[T](fn: TaskFn, env: var T): Flowvar =
  ## Spawns a task and returns its flowvar, copying the env block
  let ctx = workerContext()
  let task = task_newSpawnEnv(ctx.currentTask, ctx.currentScope, fn,
                              env.addr, int32 sizeof T)
  result = flowvar_new(task)
  schedule(ctx, task, false)

proc sync*[T](fv: Flowvar): T =
  ## Waits for the flowvar and yields its result
  flowvar_sync(result.addr, fv.unsafeAddr, csize_t sizeof T)

# ################################################## #
#                 flow events                        #
# ################################################## #
# Dataflow task dependencies, Nim wrapper over include/flow_events.h.
# A flow event is a delayed input: consumer tasks defer until a producer
# triggers the event.

const tpcFlowEventsHeader = "include/flow_events.h"
{.pragma: tpcev, importc, cdecl, header: tpcFlowEventsHeader.}

type
  FlowEvent* {.importc: "tpc_flow_event", header: tpcFlowEventsHeader.} = object
    ## Reference-counted event handle, C `tpc_flow_event`
    e*: ptr FlowEventObj

  FlowEventObj* {.importc: "tpc_event", header: tpcFlowEventsHeader.} = object
    ## Opaque event state behind a FlowEvent handle

  FlowEventDep* {.importc: "tpc_flow_event_dep", header: tpcFlowEventsHeader.} = object
    ## One dependency of a multi-dependency task: the awaited event,
    ## and for an iteration event the iteration index
    event*: FlowEvent
    index*: int32

  FlowEventEnqueue* = proc(queue: pointer, task: ptr Task) {.cdecl.}
    ## Schedules a task drained from a triggered event's channel,
    ## matching the C `tpc_flow_event_enqueue` signature

const noIterDep* = int32(-1)
  ## FlowEventDep.index of a single (non-iteration) event dependency,
  ## TPC_NO_ITER in C

proc flowEvent_initSingle*(ev: var FlowEvent) {.tpcev, importc: "tpc_flow_event_initSingle".}
  ## Initializes a single event with reference count 1,
  ## aborts when the event is already initialized

proc flowEvent_initIter*(ev: var FlowEvent, start, stop, stride: int32) {.tpcev, importc: "tpc_flow_event_initIter".}
  ## Initializes an iteration event over [start, stop) with one channel bucket
  ## per iteration and reference count 1

proc flowEvent_ref*(ev: FlowEvent) {.tpcev, importc: "tpc_flow_event_ref".}
  ## Adds one reference to the event behind the handle

proc flowEvent_unref*(ev: var FlowEvent) {.tpcev, importc: "tpc_flow_event_unref".}
  ## Drops one reference and frees the event on the last one,
  ## every dependent task must already be drained or re-deferred

proc flowEvent_delayedUntil*(task: ptr Task, ev: FlowEvent): bool {.tpcev, importc: "tpc_flow_event_delayedUntil".}
  ## Defers task until ev triggers
  ##
  ## Returns true when the task was delayed and the triggerer schedules it.
  ## Returns false when the event already fired and the caller schedules the task.

proc flowEvent_delayedUntilIter*(task: ptr Task, ev: FlowEvent, index: int32): bool {.tpcev, importc: "tpc_flow_event_delayedUntilIter".}
  ## Defers task until iteration `index` of the iteration event ev triggers,
  ## with the same contract as flowEvent_delayedUntil

proc flowEvent_delayedUntilMulti*(task: ptr Task, deps: ptr UncheckedArray[FlowEventDep], n: int32): bool {.tpcev, importc: "tpc_flow_event_delayedUntilMulti".}
  ## Defers task until every event in deps[0 ..< n] triggered
  ##
  ## Returns true when the task was delayed on some still-untriggered event.
  ## Returns false when every dependency already fired.

proc flowEvent_delayUntilMulti*(task: ptr Task, deps: openArray[FlowEventDep]): bool =
  ## openArray convenience over flowEvent_delayedUntilMulti
  flowEvent_delayedUntilMulti(task, cast[ptr UncheckedArray[FlowEventDep]](deps[0].unsafeAddr), int32 deps.len)

proc flowEvent_trigger*(ev: FlowEvent, queue: pointer, enqueue: FlowEventEnqueue) {.tpcev, importc: "tpc_flow_event_trigger".}
  ## Fires a single event, schedules every drained dependent task through
  ## enqueue(queue, task). An event triggers once, a re-trigger aborts.

proc flowEvent_triggerIter*(ev: FlowEvent, index: int32, queue: pointer, enqueue: FlowEventEnqueue) {.tpcev, importc: "tpc_flow_event_triggerIter".}
  ## Fires iteration `index` of an iteration event, same contract as flowEvent_trigger
