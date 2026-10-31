# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# This file statically compiles the threadpool_compute C project
# and exposes its API in Nim. Original sources stay untouched.
#
# C sources compile as one translation unit (src/threadpool_runtime.c):
# it includes every source and picks the platform files by preprocessor.
# This wrapper compiles that one file at the optimization level
# chosen for the surrounding Nim program.
#
# Low-level binding over the C surface: opaque pool and task
# worker-context pointers, flowvars as plain values, cdecl task functions.
# Spawn helpers copy the environment block into the task, a task body
# reads its payload after the flowvar header where the first env word
# is the task self-pointer written by the constructors.

import std/os

const tpcRoot = currentSourcePath.parentDir

{.compile: "src/threadpool_runtime.c".}

{.passc: "-I\"" & tpcRoot & "\"".}
{.passc: "-I\"" & tpcRoot / "src" / "threadpool" & "\"".}

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
  ## Completes the flowvar, frees its task and copies the result out

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
