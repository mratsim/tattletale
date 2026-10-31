/* threadpool_compute: tasks and flowvars. */
#ifndef TPC_TASKS_FLOWVARS_H
#define TPC_TASKS_FLOWVARS_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "scoped_barriers.h"
#include "../primitives/futexes.h"
#include <stdlib.h>
#include <string.h>

/* Tasks have an efficient design so that a single heap allocation
 * is required per spawn.
 * This greatly reduces overhead and potential memory fragmentation for long-running applications.
 *
 * This is done by tasks:
 * - being an intrusive linked list
 * - integrating the channel to send results
 *
 * Flowvar is the public type created when spawning a task.
 * and can be synced to receive the task result.
 * Flowvars are also called future interchangeably.
 * (The name future is already used for IO scheduling)
 */

/** Task body signature: the env argument is the task's environment block. */
typedef void (*tpc_task_fn)(void* env);

/** Loop tasks keep their remaining coordinates in the header.
 *  Plain spawn tasks set loopStepsLeft to NotALoop.
 *  Coordinates are 32-bit: a range must satisfy ceilDiv(stop - start, stride) <= INT32_MAX,
 *  the constructors abort otherwise.
 */
#define TPC_NOT_A_LOOP ((int32_t)-1)

/* bitfield setup
 * A task has at most one waiter. Unused fields keep
 * their all-ones sentinel so registration is one fetch-add
 * replacing the sentinel with the id
 */
#define TPC_K_CAN_BE_FREED_SHIFT 31
#define TPC_K_CAN_BE_FREED       (1u << TPC_K_CAN_BE_FREED_SHIFT)          /* 0x80000000 */
#define TPC_K_WAITER_SHIFT       15
#define TPC_K_THIEF_MASK         ((1u << TPC_K_WAITER_SHIFT) - 1)          /* 0x00007FFF */
#define TPC_K_WAITER_MASK        (TPC_K_THIEF_MASK << TPC_K_WAITER_SHIFT)  /* 0x3FFF8000 */
#define TPC_SENTINEL_WAITER      TPC_K_WAITER_MASK
#define TPC_SENTINEL_THIEF       TPC_K_THIEF_MASK

/* Every task's environment block starts with the task self-pointer,
 * which the body reads through env alone, and the payload sits
 * after it. In a flowvar task the payload starts with the result
 * slot (written by readyWith, read by copyResult).
 */
#define TPC_FLOWVAR_HEADER_SIZE (sizeof(tpc_task*))

typedef struct tpc_task {
    _Atomic uint32_t completed;   /* completion futex word */
    /** This state allows synchronization between:
     *  - a waiter that may sleep if no work and task is incomplete
     *  - a thief that completes the task
     *  - a waiter that frees task memory
     *  - a waiter that will pick up the task continuation
     *
     *  Supports up to 2¹⁵ = 32768 threads
     */
    _Atomic uint32_t synchro;
    /* type synchro = object
     *   canBeFreed {.bitsize:  1.}: uint32 - Transfer ownership from thief to waiter
     *   pad        {.bitsize:  1.}: uint32
     *   waiterID   {.bitsize: 15.}: uint32 - ID of the waiter blocked on the task completion.
     *   thiefID    {.bitsize: 15.}: uint32 - ID of the worker that stole and run the task. For leapfrogging.
     */
    /* Synchronization
     * ------------------
     */
    struct tpc_task* parent;      /* Latency: When a task is awaited, a thread can quickly prioritize its direct children. */
    tpc_scoped_barrier* scopedBarrier;
    bool hasFuture;               /* Ownership: if a task has a future, the future deallocates it. Otherwise the worker thread does. */

    /* Data parallelism
     * ------------------
     */
    bool isFirstIter;    /* Load-Balancing: New loops are split before first iter. Split loops are run once before reconsidering split. */
    int32_t envSize;     /* Metadata: In splittable loops we need to copy the `env` upon splitting */
    int32_t loopStart, loopStop, loopStride;
    int32_t loopStepsLeft;

    /* Execution
     * ------------------
     */
    tpc_task_fn fn;
    _Alignas(8) unsigned char env[];
} tpc_task;

/* Flowvar is a public object, but we don't want
 * end-user to access the underlying task, so keep the field private.
 */
typedef struct tpc_flowvar {
    tpc_task* task;
} tpc_flowvar;


/* Task State
 * -------------------------------------------------------------------------
 *
 * Tasks have the following lifecycle:
 * - A task creator that schedule a task on its queue
 * - A task runner, task creator or thief, that runs the task
 * - Once the task is finished:
 *   - if the task has no future, the task runner frees the task
 *   - if the task has a future,
 *     - the task runner can immediately pick up new work
 *     - the awaiting thread frees the task
 *     - the awaiting thread might be sleeping and need to be woken up.
 *
 * There is a delicate dance as we are need to prevent 2 issues:
 *
 * 1. A deadlock:        if the waiter is never woken up after the thief completes the task
 * 2. A use-after-free:  if the thief tries to access the task after the waiter frees it.
 *
 * To solve 1, we need to set a `completed` flag, then check again if the waiter parked before.
 * To solve 2, we either need to ensure that after the `completed` flag is set, the task runner
 *             doesn't access the task anymore which is impossible due to 1;
 *             or we have the waiter spinlock on another flag `canBeFreed`.
 */

/** Reset a fresh task's synchronization state. */
static inline void tpc_task_initSynchroState(tpc_task* task) {
    atomic_store_explicit(&task->completed, 0, memory_order_relaxed);
    atomic_store_explicit(&task->synchro, TPC_SENTINEL_WAITER | TPC_SENTINEL_THIEF, memory_order_relaxed);
}


/** Check if task can be freed by the waiter if it was stolen. */
static inline bool tpc_task_isGcReady(const tpc_task* task) {
    return (atomic_load_explicit(&task->synchro, memory_order_acquire) & TPC_K_CAN_BE_FREED) != 0;
}


/** Thief transfers full task ownership to waiter. */
static inline void tpc_task_setGcReady(tpc_task* task) {
    atomic_fetch_add_explicit(&task->synchro, TPC_K_CAN_BE_FREED, memory_order_release);
}


/** Check task completion. */
static inline bool tpc_task_isCompleted(const tpc_task* task) {
    return atomic_load_explicit(&task->completed, memory_order_acquire) != 0;
}


/** Set a task to `complete`
 *  Wake a waiter thread if there is one
 *  Ownership: the runner frees a future-less task on completion,
 *  the awaiting thread frees the rest once canBeFreed shows up
 *  (see setGcReady).
 */
static inline void tpc_task_setCompleted(tpc_task* task) {
    /* release keeps the task's writes ordered before the completion flag */
    atomic_store_explicit(&task->completed, 1, memory_order_release);  /* Correctness on weak memory models like ARM: tests/t_ethereum_eip4844_deneb_kzg_parallel.nim */
    atomic_thread_fence(memory_order_seq_cst);                         /* Avoid deadlock on Windows: benchmarks-threadpool/fibonacci/threadpool_fib.nim */
    uint32_t waiter = atomic_load_explicit(&task->synchro,
                                           memory_order_relaxed);
    if ((waiter & TPC_K_WAITER_MASK) != TPC_SENTINEL_WAITER) {
        tpc_futex_wake(&task->completed, 1);
    }
}


/** Sleep while waiting for task completion
 *  waiterID registers the wait and the completer wakes every registered
 *  waiter exactly once.
 */
static inline void tpc_task_sleepUntilComplete(tpc_task* task, int32_t waiterID) {
    /* (id << 15) - sentinel replaces the all-ones sentinel with the id
     * in one fetch-add
     */
    uint32_t waiter = ((uint32_t)waiterID << TPC_K_WAITER_SHIFT) - TPC_SENTINEL_WAITER;
    atomic_fetch_add_explicit(&task->synchro, waiter, memory_order_relaxed);
    atomic_thread_fence(memory_order_acquire);
    while (atomic_load_explicit(&task->completed, memory_order_relaxed) == 0) {
        tpc_futex_wait(&task->completed, 0);
    }
}


/** Worker that stole this task, or TPC_SENTINEL_THIEF. */
static inline uint32_t tpc_task_getThief(const tpc_task* task) {
    return atomic_load_explicit(&task->synchro, memory_order_acquire) & TPC_K_THIEF_MASK;
}


/** Record the stealing worker (called by the deque's steal). */
static inline void tpc_task_setThief(tpc_task* task, int32_t thiefID) {
    uint32_t thief = (uint32_t)thiefID - TPC_SENTINEL_THIEF;
    atomic_fetch_add_explicit(&task->synchro, thief, memory_order_release);
}

/* Constructors
 * ---------------------------------------------------------------- */


/* Constructors
 * ---------------------------------------------------------------- */

/** Allocate a plain spawn task. The scoped barrier registers the task
 *  as a descendant. NULL runs the task outside any scope.
 */
static inline void task_payloadCopy(tpc_task* task, const void* env, int32_t envSize) {
    if (envSize > 0) {
        memcpy(task->env + TPC_FLOWVAR_HEADER_SIZE, env, (size_t)envSize);
    }
}

static inline tpc_task* task_alloc(size_t payloadSize) {
    /* env layout: [task self-pointer][payload] */
    tpc_task* task = malloc(sizeof(tpc_task) + TPC_FLOWVAR_HEADER_SIZE + payloadSize);
    if (task == NULL) {
        abort();
    }
    tpc_task* self = task;
    memcpy(task->env, &self, sizeof(self));
    return task;
}

static inline tpc_task* tpc_task_newSpawn(tpc_task* parent, tpc_scoped_barrier* barrier, tpc_task_fn fn) {
    tpc_scoped_barrier_registerDescendant(barrier);
    tpc_task* task = task_alloc(0);
    tpc_task_initSynchroState(task);
    task->parent = parent;
    task->scopedBarrier = barrier;
    task->hasFuture = false;
    task->fn = fn;
    task->envSize = 0;
    task->isFirstIter = false;
    task->loopStepsLeft = TPC_NOT_A_LOOP;
    return task;
}


/** Allocate a plain spawn task with an envSize-byte environment block,
 *  copied from env.
 */
static inline tpc_task* tpc_task_newSpawnEnv(tpc_task* parent, tpc_scoped_barrier* barrier,
                               tpc_task_fn fn, const void* env,
                               int32_t envSize) {
    tpc_scoped_barrier_registerDescendant(barrier);
    tpc_task* task = task_alloc((size_t)envSize);
    tpc_task_initSynchroState(task);
    task->parent = parent;
    task->scopedBarrier = barrier;
    task->hasFuture = false;
    task->fn = fn;
    task->envSize = envSize;
    task->isFirstIter = false;
    task->loopStepsLeft = TPC_NOT_A_LOOP;
    task_payloadCopy(task, env, envSize);
    return task;
}


/** Allocate a loop task over [start, stop) with the given stride.
 *  start < stop is a caller contract.
 */
static inline int32_t ceilDiv_vartime(int64_t a, int64_t b) {
    /* 64-bit intermediate: stop - start can exceed INT32_MAX before the division.
     * Results must fit int32
     */
    int64_t q = (a + b - 1) / b;
    if (q > INT32_MAX) {
        abort();
    }
    return (int32_t)q;
}

/** Allocate a loop task with an envSize-byte environment block. */
static inline tpc_task* tpc_task_newLoopEnv(tpc_task* parent, tpc_scoped_barrier* barrier,
                              tpc_task_fn fn, const void* env,
                              int32_t envSize, int32_t start, int32_t stop,
                              int32_t stride, bool isFirstIter) {
    tpc_scoped_barrier_registerDescendant(barrier);
    tpc_task* task = task_alloc((size_t)envSize);
    tpc_task_initSynchroState(task);
    task->parent = parent;
    task->scopedBarrier = barrier;
    task->hasFuture = false;
    task->fn = fn;
    task->envSize = envSize;
    task_payloadCopy(task, env, envSize);
    task->isFirstIter = isFirstIter;
    task->loopStart = start;
    task->loopStop = stop;
    task->loopStride = stride;
    task->loopStepsLeft = ceilDiv_vartime((int64_t)stop - start, stride);
    return task;
}

static inline tpc_task* tpc_task_newLoop(tpc_task* parent, tpc_scoped_barrier* barrier,
                           tpc_task_fn fn, int32_t start, int32_t stop,
                           int32_t stride, bool isFirstIter) {
    return tpc_task_newLoopEnv(parent, barrier, fn, NULL, 0, start, stop, stride, isFirstIter);
}

/* Flowvars
 * ---------------------------------------------------------------- */


/* Flowvars
 * ---------------------------------------------------------------- */

/* Task with future references themselves so that readyWith can be called
 * within the constructed
 *   proc threadpoolSpawn_fn(env: pointer) {.nimcall.}
 * that can only access env
 */
static inline tpc_flowvar tpc_flowvar_new(tpc_task* task) {
    tpc_flowvar fv;
    fv.task = task;
    task->hasFuture = true;
    return fv;
}


/** Waiter-side free: wait until the completer's canBeFreed bit shows up, then free the task.
 */
static inline void tpc_flowvar_cleanup(tpc_flowvar* fv) {
    while (!tpc_task_isGcReady(fv->task)) {
        tpc_cpuRelax();
    }
    free(fv->task);
    fv->task = NULL;
}


/** Returns true if a flowvar is spawned
 *  This may be useful for recursive algorithms that
 *  may or may not spawn a flowvar depending on a condition.
 *  This is similar to Option or Maybe types
 */
static inline bool tpc_flowvar_isSpawned(tpc_flowvar fv) {
    return fv.task != NULL;
}


/** Returns true if the result of a Flowvar is ready.
 *  In that case `sync` will not block.
 *  Otherwise the current will block to help on all the pending tasks
 *  until the Flowvar is ready.
 */
static inline bool tpc_flowvar_isReady(tpc_flowvar fv) {
    return tpc_task_isCompleted(fv.task);
}


static inline tpc_task* tpc_flowvar_getTask(tpc_flowvar fv) {
    return fv.task;
}


/** Send the result from the task body to its waiter: resultSize bytes
 *  into the result slot after the flowvar header.
 *
 *  Parameters:
 *  In:      task:        the running task.
 *          resultSize:  the result slot size.
 *          result:      the result bytes.
 */
static inline void tpc_task_readyWith(tpc_task* task, const void* result, size_t resultSize) {
    memcpy(task->env + TPC_FLOWVAR_HEADER_SIZE, result, resultSize);
}


/** Copy the result of a ready Flowvar to `dst`
 *
 *  Parameters:
 *  Out:     dst:         the result destination.
 *  In:      resultSize:  the result slot size.
 *          fv:          the ready flowvar.
 */
static inline void tpc_flowvar_copyResult(void* dst, tpc_flowvar fv, size_t resultSize) {
    memcpy(dst, fv.task->env + TPC_FLOWVAR_HEADER_SIZE, resultSize);
}


#endif /* TPC_TASKS_FLOWVARS_H */
