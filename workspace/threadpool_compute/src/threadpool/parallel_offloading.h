/* threadpool_compute: task spawn and flowvar sync.
 *
 * Parallel offloading API
 * -----------------------
 *
 * This file implements the runtime helpers of the parallelism paradigms supported:
 *
 * - Task parallelism
 * - Data parallelism / parallel for
 *   - parallel-for with thread-local prologue and epilogue
 * - Dataflow parallelism
 *   - also known as:
 *     - Graph parallelism
 *     - Stream parallelism
 *     - Pipeline parallelism
 *     - Data-driven (task) parallelism
 *     with precise input/output dependencies
 */
#ifndef TPC_PARALLEL_OFFLOADING_H
#define TPC_PARALLEL_OFFLOADING_H

#include <stddef.h>
#include <stdint.h>

#include "crossthread/tasks_flowvars.h"
#include "threadpool.h"

/** Get the calling worker's pool. */
static inline tpc_threadpool* tpc_threadpool_current(void) {
    return tpc_workerContext()->threadpool;
}

/** Spawn a task running fn over a copy of the env block, parented on
 *  the calling worker's current task and scope.
 *
 *  Parameters:
 *  In:      fn:       the task body, it receives the raw env with the task self-pointer in the first 8 bytes
 *                   and the payload after
 *          envSize:  payload size in bytes.
 *          env:      the payload to copy, NULL when envSize is 0.
 *
 *  Returns: the task's flowvar view.
 */
static inline tpc_flowvar tpc_spawnNew(tpc_task_fn fn, const void* env, int32_t envSize) {
    tpc_worker_context* ctx = tpc_workerContext();
    tpc_task* task = tpc_task_newSpawnEnv(ctx->currentTask, ctx->currentScope, fn, env, envSize);
    tpc_flowvar fv = tpc_flowvar_new(task);
    tpc_schedule(ctx, task, false);
    return fv;
}

/** Wait for the flowvar and copy its resultSize-byte result into dst.
 *
 *  While waiting the worker runs other work in this order:
 *  1. direct children of the awaited task (they unblock it fastest),
 *  2. leapfrog: steal from the thief that took the awaited task,
 *     its deque contains the tasks generated next to it,
 *  3. steal from any deque,
 *  4. run the own deque,
 *  5. park on the awaited task's completion futex.
 *
 *  Parameters:
 *  Out:     dst:         the result destination.
 *  In:      resultSize:  the flowvar's result slot size.
 *          fv:          the awaited flowvar.
 */
static inline void tpc_flowvar_sync(void* dst, tpc_flowvar* fv, size_t resultSize) {
    tpc_completeFuture(fv, dst, resultSize);
    tpc_flowvar_cleanup(fv);
}

#endif /* TPC_PARALLEL_OFFLOADING_H */
