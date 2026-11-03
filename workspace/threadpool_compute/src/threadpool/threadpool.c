/* threadpool_compute: the scheduler. */
#include "threadpool.h"
#include "primitives/futexes.h"
#include "primitives/topology.h"
#include "primitives/allocs.h"

#include <stdlib.h>
#include <string.h>

static _Thread_local tpc_worker_context tpc_tlsWorkerContext;

tpc_worker_context* tpc_workerContext(void) {
    return &tpc_tlsWorkerContext;
}

/* ############################################################
 * #                                                          #
 * #                            RNG                           #
 * #                                                          #
 * ############################################################
 *
 * We don't need a CSPRNG, the RNG is to select a random victim when work-stealing
 *
 * - Scrambled Linear Pseudorandom Number Generators
 *   Blackman, Vigna, 2021
 *   https://vigna.di.unimi.it/ftp/papers/ScrambledLinear.pdf
 *   https://prng.di.unimi.it/
 */

void tpc_rng_seed(tpc_work_stealing_rng* rng, uint64_t x) {
    uint64_t sm = x;
    for (int i = 0; i < 4; i++) {
        sm += 0x9e3779b97f4a7c15ull;
        uint64_t z = sm;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        rng->s[i] = z ^ (z >> 31);
    }
}

uint32_t tpc_rng_nextU32(tpc_work_stealing_rng* rng) {
    /* Need to use the high bits */
    uint32_t result = (uint32_t)((rng->s[0] + rng->s[3]) >> 32);

    uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = (rng->s[3] << 45) | (rng->s[3] >> 19);
    return result;
}

void tpc_perm_init(tpc_perm* perm, uint32_t randomSeed, int32_t maxExclusive) {
    /* LCG over M = next power of two of maxExclusive: every odd c is
     * coprime with M and a = 4k+1 satisfies Hull-Dobell, so the walk
     * visits every residue exactly once per cycle, no GCD search,
     * no coprime cache
     */
    uint32_t M = 1;
    while (M < (uint32_t)maxExclusive) {
        M <<= 1;
    }
    perm->mask = M - 1;
    perm->c = (randomSeed & ((M >> 1) - 1)) * 2 + 1;
    perm->a = (randomSeed & ((M >> 2) - 1)) * 4 + 1;
    perm->start = randomSeed & perm->mask;
    perm->x = perm->start;
    perm->maxExclusive = (uint32_t)maxExclusive;
    perm->done = false;
}

int32_t tpc_perm_next(tpc_perm* perm) {
    if (perm->done) {
        return -1;
    }
    uint32_t x = perm->x;
    perm->x = (perm->a * x + perm->c) & perm->mask;
    if (perm->x == perm->start) {
        perm->done = true;   /* cycle completed */
    }
    if (x >= perm->maxExclusive) {
        return tpc_perm_next(perm);   /* skip residues outside the range */
    }
    return (int32_t)x;
}

/* Tasks
 * ---------------------------------------------------------------- */

void tpc_run(tpc_worker_context* ctx, tpc_task* task) {
    tpc_task* suspendedTask = ctx->currentTask;
    tpc_scoped_barrier* suspendedScope = ctx->currentScope;

    ctx->currentTask = task;
    ctx->currentScope = task->scopedBarrier;

    tpc_debug_log("Worker %3d: running task %p\n", ctx->id, (void*)task);
    task->fn(task->env);
    tpc_scoped_barrier_unlistDescendant(task->scopedBarrier);

    ctx->currentTask = suspendedTask;
    ctx->currentScope = suspendedScope;

    ctx->counters.tasksExecuted++;
    if (task->loopStepsLeft != TPC_NOT_A_LOOP) {
        ctx->counters.itersExecuted += ((int64_t)task->loopStop - task->loopStart + task->loopStride - 1) / task->loopStride;
    }

    if (!task->hasFuture) {
        /* final owner frees the task */
        free(task);
        return;
    }
    /* Sync with an awaiting thread in completeFuture that didn't find work
     * and transfer ownership of the task to it.
     */
    tpc_task_setCompleted(task);
    tpc_task_setGcReady(task);
}

void tpc_schedule(tpc_worker_context* ctx, tpc_task* task, bool forceWake) {
    /* Instead of notifying every time a task is scheduled, we notify
     * only when the worker queue is empty. This is a good approximation
     * of starvation in work-stealing.
     */
    bool wasEmpty = tpc_taskqueue_peek(ctx->taskqueue) == 0;
    tpc_taskqueue_push(ctx->taskqueue, task);

    ctx->counters.tasksScheduled++;
    if (task->loopStepsLeft != TPC_NOT_A_LOOP) {
        ctx->counters.itersScheduled += task->loopStepsLeft;
    }

    if (forceWake || wasEmpty) {
        tpc_eventcount_wake(&ctx->threadpool->globalBackoff);
        ctx->counters.backoffGlobalSignalSent++;
    }
}

/* ############################################################
 * #                                                          #
 * #                       Scheduler                          #
 * #                                                          #
 * ############################################################ */

tpc_task* tpc_tryStealOne(tpc_worker_context* ctx) {
    uint32_t seed = tpc_rng_nextU32(&ctx->rng);
    tpc_perm perm;
    tpc_perm_init(&perm, seed, ctx->threadpool->numThreads);
    for (;;) {
        int32_t targetId = tpc_perm_next(&perm);
        if (targetId < 0) {
            return NULL;
        }
        if (targetId == ctx->id) {
            continue;
        }
        tpc_task* stolen = tpc_taskqueue_steal(ctx->id, &ctx->threadpool->workerQueues[targetId]);
        if (stolen != NULL) {
            return stolen;
        }
    }
}

tpc_task* tpc_tryLeapfrog(tpc_worker_context* ctx, tpc_task* awaitedTask) {
    /* Leapfrogging:
     *
     * - Leapfrogging: a portable technique for implementing efficient futures,
     *   David B. Wagner, Bradley G. Calder, 1993
     *   https://dl.acm.org/doi/10.1145/173284.155354
     *
     * When awaiting a future, we can look in the thief queue first. They steal when they run out of tasks.
     * If they have tasks in their queue, it's the task we are awaiting that created them and it will likely be stuck
     * on those tasks as well, so we need to help them help us.
     */
    uint32_t thiefID;
    for (;;) {
        thiefID = tpc_task_getThief(awaitedTask);
        if (thiefID != TPC_SENTINEL_THIEF) {
            break;
        }
        tpc_cpuRelax();
    }
    tpc_ascertain(thiefID < (uint32_t)ctx->threadpool->numThreads);
    return tpc_taskqueue_steal(ctx->id, &ctx->threadpool->workerQueues[thiefID]);
}

/* Worker loop
 * ---------------------------------------------------------------- */


void tpc_eventLoop(tpc_worker_context* ctx) {
    for (;;) {
        /* 1. Pick from local queue */
        tpc_task* task;
        while ((task = tpc_taskqueue_pop(ctx->taskqueue)) != NULL) {
            tpc_run(ctx, task);
        }

        /* 2. Run out of tasks, become a thief */
        tpc_parking_ticket ticket =
            tpc_eventcount_sleepy(&ctx->threadpool->globalBackoff);
        tpc_task* stolen = tpc_tryStealOne(ctx);
        if (stolen != NULL) {
            /* We manage to steal a task, cancel sleep */
            /* Theft successful, there might be more work for idle threads, wake one
             * cancelSleep must be done before as wake has an optimization
             * to not notify when a thread is sleepy
             */
            tpc_eventcount_cancelSleep(&ctx->threadpool->globalBackoff);
            /* a theft signals global work: wake one more hunter */
            tpc_eventcount_wake(&ctx->threadpool->globalBackoff);
            ctx->counters.backoffGlobalSignalSent++;

            ctx->counters.theftsIdle++;
            ctx->counters.tasksStolen++;
            if (stolen->loopStepsLeft != TPC_NOT_A_LOOP) {
                ctx->counters.itersStolen += stolen->loopStepsLeft;
            }
            tpc_run(ctx, stolen);
        } else if (atomic_load_explicit(&ctx->signal->terminate,
                                        memory_order_acquire)) {
            tpc_eventcount_cancelSleep(&ctx->threadpool->globalBackoff);
            break;
        } else {
            /* 2.c Park the thread until a new task enters the threadpool */
            ctx->counters.backoffGlobalSleep++;
            tpc_eventcount_sleep(&ctx->threadpool->globalBackoff, ticket);
        }
    }
}

/* ############################################################
 * #                                                          #
 * #                Futures & Synchronization                 #
 * #                                                          #
 * ############################################################ */

void tpc_completeFuture(tpc_flowvar* fv, void* parentResult, size_t resultSize) {
    tpc_worker_context* ctx = tpc_workerContext();

    if (tpc_flowvar_isReady(*fv)) {
        tpc_flowvar_copyResult(parentResult, *fv, resultSize);
        return;
    }

    /* Design tradeoffs
     * ----------------
     * see ./docs/design.md
     *
     * Rundown:
     *
     * At this point, we have significant design decisions:
     * - Do we steal from other workers in hope we advance our awaited task?
     *   Note: A greedy scheduler (no worker idle as long as there is a runnable task)
     *         is at most 2x slower than the optimal schedule (proof in Cilk paper)
     * - Do we advance our own queue for tasks that are not child of our awaited tasks?
     * - Do we park instead of working on unrelated task?
     *   Note: With hyperthreading, real hardware resources are 2x less than the reported number of cores.
     *         Hence parking might free contended memory bandwitdh or execution ports.
     * - Do we just not sleep, potentially wasting energy?
     *   Note: on "empty tasks" (like Fibonacci), the constant hammering of threads
     *         actually slows down performance by 2x.
     *         This is a realistic scenario for IO (waiting to copy a network buffer for example)
     *         but the second almost-empty benchmark, depth-first-search is actually faster by 17%.
     *
     * - If we work, we maximize throughput, but we increase latency to handle the future's continuation.
     *   If that continuation would have created more parallel work, we would actually have restricted parallelism.
     * - If we park with tasks left, we minimize latency on the continuation, but we don't use hardware resources fully,
     *   and there are CPUs without hyperthreading (on ARM for example)
     * - If we park when no tasks are left, if more work is enqueued, as we don't park on the global backoff we will miss it.
     *   Note: we don't park on the global backoff, because it's not possible to control which thread to wake with it (or we wake all).
     * - Wakeup latency is high, having "reserve threads" that take over the active slot of the awaiting thread
     *   in theory maintains throughput and minimize the latency of the future's continuation
     *   but in practice, performance can worsen significantly on fine-grained parallelism.
     */

    /* 1. Process all the children of the current tasks first, ignoring the rest.
     */
    tpc_task* task;
    while ((task = tpc_taskqueue_pop(ctx->taskqueue)) != NULL) {
        if (task->parent != ctx->currentTask) {
            /* unrelated: hand it over and wake a sibling to take it */
            tpc_schedule(ctx, task, true);
            break;
        }
        tpc_run(ctx, task);
        if (tpc_flowvar_isReady(*fv)) {
            tpc_flowvar_copyResult(parentResult, *fv, resultSize);
            return;
        }
    }
    if (tpc_flowvar_isReady(*fv)) {
        tpc_flowvar_copyResult(parentResult, *fv, resultSize);
        return;
    }

    /* 2. bottlenecked on dependencies in other threads: abandon our
     * own queue and hunt. Proactive work-stealing for futures
     * (Singer, Xu, Lee 2019).
     */
    while (!tpc_flowvar_isReady(*fv)) {
        tpc_task* leapTask = tpc_tryLeapfrog(ctx, tpc_flowvar_getTask(*fv));
        if (leapTask != NULL) {
            tpc_eventcount_wake(&ctx->threadpool->globalBackoff);
            ctx->counters.backoffGlobalSignalSent++;
            ctx->counters.theftsLeapfrog++;
            ctx->counters.tasksStolen++;
            if (leapTask->loopStepsLeft != TPC_NOT_A_LOOP) {
                ctx->counters.itersStolen += leapTask->loopStepsLeft;
            }
            tpc_run(ctx, leapTask);
            continue;
        }
        tpc_task* stolen = tpc_tryStealOne(ctx);
        if (stolen != NULL) {
            tpc_eventcount_wake(&ctx->threadpool->globalBackoff);
            ctx->counters.backoffGlobalSignalSent++;
            ctx->counters.theftsAwaiting++;
            ctx->counters.tasksStolen++;
            if (stolen->loopStepsLeft != TPC_NOT_A_LOOP) {
                ctx->counters.itersStolen += stolen->loopStepsLeft;
            }
            tpc_run(ctx, stolen);
            continue;
        }
        task = tpc_taskqueue_pop(ctx->taskqueue);
        if (task != NULL) {
            /* We advance our own queue...
             * throughput over latency: our awaited task progresses on its thief either way
             */
            ctx->counters.unrelatedTasksExecuted++;
            tpc_run(ctx, task);
            continue;
        }
        /* Nothing to do, we park.
         * - park on the awaited task's own futex
         * - precise wakeup on completion
         */
        ctx->counters.backoffTaskAwaited++;
        tpc_task_sleepUntilComplete(tpc_flowvar_getTask(*fv), ctx->id);
    }
    tpc_flowvar_copyResult(parentResult, *fv, resultSize);
}

void tpc_syncAll(tpc_threadpool* tp) {
    tpc_worker_context* ctx = tpc_workerContext();

    tpc_preCondition(ctx->id == 0);
    tpc_preCondition(tpc_isRootTask(ctx->currentTask));

    for (;;) {
        /* 1. Empty local tasks */
        tpc_task* task;
        while ((task = tpc_taskqueue_pop(ctx->taskqueue)) != NULL) {
            tpc_run(ctx, task);
        }

        /* 2. Help other threads */
        tpc_task* stolen = tpc_tryStealOne(ctx);
        if (stolen != NULL) {
            tpc_eventcount_wake(&tp->globalBackoff);
            ctx->counters.backoffGlobalSignalSent++;
            ctx->counters.theftsIdle++;
            ctx->counters.tasksStolen++;
            if (stolen->loopStepsLeft != TPC_NOT_A_LOOP) {
                ctx->counters.itersStolen += stolen->loopStepsLeft;
            }
            tpc_run(ctx, stolen);
        } else {
            int32_t preSleep, committedSleep;
            tpc_eventcount_getNumWaiters(&tp->globalBackoff, &preSleep, &committedSleep);
            if (preSleep == 0 && committedSleep == tp->numThreads - 1) {
                /* Don't count ourselves
                 * 2.b all threads besides the current are parked
                 */
                break;
            }
            /* 2.c We don't park as there is no notif for task completion */
            tpc_cpuRelax();
        }
    }
}

void tpc_syncScopeWait(tpc_scoped_barrier* scopedBarrier) {
    tpc_worker_context* ctx = tpc_workerContext();

    while (tpc_scoped_barrier_hasDescendantTasks(scopedBarrier)) {
        /* local deque tasks all belong to this scope or a child scope */
        tpc_task* task;
        while ((task = tpc_taskqueue_pop(ctx->taskqueue)) != NULL) {
            tpc_run(ctx, task);
            if (!tpc_scoped_barrier_hasDescendantTasks(scopedBarrier)) {
                return;
            }
        }
        tpc_task* stolen = tpc_tryStealOne(ctx);
        if (stolen != NULL) {
            tpc_eventcount_wake(&ctx->threadpool->globalBackoff);
            ctx->counters.backoffGlobalSignalSent++;
            ctx->counters.theftsIdle++;
            ctx->counters.tasksStolen++;
            if (stolen->loopStepsLeft != TPC_NOT_A_LOOP) {
                ctx->counters.itersStolen += stolen->loopStepsLeft;
            }
            tpc_run(ctx, stolen);
        } else {
            tpc_cpuRelax();
        }
    }
}

/* Data-parallel load balancing
 * ----------------------------------------------------------------
 *
 * Lazy binary splitting (Tzannes et al., PPoPP 2010) with log-log
 * iterated backoff (Bender et al., 2005). Loops split just-in-time.
 * Split counts track idle threads instead of core counts.
 * Parallel-for stays portable across CPUs and inner algorithms.
 * No grain-size threshold exists.
 *
 * An idle-count check costs a guaranteed cache miss, so the check
 * cadence backs off. After an unproductive check the next one moves
 * 2^windowLogSize iterations out, and windowLogSize grows
 * after log2(windowLogSize) idle rounds. A productive split
 * shrinks the window at once.
 */

static uint32_t log2_vartime(uint32_t v) {
    uint32_t r = 0;
    while ((v >>= 1) != 0) {
        r++;
    }
    return r;
}

static void balancer_increase(tpc_balancer_backoff* backoff) {
    /* On failure, we use log-log iterated backoff, an optimal backoff strategy
     * suitable for bursts and adversarial conditions.
     */
    backoff->round++;
    if (backoff->round >= log2_vartime(backoff->windowLogSize)) {
        backoff->round = 0;
        backoff->windowLogSize++;
    }
}

static void balancer_decrease(tpc_balancer_backoff* backoff) {
    /* On success, we exponentially reduce check window.
     * Note: the thieves will start contributing as well.
     */
    if (backoff->windowLogSize > 0) {
        backoff->windowLogSize--;
    }
    backoff->round = 0;
}

void tpc_splitUpperRanges(tpc_task* task, int32_t curLoopIndex, int32_t approxIdle, tpc_upper_split* split) {
    tpc_preCondition(task->loopStepsLeft > 1);
    tpc_preCondition(curLoopIndex % task->loopStride == 0);

    /* Send a chunk of work to all idle workers + ourselves */
    int32_t availableWorkers = approxIdle + 1;
    int32_t baseChunkSize =
        (int32_t)((int64_t)task->loopStepsLeft / availableWorkers);
    int32_t cutoff = (int32_t)((int64_t)task->loopStepsLeft % availableWorkers);

    /* chunkID 0 is ours! My precious!!! */
    split->prefixSteps = baseChunkSize + (cutoff > 0 ? 1 : 0);
    task->loopStepsLeft = split->prefixSteps;
    int64_t stop = (int64_t)curLoopIndex + (int64_t)split->prefixSteps * task->loopStride;
    task->loopStop = stop < task->loopStop ? (int32_t)stop : task->loopStop;

    split->numChunks = availableWorkers - 1;
    split->baseChunkSize = baseChunkSize;
    split->cutoff = cutoff;
}

void tpc_splitAndDispatchLoop(tpc_worker_context* ctx, tpc_task* task, int32_t curLoopIndex, int32_t approxIdle) {
    int64_t stop = task->loopStop;
    tpc_upper_split split;
    tpc_splitUpperRanges(task, curLoopIndex, approxIdle, &split);

    for (int32_t chunkID = 1; chunkID <= split.numChunks; chunkID++) {
        int32_t chunkSize =
            split.baseChunkSize + (chunkID < split.cutoff ? 1 : 0);
        if (chunkSize == 0) {
            break;
        }
        int64_t offset;
        if (chunkID < split.cutoff) {
            offset = (int64_t)curLoopIndex + (int64_t)task->loopStride * chunkSize * chunkID;
        } else {
            offset = (int64_t)curLoopIndex + (int64_t)task->loopStride * ((int64_t)split.baseChunkSize * chunkID + split.cutoff);
        }

        /* a loop chunk copies the parent's env block whole */
        tpc_task* upper = tpc_task_newLoopEnv(
            task, task->scopedBarrier, task->fn,
            task->env + 8, task->envSize,
            (int32_t)offset,
            stop < (int64_t)offset + (int64_t)chunkSize * task->loopStride
                ? (int32_t)stop
                : (int32_t)(offset + (int64_t)chunkSize * task->loopStride),
            task->loopStride, false);
        upper->isFirstIter = false;
        upper->loopStepsLeft = chunkSize;
        tpc_taskqueue_push(ctx->taskqueue, upper);
    }

    tpc_eventcount_wakeAll(&ctx->threadpool->globalBackoff);
    ctx->counters.backoffGlobalSignalSent++;
}

void tpc_loadBalanceLoop(tpc_worker_context* ctx, tpc_task* task, int32_t curLoopIndex, tpc_balancer_backoff* backoff) {
    /* We might want to make this inline to cheapen the first check
     * but it is 10% faster not inline on the transpose benchmark (memory-bandwidth bound)
     */
    if (task->loopStepsLeft > 1 && curLoopIndex == backoff->nextCheck) {
        if (tpc_taskqueue_peek(ctx->taskqueue) == 0) {
            int32_t preSleep, committedSleep;
            tpc_eventcount_getNumWaiters(&ctx->threadpool->globalBackoff, &preSleep, &committedSleep);
            /* We assume that the worker that scheduled the task will work on it. I.e. idleness is underestimated.
             */
            int32_t approxIdle = preSleep + committedSleep + (task->isFirstIter ? 1 : 0);
            if (approxIdle > 0) {
                tpc_splitAndDispatchLoop(ctx, task, curLoopIndex, approxIdle);
                ctx->counters.loopsSplit++;
                balancer_decrease(backoff);
            } else {
                balancer_increase(backoff);
            }
        } else {
            balancer_increase(backoff);
        }
        backoff->nextCheck += task->loopStride << backoff->windowLogSize;
    }
}

void tpc_parallelForRunner(tpc_worker_context* ctx, tpc_task* task, tpc_range_fn body) {
    tpc_balancer_backoff backoff;
    backoff.nextCheck = task->loopStart;
    backoff.windowLogSize = 0;
    backoff.round = 0;
    if (!task->isFirstIter) {
        /* Task was just stolen, no need to check runtime status. do one loop first */
        backoff.nextCheck += task->loopStride;
    }

    void* userEnv = task->env + 8;
    for (int32_t idx = task->loopStart; idx < task->loopStop;
         idx += task->loopStride) {
        tpc_loadBalanceLoop(ctx, task, idx, &backoff);
        body(idx, userEnv);
        task->loopStepsLeft--;
    }
}

/* ############################################################
 * #                                                          #
 * #                       Runtime API                        #
 * #                                                          #
 * ############################################################ */

/** Initialize the thread-local context of a worker
 *  Requires the ID and threadpool fields to be initialized
 */
static void setupWorker(tpc_worker_context* ctx) {
    tpc_preCondition(ctx->threadpool != NULL);
    tpc_preCondition(0 <= ctx->id && ctx->id < ctx->threadpool->numThreads);

    /* Thefts */
    tpc_rng_seed(&ctx->rng, 0xEFFACEDull + (uint64_t)ctx->id);

    /* Synchronization */
    ctx->currentScope = NULL;
    ctx->signal = &ctx->threadpool->workerSignals[ctx->id];
    atomic_store_explicit(&ctx->signal->terminate, false, memory_order_relaxed);

    /* Tasks */
    ctx->taskqueue = &ctx->threadpool->workerQueues[ctx->id];
    ctx->currentTask = NULL;

    /* Init */
    tpc_taskqueue_init(ctx->taskqueue, 32);
}

/** Cleanup the thread-local context of a worker */
static void teardownWorker(tpc_worker_context* ctx) {
    tpc_taskqueue_teardown(ctx->taskqueue);
}

/* On the start of the threadpool workers will execute this
 * until they receive a termination signal
 */
static int workerEntryFn(void* arg) {
    /* We assume that thread_local variables start all at their binary zero value */
    tpc_threadpool_args* params = (tpc_threadpool_args*)arg;
    tpc_worker_context* ctx = tpc_workerContext();

    ctx->id = params->id;
    ctx->threadpool = params->tp;

    setupWorker(ctx);

    /* 1 matching barrier in tpc_threadpool_new */
    tpc_barrier_wait(&params->tp->barrier);

    tpc_eventLoop(ctx);

    tpc_debug_termination(">>> Worker %3d shutting down <<<\n", ctx->id);

    /* 1 matching barrier in tpc_threadpool_shutdown */
    tpc_barrier_wait(&params->tp->barrier);

    teardownWorker(ctx);
    return 0;
}

tpc_threadpool* tpc_threadpool_new(int32_t numThreads) {
    tpc_threadpool* tp = (tpc_threadpool*)tpc_alignedAlloc(_Alignof(tpc_threadpool), sizeof(*tp));
    if (tp == NULL) {
        abort();
    }
    memset(tp, 0, sizeof(*tp));
    tpc_barrier_init(&tp->barrier, (uint32_t)numThreads);
    tpc_eventcount_initialize(&tp->globalBackoff);
    tp->numThreads = numThreads;
    tp->workerQueues = (tpc_taskqueue*)tpc_alignedAlloc(_Alignof(tpc_taskqueue), (size_t)numThreads * sizeof(tpc_taskqueue));
    tp->workers = (tpc_thread_t*)malloc((size_t)numThreads * sizeof(tpc_thread_t));
    tp->workerSignals = (struct tpc_worker_signal*)tpc_alignedAlloc(_Alignof(struct tpc_worker_signal), (size_t)numThreads * sizeof(*tp->workerSignals));
    tp->workerArgs = (tpc_threadpool_args*)malloc((size_t)numThreads * sizeof(*tp->workerArgs));
    if (tp->workerQueues == NULL || tp->workers == NULL || tp->workerSignals == NULL || tp->workerArgs == NULL) {
        abort();
    }

    /* Setup master thread */
    tpc_worker_context* ctx = tpc_workerContext();
    ctx->id = 0;
    ctx->threadpool = tp;

    for (int32_t i = 1; i < numThreads; i++) {
        tp->workerArgs[i].tp = tp;
        tp->workerArgs[i].id = i;
        if (tpc_thread_create(&tp->workers[i], workerEntryFn, &tp->workerArgs[i]) != 0) {
            /* The barrier counts numThreads participants, a missing worker
             * would hang the pool at startup */
            abort();
        }
    }

    setupWorker(ctx);

    /* Root task, this is a sentinel task that is never called. */
    ctx->currentTask = TPC_ROOT_TASK;

    tpc_barrier_wait(&tp->barrier);
    return tp;
}

void tpc_threadpool_shutdown(tpc_threadpool* tp) {
    tpc_worker_context* ctx = tpc_workerContext();
    tpc_preCondition(tpc_isRootTask(ctx->currentTask));

    tpc_syncAll(tp);

    /* Signal termination to all threads */
    for (int32_t i = 0; i < tp->numThreads; i++) {
        atomic_store_explicit(&tp->workerSignals[i].terminate, true, memory_order_release);
    }
    tpc_eventcount_wakeAll(&tp->globalBackoff);

    /* 1 matching barrier in workerEntryFn */
    tpc_barrier_wait(&tp->barrier);

    teardownWorker(ctx);

    for (int32_t i = 1; i < tp->numThreads; i++) {
        tpc_thread_join(tp->workers[i]);
    }

    tpc_alignedFree(tp->workerSignals);
    free(tp->workers);
    tpc_alignedFree(tp->workerQueues);
    free(tp->workerArgs);
    tpc_alignedFree(tp);

    ctx->currentTask = NULL;
}

const tpc_counters* tpc_threadpool_workerCounters(void) {
    return &tpc_workerContext()->counters;
}
