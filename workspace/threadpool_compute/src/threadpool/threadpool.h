/* threadpool_compute: the scheduler. */
#ifndef TPC_THREADPOOL_H
#define TPC_THREADPOOL_H

#include "primitives/atomics_compat.h"
#include <stdbool.h>
#include <stdint.h>

#include "crossthread/backoff.h"
#include "crossthread/scoped_barriers.h"
#include "crossthread/taskqueues.h"
#include "crossthread/tasks_flowvars.h"
#include "instrumentation.h"
#include "primitives/barriers.h"
#include "primitives/threads.h"

/* Sentinel values */
#define TPC_ROOT_TASK ((tpc_task*)0xEFFACED0ull)

/* Scheduler counters, one set per worker.
 * tasksStolen counts every theft, idle or awaiting or leapfrog.
 */
typedef struct tpc_counters {
    int64_t tasksScheduled;
    int64_t tasksStolen;
    int64_t tasksExecuted;
    int64_t unrelatedTasksExecuted;
    int64_t loopsSplit;
    int64_t itersScheduled;
    int64_t itersStolen;
    int64_t itersExecuted;
    int64_t theftsIdle;
    int64_t theftsAwaiting;
    int64_t theftsLeapfrog;
    int64_t backoffGlobalSleep;
    int64_t backoffGlobalSignalSent;
    int64_t backoffTaskAwaited;
} tpc_counters;

/** This is the state of a Xoshiro256+ PRNG
 *  It is used for work-stealing. The low bits have low linear complexity.
 *  So we use the high 32 bits to seed our pseudo random walk of thread taskqueues.
 *
 *  We don't need a CSPRNG, the RNG is to select a random victim when work-stealing
 *
 *  - Scrambled Linear Pseudorandom Number Generators
 *    Blackman, Vigna, 2021
 *    https://vigna.di.unimi.it/ftp/papers/ScrambledLinear.pdf
 *    https://prng.di.unimi.it/
 */
typedef struct tpc_work_stealing_rng {
    uint64_t s[4];
} tpc_work_stealing_rng;

/** Seed the random number generator with a fixed seed */
void tpc_rng_seed(tpc_work_stealing_rng* rng, uint64_t x);

/** Compute a random uint32 */
uint32_t tpc_rng_nextU32(tpc_work_stealing_rng* rng);

/** Create (low-quality) pseudo-random permutations for [0, max)
 *
 *  Design considerations and randomness constraint for work-stealing, see docs/random_permutations.md
 *
 *  Linear Congruential Generator: https://en.wikipedia.org/wiki/Linear_congruential_generator
 *
 *  Xₙ₊₁ = aXₙ+c (mod m) generates all random number mod m without repetition
 *  if and only if (Hull-Dobell theorem):
 *  1. c and m are coprime
 *  2. a-1 is divisible by all prime factors of m
 *  3. a-1 is divisible by 4 if m is divisible by 4
 *
 *  Alternative 1. By choosing a=1, all conditions are easy to reach.
 *
 *  The randomness quality is not important besides distributing potential contention,
 *  i.e. randomly trying thread i, then i+1, then i+n-1 (mod n) is good enough.
 *
 *  Assuming 6 threads, co-primes are [1, 5], which means the following permutations
 *  assuming we start with victim 0:
 *  - [0, 1, 2, 3, 4, 5]
 *  - [0, 5, 4, 3, 2, 1]
 *  While we don't care much about randoness quality, it's a bit disappointing.
 *
 *  Alternative 2. We can choose m to be the next power of 2, meaning all odd integers are co-primes,
 *  consequently:
 *  - we don't need a GCD to find the coprimes
 *  - we don't need to cache coprimes, removing a cache-miss potential
 *  - a != 1, so we now have a multiplicative factor, which makes output more "random looking".
 */
typedef struct tpc_perm {
    uint32_t a, c, mask, start, x, maxExclusive;
    bool done;
} tpc_perm;

void tpc_perm_init(tpc_perm* perm, uint32_t randomSeed, int32_t maxExclusive);
/** Next victim, or -1 when the cycle completed. */
int32_t tpc_perm_next(tpc_perm* perm);

typedef struct tpc_threadpool tpc_threadpool;

/** Worker termination signal, declared at file scope: C++ scopes a struct tag declared inside a struct to that struct. */
struct tpc_worker_signal {
    TPC_ALIGNAS(64) TPC_ATOMIC(bool) terminate;
};

/** Per-worker, thread-local context.
 *  We assume that a threadpool has exclusive ownership.
 */
typedef struct tpc_worker_context {
    /* Params */
    int32_t id;
    tpc_threadpool* threadpool;

    /* Tasks */
    tpc_taskqueue* taskqueue;      /* owned task queue */
    tpc_task* currentTask;

    /* Synchronization */
    tpc_scoped_barrier* currentScope;   /* need to be exported for syncScope template */

    struct tpc_worker_signal* signal;   /* owned signal */

    /* Thefts */
    tpc_work_stealing_rng rng;     /* RNG state to select victims */
    tpc_counters counters;
} tpc_worker_context;

/** Thread-local Worker context
 *  We assume that a threadpool has exclusive ownership
 *
 *  TODO: if we want to allow non-conflicting threadpools instantiated by the same thread:
 *        - only the threadID per threadpool should be stored and the associated
 *          context should be stored at the Threadpool-level.
 *          Then we need to associate threadpool pointer to workerID in that threadpool
 *        - Or we completely remove thread-local data
 *          and use a Minimal Perfect Hash Function.
 *          We can approximate a threadID by retrieving the address of a dummy thread-local variable.
 *        - Or we sort threadID and use binary search
 *        The FlowVar would also need to store the Threadpool
 */
tpc_worker_context* tpc_workerContext(void);

typedef struct tpc_threadpool_args {
    struct tpc_threadpool* tp;
    int32_t id;
} tpc_threadpool_args;

struct tpc_threadpool {
    /* All synchronization objects are put in their own cache-line to avoid invalidating
     * and reloading cache for immutable fields
     */
    tpc_sync_barrier barrier;                       /* Barrier for initialization and teardown */
    /* -- align: 64 */
    TPC_ALIGNAS(64) tpc_eventcount globalBackoff;      /* Multi-Producer Multi-Consumer backoff */
    /* -- align: 64 */
    TPC_ALIGNAS(64) int32_t numThreads;                /* N regular workers */
    tpc_taskqueue* workerQueues;                    /* size N */
    tpc_thread_t* workers;                          /* size N */
    struct tpc_worker_signal* workerSignals;        /* size N */
    tpc_threadpool_args* workerArgs;
};

/** Scheduler counters of the calling worker, thread-local. */
const tpc_counters* tpc_threadpool_workerCounters(void);

/** Create a pool with numThreads workers.
 *
 *  Returns: the pool. numThreads >= 1 is a caller contract, allocation
 *           failure aborts.
 *
 *  Parameters:
 *  In:      numThreads:  workers 1..numThreads-1 run on new threads, worker 0 is the calling thread.
 */
tpc_threadpool* tpc_threadpool_new(int32_t numThreads);

/** Sync the root scope, signal termination, and join every worker.
 *
 *  Parameters:
 *  Context: tp:  the pool returned by tpc_threadpool_new.
 */
void tpc_threadpool_shutdown(tpc_threadpool* tp);

/** Run a task, frees it if it is not owned by a Flowvar.
 *  Scheduler internals shared with parallel_offloading.
 */
void tpc_run(tpc_worker_context* ctx, tpc_task* task);

/* Task notification overview
 *
 * 2 strategies can be used to notify idle workers of new tasks entering the runtime
 *
 * 1. "notify-on-new": Always try to wake a worker on the backoff on new tasks.
 * 2. "notify-on-transition": Wake a worker if-and-only-if our queue was empty when scheduling the task.
 *
 * In the second case, we also need a notify on successful theft to maintain the invariant that
 * there is at least a thread looking for work if work is available, or all threads are busy.
 *
 * The notify-on-transition strategy minimizes kernel syscalls at the expense of reading an atomic,
 * our dequeue status (a guaranteed cache miss).
 * This is almost always the better tradeoff.
 * Furthermore, in work-stealing, having an empty dequeue is a good approximation for starvation.
 *
 * We can minimize syscalls in the "notify-on-new" strategy as well by reading the backoff status
 * and checking if there is an idle worker not yet parked or no parked threads at all.
 * In that case we also need a notification on successful theft,
 * in case 2 threads enqueueing work find concurrently the same non-parked idle worker, otherwise one task will be missed.
 *
 * The "notify-on-new" minimizes latency in case a producer enqueues tasks quickly.
 *
 * Lastly, when awaiting a future, a worker can give up its own queue if tasks are unrelated to the awaited task.
 * In "notify-on-transition" strategy, that worker needs to wake up a relay.
 *
 * Concretely on almost-empty tasks like fibonacci or DFS, "notify-on-new" is 10x slower.
 * However, when quickly enqueueing tasks, like Multi-Scalar Multiplication,
 * There is a noticeable ramp-up.This might be solved with steal-half.
 */

/** Schedule a task in the threadpool
 *  This wakes another worker if our local queue is empty
 *  or forceWake is true.
 */
void tpc_schedule(tpc_worker_context* ctx, tpc_task* task, bool forceWake);
/** Try to steal a task. */
tpc_task* tpc_tryStealOne(tpc_worker_context* ctx);

/** Leapfrogging:
 *
 *  - Leapfrogging: a portable technique for implementing efficient futures,
 *    David B. Wagner, Bradley G. Calder, 1993
 *    https://dl.acm.org/doi/10.1145/173284.155354
 *
 *  When awaiting a future, we can look in the thief queue first. They steal when they run out of tasks.
 *  If they have tasks in their queue, it's the task we are awaiting that created them and it will likely be stuck
 *  on those tasks as well, so we need to help them help us.
 */
tpc_task* tpc_tryLeapfrog(tpc_worker_context* ctx, tpc_task* awaitedTask);
/** Each worker thread executes this loop over and over. */
void tpc_eventLoop(tpc_worker_context* ctx);

/** Eagerly complete an awaited FlowVar
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
 *  Out:     parentResult:  resultSize bytes, written on return.
 *  In:      resultSize:    the flowvar's result slot size.
 *          fv:            the awaited flowvar.
 */
void tpc_completeFuture(tpc_flowvar* fv, void* parentResult, size_t resultSize);

/** Blocks until all pending tasks are completed
 *  This MUST only be called from the root scope that created the threadpool
 *
 *  Parameters:
 *  Context: tp:  the calling worker's pool.
 */
void tpc_syncAll(tpc_threadpool* tp);

/** Wait at barrier until all descendant tasks are completed
 *
 *  Parameters:
 *  In:      scopedBarrier:  a barrier with tasks the caller registered.
 */
void tpc_syncScopeWait(tpc_scoped_barrier* scopedBarrier);

/* ############################################################
 * #                                                          #
 * #              Parallel-loops load-balancing               #
 * #                                                          #
 * ############################################################
 *
 * Inpired by
 * - Lazy binary-splitting: a run-time adaptive work-stealing scheduler.
 *   A. Tzannes, G. C. Caragea, R. Barua, and U. Vishkin.
 *   In PPoPP '10, Bangalore, India, January 2010. ACM, pp. 179-190.
 *   https://user.eng.umd.edu/~barua/ppopp164.pdf
 * - Embracing Explicit Communication in Work-Stealing Runtime Systems.
 *   Andreas Prell, 2016
 *   https://epub.uni-bayreuth.de/id/eprint/2990/
 *
 * Instead of splitting loops ahead of time depending on the number of cores,
 * we split just-in-time depending on idle threads.
 * This allows the code to lazily evaluate when it's profitable to split,
 * making parallel-for performance portable to any CPU and any inner algorithm
 * unlike OpenMP or TBB for example, see design.md for performance unportability benchmark numbers.
 * This frees the developer from grain size / work splitting thresholds.
 */

/** We want to dynamically split parallel loops depending on the number of idle threads.
 *  However checking an atomic variable require synchronization which at the very least means
 *  reloading its value in all caches, a guaranteed cache miss. In a tight loop,
 *  this might be a significant cost, especially given that memory is often the bottleneck.
 *
 *  There is no synchronization possible with thieves, unlike Prell PhD thesis.
 *  We want to avoid the worst-case scenario in Tzannes paper, tight-loop with too many available cores
 *  so the producer deque is always empty, leading to it spending all its CPU time splitting loops.
 *  For this we split depending on the numbers of idle CPUs. This prevents also splitting unnecessarily.
 *
 *  Tzannes et al mentions that checking the thread own deque emptiness is a good approximation of system load
 *  with low overhead except in very fine-grained parallelism.
 *  With a better approximation, by checking the number of idle threads we can instead
 *  directly do the correct number of splits or avoid splitting. But this check is costly.
 *
 *  To minimize checking cost while keeping latency low, even in bursty cases,
 *  we use log-log iterated backoff.
 *  - Adversarial Contention Resolution for Simple Channels
 *    Bender, Farach-Colton, He, Kuszmaul, Leiserson, 2005
 *    https://people.csail.mit.edu/bradley/papers/BenderFaHe05.pdf
 */
typedef struct tpc_balancer_backoff {
    int32_t nextCheck;
    uint32_t windowLogSize; /* while loopIndex < lastCheck + 2^windowLogSize, don't recheck. */
    uint32_t round;         /* windowSize += 1 after log(windowLogSize) rounds */
} tpc_balancer_backoff;

/** User body of a parallel-for: one call per iteration. */
typedef void (*tpc_range_fn)(int32_t idx, void* env);

/** Split the task's remaining range for approxIdle idle workers:
 *  the low prefix stays with the task, its loopStop shrinks,
 *  the upper chunks go out for dispatch. Chunk sizes differ
 *  by at most 1, balancing costs one division by the worker count.
 */
typedef struct tpc_upper_split {
    int32_t numChunks;      /* upper chunks, excluding the kept prefix */
    int32_t baseChunkSize;
    int32_t cutoff;         /* chunks below cutoff carry +1 */
    int32_t prefixSteps;    /* steps the task keeps */
} tpc_upper_split;

/** Split the iteration range based on the number of idle threads
 *  returns chunks with parameters (start, stopEx, len)
 *
 *  - Chunks are balanced, their size differs by at most 1.
 *    Balanced workloads are scheduled with overhead similar to static scheduling.
 *  - Split is adaptative, unlike static scheduling or guided scheduling in OpenMP
 *    it is based on idle workers and not the number of cores.
 *    If enough parallelism is exposed, for example due to nested parallelism,
 *    there is no splitting overhead.
 *
 *  - Updates the current task loopStop with the lower range
 *
 *  Unbalanced example:
 *   Splitting 40 iterations on 12 threads
 *   A simple chunking algorithm with division + remainder
 *   will lead to a base chunk size of 40/12 = 3.
 *   3*11 = 33, so last thread will do 7 iterations, more than twice the work.
 *
 *  Note: Each division costs 55 cycles, 55x more than addition/substraction/bit operations
 *        and they can't use instruction-level parallelism.
 *        When dealing with loop ranges and strides we need to carefully craft our code to
 *        only use division where unavoidable: dividing by the number of idle threads.
 *        Loop metadata should allow us to avoid loop-bounds-related divisions completely.
 */
void tpc_splitUpperRanges(tpc_task* task, int32_t curLoopIndex, int32_t approxIdle, tpc_upper_split* split);

/** Publish the upper chunks as loop tasks in the own deque and wake
 *  the idle workers.
 */
void tpc_splitAndDispatchLoop(tpc_worker_context* ctx, tpc_task* task, int32_t curLoopIndex, int32_t approxIdle);

/** Split check for the loop balancer: the loop body calls it every
 *  window: split when idle workers exist and the own deque is empty.
 */
void tpc_loadBalanceLoop(tpc_worker_context* ctx, tpc_task* task, int32_t curLoopIndex, tpc_balancer_backoff* backoff);

/** Loop-task runner: balances, runs the body per iteration,
 *  decrements the step count. The body reads its user env at env + 8
 *  where the first 8 bytes are the task self-pointer.
 */
void tpc_parallelForRunner(tpc_worker_context* ctx, tpc_task* task, tpc_range_fn body);

/** True when the task is the root sentinel. */
static inline bool tpc_isRootTask(const tpc_task* task) {
    return task == TPC_ROOT_TASK;
}

#endif /* TPC_THREADPOOL_H */
