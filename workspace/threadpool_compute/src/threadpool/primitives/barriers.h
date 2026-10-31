/* threadpool_compute: initialization and teardown barrier. */
#ifndef TPC_BARRIERS_H
#define TPC_BARRIERS_H

#include <stdatomic.h>
#include <stdint.h>


typedef struct tpc_sync_barrier {
    _Alignas(64) _Atomic uint32_t arrived;    /* arrivals this phase */
    _Alignas(64) _Atomic uint32_t phase;      /* released generation */
    uint32_t numThreads;
} tpc_sync_barrier;

/* Implementations
 * ---------------------------------------------------------------- */

/** Initialize a synchronization barrier that will block numThreads
 *  threads before release.
 */
static inline void tpc_barrier_init(tpc_sync_barrier* barrier, uint32_t numThreads) {
    atomic_store_explicit(&barrier->arrived, 0, memory_order_relaxed);
    atomic_store_explicit(&barrier->phase, 0, memory_order_relaxed);
    barrier->numThreads = numThreads;
}

/** Blocks thread at a synchronization barrier.
 *  The last arrival releases everyone by bumping the generation.
 *  Reuse across phases works because a waiter re-checks the generation
 *  it slept on before sleeping again.
 */
static inline void tpc_barrier_wait(tpc_sync_barrier* barrier) {
    /* phase snapshot of the arriving thread: if the phase moved past it,
     * release already fired
     */
    uint32_t phase = atomic_load_explicit(&barrier->phase, memory_order_acquire);
    uint32_t arrived = atomic_fetch_add_explicit(&barrier->arrived, 1, memory_order_acq_rel) + 1;
    if (arrived == barrier->numThreads) {
        /* last arrival: reset the counter for the next phase and release it,
         * this release orders the reset before any participant's
         * next-phase arrival
         */
        atomic_store_explicit(&barrier->arrived, 0, memory_order_relaxed);
        atomic_store_explicit(&barrier->phase, phase + 1, memory_order_release);
        tpc_futex_wake_all(&barrier->phase);
        return;
    }
    while (atomic_load_explicit(&barrier->phase, memory_order_acquire) == phase) {
        tpc_futex_wait(&barrier->phase, phase);
    }
}

/** Deletes a synchronization barrier.
 *  This assumes no race between waiting at a barrier and deleting it,
 *  and reuse of the barrier requires initialization.
 */
static inline void tpc_barrier_delete(tpc_sync_barrier* barrier) {
    (void)barrier;
}


#endif /* TPC_BARRIERS_H */
