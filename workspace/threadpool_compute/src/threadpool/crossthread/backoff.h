/* threadpool_compute: the parking primitive: a lock-free eventcount. */
#ifndef TPC_BACKOFF_H
#define TPC_BACKOFF_H

#include <stdatomic.h>
#include <stdint.h>
#include "../primitives/futexes.h"

/* ############################################################
 * #
 * #                      Eventcount
 * #
 * ############################################################ */

typedef struct tpc_eventcount {
    /** The lock-free equivalent of a condition variable.
     *
     *  Usage, if a thread needs to be parked until a condition is true
     *        and signaled by another thread:
     *  ```
     *  if (condition)
     *      return;
     *
     *  while (true) {
     *      ticket = tpc_eventcount_sleepy(&ec);
     *      if (condition) {
     *          tpc_eventcount_cancelSleep(&ec);
     *          break;
     *      } else {
     *          tpc_eventcount_sleep(&ec, ticket);
     *      }
     *  }
     *  ```
     */
    _Atomic uint32_t waitset;
    /* type waitset = object
     *   preSleep {.bitsize: 16.}: uint32
     *   committedSleep {.bitsize: 16.}: uint32
     *
     * We need precise committed sleep count for the `syncAll` barrier because a `preSleep` waiter
     * may steal a task and create more work.
     */
    _Atomic uint32_t events;   /* futex word */
} tpc_eventcount;

/** A ticket from sleepy(). The sleep() call blocks until the event
 *  counter leaves this epoch.
 */
typedef struct tpc_parking_ticket {
    uint32_t epoch;
} tpc_parking_ticket;

/* bitfield setup
 *   Low 16 bits are waiters, up to 2¹⁶ = 65536 threads are supported
 *   Next 16 bits are pre-waiters, planning to wait but not committed.
 *
 * OS limitations:
 * - Windows 10 supports up to 256 cores (https://www.microsoft.com/en-us/microsoft-365/blog/2017/12/15/windows-10-pro-workstations-power-advanced-workloads/)
 * - Linux CPUSET supports up to 1024 threads (https://man7.org/linux/man-pages/man3/CPU_SET.3.html)
 *
 * Hardware limitations:
 * - Xeon Platinum 8490H, 60C/120T per socket
 *   - up to 8 sockets: 960 threads
 */
#define TPC_K_PRE_WAIT_SHIFT 16u
#define TPC_K_PRE_WAIT       (1u << TPC_K_PRE_WAIT_SHIFT)
#define TPC_K_WAIT           1u
#define TPC_K_COMMIT_TO_WAIT (TPC_K_WAIT - TPC_K_PRE_WAIT)
#define TPC_K_WAIT_MASK      (TPC_K_PRE_WAIT - 1u)
#define TPC_K_PRE_WAIT_MASK  (~TPC_K_WAIT_MASK)

/* Implementations
 * ---------------------------------------------------------------- */

/** Reset a fresh eventcount to the empty state. */
static inline void tpc_eventcount_initialize(tpc_eventcount* ec) {
    atomic_store_explicit(&ec->waitset, 0, memory_order_relaxed);
    atomic_store_explicit(&ec->events, 0, memory_order_relaxed);
}

/** To be called before checking if the condition to not sleep is met.
 *  Returns a ticket to be used when committing to sleep
 */
static inline tpc_parking_ticket tpc_eventcount_sleepy(tpc_eventcount* ec) {
    atomic_fetch_add_explicit(&ec->waitset, TPC_K_PRE_WAIT, memory_order_release);
    tpc_parking_ticket ticket;
    ticket.epoch = atomic_load_explicit(&ec->events, memory_order_acquire);
    return ticket;
}

/** Put a thread to sleep until notified.
 *  If the ticket becomes invalid (a notification has been received)
 *  by the time sleep is called, the thread won't enter sleep
 */
static inline void tpc_eventcount_sleep(tpc_eventcount* ec, tpc_parking_ticket ticket) {
    /* commit: prewait -1, committed +1, in one subtractive step */
    atomic_fetch_add_explicit(&ec->waitset, TPC_K_COMMIT_TO_WAIT, memory_order_release);
    while (atomic_load_explicit(&ec->events, memory_order_acquire) == ticket.epoch) {
        tpc_futex_wait(&ec->events, ticket.epoch);
    }
    atomic_fetch_sub_explicit(&ec->waitset, TPC_K_WAIT, memory_order_release);
}

/** Cancel a sleep that was scheduled. */
static inline void tpc_eventcount_cancelSleep(tpc_eventcount* ec) {
    atomic_fetch_sub_explicit(&ec->waitset, TPC_K_PRE_WAIT, memory_order_release);
}

/** Prevent an idle thread from sleeping
 *  or wait a sleeping one if there wasn't any idle
 *
 *  1. wake one committed sleeper,
 *  2. a wake firing while prewaiters exist is absorbed by them:
 *     their commit re-checks the counter and skips the syscall,
 *  3. an absorbed wake loses nothing: the scheduler's retry loops
 *     re-run their checks and find the work in a deque.
 */
static inline void tpc_eventcount_wake(tpc_eventcount* ec) {
    atomic_fetch_add_explicit(&ec->events, 1, memory_order_release);
    uint32_t waiters = atomic_load_explicit(&ec->waitset, memory_order_acquire);
    if ((waiters & TPC_K_PRE_WAIT_MASK) != 0) {
        /* Some threads are in prewait and will see the event count change
         * no need to do an expensive syscall
         */
        return;
    }
    if (waiters != 0) {
        tpc_futex_wake(&ec->events, 1);
    }
}

/** Wake all threads if at least 1 is parked */
static inline void tpc_eventcount_wakeAll(tpc_eventcount* ec) {
    atomic_fetch_add_explicit(&ec->events, 1, memory_order_release);
    uint32_t waiters = atomic_load_explicit(&ec->waitset, memory_order_acquire);
    if ((waiters & TPC_K_WAIT_MASK) != 0) {
        tpc_futex_wake_all(&ec->events);
    }
}

/** Get the number of idle threads:
 *  (preSleep, committedSleep)
 *
 *  The parallel-for balancer decides from them whether idle workers
 *  exist and a split pays.
 *
 *  Parameters:
 *  Out:     preSleep:        ticket holders not yet committed.
 *          committedSleep:  sleepers blocked in the kernel futex.
 *  In:      ec:              the pool's eventcount.
 */
static inline void tpc_eventcount_getNumWaiters(const tpc_eventcount* ec, int32_t* preSleep, int32_t* committedSleep) {
    uint32_t waiters = atomic_load_explicit(&ec->waitset, memory_order_acquire);
    *preSleep = (int32_t)((waiters & TPC_K_PRE_WAIT_MASK) >>
                          TPC_K_PRE_WAIT_SHIFT);
    *committedSleep = (int32_t)(waiters & TPC_K_WAIT_MASK);
}

#endif /* TPC_BACKOFF_H */
