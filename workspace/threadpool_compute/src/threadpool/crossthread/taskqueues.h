/* threadpool_compute: work-stealing deque.
 *
 * This file implements a single-producer multi-consumer
 * task queue for work-stealing schedulers.
 *
 * Papers:
 * - Dynamic Circular Work-Stealing Deque
 *   David Chase, Yossi Lev, 1993
 *   https://www.dre.vanderbilt.edu/~schmidt/PDF/work-stealing-dequeue.pdf
 *
 * - Correct and Efficient Work-Stealing for Weak Memory Models
 *   Nhat Minh Lê, Antoniu Pop, Albert Cohen, Francesco Zappa Nardelli, 2013
 *   https://fzn.fr/readings/ppopp13.pdf
 *
 * The task queue implements the following push, pop, steal
 *
 *    front                                              back
 *                 ---------------------------------
 *  steal()     <- |         |          |          | <- push()
 *                 | Task 0  |  Task 1  |  Task 2  | -> pop()
 *  any thread     |         |          |          |    owner-only
 *                 ---------------------------------
 *
 * To reduce contention, stealing is done on the opposite end from push/pop
 * so that there is a race only for the very last task.
 */
#ifndef TPC_TASKQUEUES_H
#define TPC_TASKQUEUES_H

#include <stdatomic.h>
#include <stdint.h>

typedef struct tpc_task tpc_task;

/* Backend buffer of a Taskqueue
 * `capacity` MUST be a power of 2
 */
typedef struct tpc_queuebuf {
    struct tpc_queuebuf* prevRetired;   /* intrusive linked list. Used for garbage collection */
    uint32_t capacity;
    _Atomic(tpc_task*) slots[];
} tpc_queuebuf;

/* This implements a lock-free, growable, work-stealing task queue.
 * The owning thread enqueues and dequeues at the back
 * Foreign threads steal at the front.
 *
 * The front and back indices are machine-word sized so the compare-exchange
 * stays lock-free on every platform, a 64-bit index on 64-bit targets and
 * a 32-bit index where the atomics are 32-bit.
 */
typedef struct tpc_taskqueue {
    _Alignas(64) _Atomic intptr_t front;   /* Consumers - steal */
    _Alignas(64) _Atomic intptr_t back;    /* Producer  - push/pop */
    _Alignas(64) _Atomic(tpc_queuebuf*) buf;
    struct tpc_queuebuf* garbage;          /* retired buffers */
} tpc_taskqueue;

/** Estimates the number of items pending in the channel
 *  In a SPMC setting
 *  - If called by the producer the true number might be less
 *    due to consumers removing items concurrently.
 *  - If called by a consumer the true number is undefined
 *    as other consumers also remove items concurrently and
 *    the producer removes them concurrently.
 *
 *  If the producer peeks and the Chase-Lev Deque returns 0,
 *  the queue is empty.
 *
 *  This is a non-locking operation.
 *
 *  Returns: back - front when the owner calls it, 0 when a thief calls it
 *                   on a deque the owner concurrently emptied.
 *                   Estimates stay undefined for thieves otherwise.
 */
intptr_t tpc_taskqueue_peek(tpc_taskqueue* tq);

/** Initialize a deque with an initial capacity of capPow2 slots. */
void tpc_taskqueue_init(tpc_taskqueue* tq, int32_t capPow2);

/** Free the deque and every retired buffer. Retired buffers are freed
 *  only by an owner operation that leaves the deque empty, a thief cannot
 *  win a compare-exchange on the retired generation.
 *  Requires a quiescent deque (no thief mid-steal).
 */
void tpc_taskqueue_teardown(tpc_taskqueue* tq);

/** Enqueue an item at the back
 *  As the task queue takes ownership of it. The item must not be used afterwards.
 *  This is intended for the producer only.
 *
 *  The owner and a thief race only for the very last item.
 *  It resolves in the seq_cst compare-exchange on front (Chase-Lev 1993, Lê PPoPP 2013).
 *
 *  Parameters:
 *  In:      tq:    the calling worker's deque.
 *          item:  the task to enqueue, the deque takes ownership.
 */
void tpc_taskqueue_push(tpc_taskqueue* tq, tpc_task* item);

/** Dequeue an item at the back. Takes ownership of the item
 *  This is intended for the producer only.
 *
 *
 *  Returns: the task, or NULL when the deque is empty or a thief won
 *                   the race for the last item.
 */
tpc_task* tpc_taskqueue_pop(tpc_taskqueue* tq);

/** Dequeue an item at the front. Takes ownership of the item
 *  This is intended for consumers.
 *
 *
 *  Returns: the task, or NULL when the victim is empty or the owner won
 *                   the race.
 */
tpc_task* tpc_taskqueue_steal(uint32_t thiefID, tpc_taskqueue* tq);

#endif /* TPC_TASKQUEUES_H */
