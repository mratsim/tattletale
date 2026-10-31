/* threadpool_compute: the work-stealing deque. */
#include "taskqueues.h"

#include <stdlib.h>
#include <string.h>

static tpc_queuebuf* queuebuf_new(int32_t capacity) {
    /* Tasks have a destructor
     * static:
     *   doAssert supportsCopyMem(T), $T & " must be a (POD) plain-old-data type: no seq, string, ref."
     */
    tpc_queuebuf* buf = malloc(sizeof(tpc_queuebuf) + (size_t)capacity * sizeof(_Atomic(tpc_task*)));
    if (buf == NULL) {
        abort();
    }
    buf->prevRetired = NULL;
    buf->capacity = capacity;
    memset(buf->slots, 0, capacity * sizeof(_Atomic(tpc_task*)));
    return buf;
}

void tpc_taskqueue_init(tpc_taskqueue* tq, int32_t capPow2) {
    memset(tq, 0, sizeof(*tq));
    atomic_store_explicit(&tq->buf, queuebuf_new(capPow2), memory_order_relaxed);
}

/** Retire the buffer and free every buffer that a subsequent empty-deque
 *  owner operation finds in the chain: a thief that read a retired
 *  buffer has already lost its compare-exchange by the time the deque
 *  is empty again.
 */
static void queuebuf_retire(tpc_taskqueue* tq, tpc_queuebuf* old) {
    old->prevRetired = tq->garbage;
    tq->garbage = old;
}

static void queuebuf_garbageCollect(tpc_taskqueue* tq) {
    tpc_queuebuf* node = tq->garbage;
    while (node != NULL) {
        tpc_queuebuf* next = node->prevRetired;
        free(node);
        node = next;
    }
    tq->garbage = NULL;
}

/** Double the buffer size
 *  back is the last item index
 *
 *  To handle race-conditions the current "front", "back" and "buf"
 *  have to be saved before calling this procedure.
 *  It reads and writes the "tq.buf" and "tq.garbage"
 */
static void queuebuf_grow(tpc_taskqueue* tq, tpc_queuebuf** buf, int32_t newCapacity, intptr_t front, intptr_t back) {
    /* Read -> Copy -> Update */
    tpc_queuebuf* fresh = queuebuf_new(newCapacity);
    for (int32_t i = front; i < back; i++) {
        atomic_store_explicit(
            &fresh->slots[i & (fresh->capacity - 1)],
            atomic_load_explicit(&(*buf)->slots[i & ((*buf)->capacity - 1)],
                                 memory_order_relaxed),
            memory_order_relaxed);
    }
    queuebuf_retire(tq, *buf);
    /* publish globally, release pairs with the consume load of thieves
     * so the copied slots are visible before the new buffer is */
    atomic_store_explicit(&tq->buf, fresh, memory_order_release);
    /* publish locally */
    *buf = fresh;
}

void tpc_taskqueue_push(tpc_taskqueue* tq, tpc_task* item) {
    intptr_t back = atomic_load_explicit(&tq->back, memory_order_relaxed);
    intptr_t front = atomic_load_explicit(&tq->front, memory_order_acquire);
    tpc_queuebuf* buf = atomic_load_explicit(&tq->buf, memory_order_relaxed);

    if (back - front > (int32_t)buf->capacity - 1) {
        /* Full queue */
        queuebuf_grow(tq, &buf, buf->capacity * 2, front, back);
    }
    if (tq->garbage != NULL && front == back) {
        /* Empty queue, no thieves can have a pointer to an old retired buffer */
        queuebuf_garbageCollect(tq);
    }

    atomic_store_explicit(&buf->slots[back & (buf->capacity - 1)], item, memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit(&tq->back, back + 1, memory_order_relaxed);
}

tpc_task* tpc_taskqueue_pop(tpc_taskqueue* tq) {
    intptr_t back = atomic_load_explicit(&tq->back, memory_order_relaxed) - 1;
    atomic_store_explicit(&tq->back, back, memory_order_relaxed);
    atomic_thread_fence(memory_order_seq_cst);
    intptr_t front = atomic_load_explicit(&tq->front, memory_order_relaxed);
    tpc_queuebuf* buf = atomic_load_explicit(&tq->buf, memory_order_relaxed);

    tpc_task* item = NULL;
    if (front <= back) {
        /* Non-empty queue. */
        item = atomic_load_explicit(&buf->slots[back & (buf->capacity - 1)], memory_order_relaxed);
        if (front == back) {
            /* Single last element in queue. */
            intptr_t expected = front;
            if (!atomic_compare_exchange_strong_explicit(&tq->front, &expected, front + 1, memory_order_seq_cst, memory_order_relaxed)) {
                /* Failed race. */
                item = NULL;
            }
            atomic_store_explicit(&tq->back, back + 1, memory_order_relaxed);
            if (tq->garbage != NULL) {
                queuebuf_garbageCollect(tq);
            }
        }
    } else {
        /* Empty queue. */
        atomic_store_explicit(&tq->back, back + 1, memory_order_relaxed);
        if (tq->garbage != NULL) {
            queuebuf_garbageCollect(tq);
        }
    }
    return item;
}

tpc_task* tpc_taskqueue_steal(uint32_t thiefID, tpc_taskqueue* tq) {
    intptr_t front = atomic_load_explicit(&tq->front, memory_order_acquire);
    atomic_thread_fence(memory_order_seq_cst);
    intptr_t back = atomic_load_explicit(&tq->back, memory_order_acquire);

    if (front >= back) {
        return NULL;
    }
    /* Non-empty queue. */
    tpc_queuebuf* buf = atomic_load_explicit(&tq->buf, memory_order_consume);
    tpc_task* item = atomic_load_explicit(&buf->slots[front & (buf->capacity - 1)], memory_order_relaxed);
    intptr_t expected = front;
    if (!atomic_compare_exchange_strong_explicit(&tq->front, &expected, front + 1, memory_order_seq_cst, memory_order_relaxed)) {
        /* Failed race. */
        return NULL;
    }
    tpc_task_setThief(item, thiefID);
    return item;
}

intptr_t tpc_taskqueue_peek(tpc_taskqueue* tq) {
    intptr_t back = atomic_load_explicit(&tq->back, memory_order_relaxed);
    intptr_t front = atomic_load_explicit(&tq->front, memory_order_acquire);
    return back >= front ? back - front : 0;
}

void tpc_taskqueue_teardown(tpc_taskqueue* tq) {
    queuebuf_garbageCollect(tq);
    free(atomic_load_explicit(&tq->buf, memory_order_relaxed));
}
