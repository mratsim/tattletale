/* flow_events: dataflow task dependencies.
 * C port of weave's cross_thread_com/flow_events.nim.
 * include/flow_events.h documents the protocol and the public API.
 */
#include "flow_events.h"

#include <stdio.h>
#include <stdlib.h>

#include "../src/threadpool/instrumentation.h"

/* TaskNode allocation
 * --------------------------------------------------------------- */

/** Allocates a zeroed TaskNode for `task` with bucketID `bucketID`.
 *
 *  dep is the node's own dependency, e == NULL for a single-dependency node.
 *  A non-NULL dep gains one event reference, released when the node frees.
 */
static tpc_tasknode* tpc_tasknode_new(tpc_task* task, tpc_flow_event dep, int32_t bucketID) {
    tpc_tasknode* node = (tpc_tasknode*)calloc(1, sizeof(tpc_tasknode));
    if (node == NULL) {
        fprintf(stderr, "flow_events: TaskNode allocation failed\n");
        abort();
    }
    node->task = task;
    node->event = dep;
    if (dep.e != NULL) {
        tpc_flow_event_ref(dep);
    }
    node->bucketID = bucketID;
    return node;
}

/** Frees a TaskNode, releasing its event reference when one exists. */
static void tpc_tasknode_free(tpc_tasknode* node) {
    if (node->event.e != NULL) {
        tpc_flow_event_unref(&node->event);
    }
    free(node);
}

/** Converts an iteration index to a channel bucket in [0, numBuckets). */
static int32_t tpc_event_bucket(const tpc_event* e, int32_t index) {
    tpc_preCondition(e->kind == tpc_event_iteration);
    tpc_preCondition(e->u.iter.start <= index && index < e->u.iter.stop);
    return (index - e->u.iter.start) / e->u.iter.stride;
}

/** Allocates one channel bucket set of `numBuckets` entries, aborting on exhaustion. */
static tpc_single_event* tpc_single_event_alloc(int32_t numBuckets) {
    tpc_single_event* singles = (tpc_single_event*)calloc((size_t)numBuckets, sizeof(tpc_single_event));
    if (singles == NULL) {
        fprintf(stderr, "flow_events: event bucket allocation failed\n");
        abort();
    }
    for (int32_t i = 0; i < numBuckets; i++) {
        pthread_mutex_init(&singles[i].lock, NULL);
    }
    return singles;
}

/* Deferring
 * --------------------------------------------------------------- */

/** Defers the node on event `e`, bucket `bucketID` (TPC_NO_ITER for a single event)
 *  unless the event already triggered.
 *
 *  Returns true when the node was delayed: ownership moved to the event
 *  channel and the triggerer schedules the task. Returns false: the caller
 *  owns the node again and schedules the task itself.
 *
 *  Protocol: the consumer enters deferredIn, rechecks the write-once
 *  triggered flag under the channel lock, sends and leaves deferredOut.
 *  The dual monotonic counters let the producer detect a consumer still
 *  mid-delay and re-drain, avoiding the ABA problem.
 */
static bool tpc_event_delayNode(tpc_event* e, int32_t bucketID, tpc_tasknode* node) {
    tpc_single_event* se = (e->kind == tpc_event_single) ? &e->u.single : &e->u.iter.singles[bucketID];

    /* Fast path: the write-once flag already fired */
    if (atomic_load_explicit(&se->triggered, memory_order_relaxed)) {
        atomic_thread_fence(memory_order_acquire);
        return false;
    }

    atomic_fetch_add_explicit(&se->deferredIn, 1, memory_order_relaxed);
    pthread_mutex_lock(&se->lock);
    if (atomic_load_explicit(&se->triggered, memory_order_acquire)) {
        pthread_mutex_unlock(&se->lock);
        atomic_fetch_add_explicit(&se->deferredOut, 1, memory_order_relaxed);
        return false;
    }
    node->next = NULL;
    if (se->chanTail == NULL) {
        se->chanHead = node;
    } else {
        se->chanTail->next = node;
    }
    se->chanTail = node;
    atomic_fetch_add_explicit(&se->deferredOut, 1, memory_order_relaxed);
    pthread_mutex_unlock(&se->lock);
    return true;
}

bool tpc_flow_event_delayedUntil(tpc_task* task, tpc_flow_event ev) {
    tpc_event* e = ev.e;
    tpc_preCondition(e != NULL);
    tpc_preCondition(e->kind == tpc_event_single);

    tpc_tasknode* node = tpc_tasknode_new(task, (tpc_flow_event){0}, TPC_NO_ITER);
    if (!tpc_event_delayNode(e, TPC_NO_ITER, node)) {
        tpc_tasknode_free(node);
        return false;
    }
    return true;
}

bool tpc_flow_event_delayedUntilIter(tpc_task* task, tpc_flow_event ev, int32_t index) {
    tpc_event* e = ev.e;
    tpc_preCondition(e != NULL);
    tpc_preCondition(e->kind == tpc_event_iteration);

    int32_t bucketID = tpc_event_bucket(e, index);
    tpc_tasknode* node = tpc_tasknode_new(task, (tpc_flow_event){0}, bucketID);
    if (!tpc_event_delayNode(e, bucketID, node)) {
        tpc_tasknode_free(node);
        return false;
    }
    return true;
}

bool tpc_flow_event_delayedUntilMulti(tpc_task* task, const tpc_flow_event_dep* deps, int32_t n) {
    tpc_preCondition(n > 0);

    tpc_tasknode* head = NULL;
    tpc_tasknode* tail = NULL;
    for (int32_t i = 0; i < n; i++) {
        tpc_event* e = deps[i].event.e;
        tpc_preCondition(e != NULL);
        int32_t bucketID = TPC_NO_ITER;
        if (e->kind == tpc_event_iteration) {
            bucketID = tpc_event_bucket(e, deps[i].index);
        }
        tpc_tasknode* node = tpc_tasknode_new(task, deps[i].event, bucketID);
        node->nextDep = NULL;
        if (tail == NULL) {
            head = node;
        } else {
            tail->nextDep = node;
        }
        tail = node;
    }

    /* Defer on the first still-untriggered dependency. The node of an already
     * triggered dependency frees during the walk and the remaining chain
     * travels with the re-deferred node.
     */
    tpc_tasknode* node = head;
    while (node != NULL) {
        if (tpc_event_delayNode(node->event.e, node->bucketID, node)) {
            return true;
        }
        tpc_tasknode* nextDep = node->nextDep;
        tpc_tasknode_free(node);
        node = nextDep;
    }
    return false;
}

/* Triggering
 * --------------------------------------------------------------- */

/** Drains the channel under the caller-held lock. Every drained task is
 *  scheduled through enqueue, unless a nextDep chain re-delays the task
 *  on a still-untriggered dependency. The remaining chain moves
 *  with the re-deferred node.
 */
static void tpc_event_drain(tpc_single_event* se, void* queue, tpc_flow_event_enqueue enqueue) {
    tpc_tasknode* node = se->chanHead;
    se->chanHead = NULL;
    se->chanTail = NULL;
    while (node != NULL) {
        tpc_tasknode* nextInChannel = node->next;
        tpc_task* task = node->task;
        bool wasDelayed = false;
        /* Re-delay attempt on every chain node. The anchor of a single-dependency
         * delay has a NULL event and counts as fired.
         * Only a chain with every dependency fired enqueues its task.
         * A successful re-delay parks the remaining chain on that channel.
         */
        while (node != NULL && node->event.e != NULL) {
            if (tpc_event_delayNode(node->event.e, node->bucketID, node)) {
                wasDelayed = true;
                break;
            }
            tpc_tasknode* nextDep = node->nextDep;
            tpc_tasknode_free(node);
            node = nextDep;
        }
        if (!wasDelayed) {
            enqueue(queue, task);
            if (node != NULL) {
                tpc_tasknode_free(node);
            }
        }
        node = nextInChannel;
    }
}

/** Drains and re-drains until every consumer that entered deferredIn has
 *  left through deferredOut. The channel is then provably empty.
 */
static void tpc_event_triggerChannel(tpc_single_event* se, void* queue, tpc_flow_event_enqueue enqueue) {
    while (true) {
        pthread_mutex_lock(&se->lock);
        tpc_event_drain(se, queue, enqueue);
        pthread_mutex_unlock(&se->lock);
        if (atomic_load_explicit(&se->deferredOut, memory_order_acquire) == atomic_load_explicit(&se->deferredIn, memory_order_acquire)) {
            break;
        }
        tpc_cpuRelax();
    }
}

void tpc_flow_event_trigger(tpc_flow_event ev, void* queue, tpc_flow_event_enqueue enqueue) {
    tpc_event* e = ev.e;
    tpc_preCondition(e != NULL);
    tpc_preCondition(e->kind == tpc_event_single);
    tpc_preCondition(!atomic_load_explicit(&e->u.single.triggered, memory_order_relaxed));

    /* Write-once flag: a consumer arriving after this store schedules
     * its task itself. The release fence orders the channel contents
     * before the flag
     */
    atomic_store_explicit(&e->u.single.triggered, true, memory_order_release);
    tpc_event_triggerChannel(&e->u.single, queue, enqueue);
}

void tpc_flow_event_triggerIter(tpc_flow_event ev, int32_t index, void* queue, tpc_flow_event_enqueue enqueue) {
    tpc_event* e = ev.e;
    tpc_preCondition(e != NULL);
    tpc_preCondition(e->kind == tpc_event_iteration);

    int32_t bucketID = tpc_event_bucket(e, index);
    tpc_single_event* se = &e->u.iter.singles[bucketID];
    tpc_preCondition(!atomic_load_explicit(&se->triggered, memory_order_relaxed));

    atomic_store_explicit(&se->triggered, true, memory_order_release);
    tpc_event_triggerChannel(se, queue, enqueue);
}

/* Lifecycle
 * --------------------------------------------------------------- */

void tpc_flow_event_initSingle(tpc_flow_event* ev) {
    tpc_preCondition(ev->e == NULL);

    tpc_event* e = (tpc_event*)calloc(1, sizeof(tpc_event));
    if (e == NULL) {
        fprintf(stderr, "flow_events: event allocation failed\n");
        abort();
    }
    e->kind = tpc_event_single;
    atomic_store_explicit(&e->refCount, 1, memory_order_relaxed);
    pthread_mutex_init(&e->u.single.lock, NULL);
    ev->e = e;
}

void tpc_flow_event_initIter(tpc_flow_event* ev, int32_t start, int32_t stop, int32_t stride) {
    tpc_preCondition(stop > start);
    tpc_preCondition(stride > 0);
    tpc_preCondition(ev->e == NULL);

    int64_t numBuckets = ((int64_t)stop - start + stride - 1) / stride;
    if (numBuckets > INT32_MAX) {
        fprintf(stderr, "flow_events: iteration event bucket count overflows int32\n");
        abort();
    }
    tpc_event* e = (tpc_event*)calloc(1, sizeof(tpc_event));
    if (e == NULL) {
        fprintf(stderr, "flow_events: event allocation failed\n");
        abort();
    }
    e->kind = tpc_event_iteration;
    atomic_store_explicit(&e->refCount, 1, memory_order_relaxed);
    e->u.iter.numBuckets = (int32_t)numBuckets;
    e->u.iter.start = start;
    e->u.iter.stop = stop;
    e->u.iter.stride = stride;
    e->u.iter.singles = tpc_single_event_alloc(e->u.iter.numBuckets);
    ev->e = e;
}

void tpc_flow_event_ref(tpc_flow_event ev) {
    tpc_preCondition(ev.e != NULL);
    atomic_fetch_add_explicit(&ev.e->refCount, 1, memory_order_relaxed);
}

void tpc_flow_event_unref(tpc_flow_event* ev) {
    tpc_event* e = ev->e;
    if (e == NULL) {
        return;
    }
    if (atomic_fetch_sub_explicit(&e->refCount, 1, memory_order_release) != 1) {
        ev->e = NULL;
        return;
    }
    atomic_thread_fence(memory_order_acquire);
    if (e->kind == tpc_event_iteration) {
        for (int32_t i = 0; i < e->u.iter.numBuckets; i++) {
            pthread_mutex_destroy(&e->u.iter.singles[i].lock);
        }
        free(e->u.iter.singles);
    } else {
        pthread_mutex_destroy(&e->u.single.lock);
    }
    free(e);
    ev->e = NULL;
}
