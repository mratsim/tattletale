/* flow_events: dataflow task dependencies.
 * C port of weave's cross_thread_com/flow_events.nim.
 *
 * A flow event is a contract between a producer task that triggers an event
 * and consumer tasks deferred until it fires.
 * A flowvar is a delayed result, a flow event is a delayed input.
 * It models precise data dependencies beyond control-flow barriers:
 *
 * - dataflow / graph parallelism
 * - pipeline parallelism
 * - data-driven task parallelism
 *
 * Protocol (weave PR #92):
 *
 *   Consumer (deferredUntil):
 *   1. fast path: the event already triggered. Return false and the caller
 *      schedules the task itself.
 *   2. deferredIn.fetchAdd(1), then recheck triggered under the channel
 *      lock: the producer may have fired in between. When the recheck sees
 *      the flag set, deferredOut.fetchAdd(1) and return false.
 *   3. otherwise send the TaskNode to the event channel, add
 *      deferredOut.fetchAdd(1), and return true.
 *
 *   Producer (trigger):
 *   1. write-once: store triggered = true (release fence), then drain the channel.
 *      Every drained task is scheduled, unless its TaskNode has a nextDep chain.
 *      Each chain entry is re-deferred on its own event, and on the first
 *      successful re-delay the remaining chain moves with it.
 *   2. spin on deferredIn == deferredOut and re-drain while a consumer is
 *      mid-delay. The two monotonic counters avoid the ABA problem.
 *
 * Multi-dependency: one TaskNode per event, linked through nextDep in dependency order.
 * Its node enters the first event's channel. Triggering an event drains its node
 * and re-defers the task on the first still-untriggered event of the chain.
 * Remaining chain entries move with the re-deferred node.
 *
 * Memory: malloc/free with no memory pool. Events are reference counted,
 * one reference per FlowEvent handle and refCount starts at 1 on init.
 * Single-dependency TaskNodes reference no event. A multi-dependency node
 * references its event once and releases it when the node frees.
 * unref frees the event. Caller contract: every dependent task was
 * drained or re-deferred, so the channels are empty at unref time.
 * unref must not race an in-flight trigger.
 */
#ifndef TPC_FLOW_EVENTS_H
#define TPC_FLOW_EVENTS_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include "../src/threadpool/primitives/atomics_compat.h"
#include "../src/threadpool/crossthread/tasks_flowvars.h"

#ifdef __cplusplus
extern "C" {
#endif

/** bucketID of a TaskNode that depends on a single (non-iteration) event. */
#define TPC_NO_ITER ((int32_t)-1)

/** Reference-counted event handle (weave's EventPtr). */
typedef struct tpc_event tpc_event;

typedef struct tpc_flow_event {
    tpc_event* e;
} tpc_flow_event;

typedef struct tpc_tasknode {
    tpc_task* task;
    struct tpc_tasknode* next;      /* next node in the event channel */
    struct tpc_tasknode* nextDep;   /* next dependency of the same task */
    tpc_flow_event event;           /* this node's own dependency, set only for multi-dependency nodes */
    int32_t bucketID;
} tpc_tasknode;

typedef enum tpc_event_kind {
    tpc_event_single,
    tpc_event_iteration
} tpc_event_kind;

/* One event channel: a mutex-guarded FIFO of TaskNodes plus the dual
 * deferred counters that let the producer detect consumers mid-delay.
 * These counters and the write-once triggered flag keep the weave protocol.
 * This channel replaces weave's lock-free MPSC with a mutex.
 * This port allocates no memory pool.
 */
typedef struct tpc_single_event {
    pthread_mutex_t lock;
    tpc_tasknode* chanHead;         /* FIFO: nodes pop from the head */
    tpc_tasknode* chanTail;
    TPC_ATOMIC(int32_t) deferredIn;
    TPC_ATOMIC(int32_t) deferredOut;
    TPC_ATOMIC(bool) triggered;
} tpc_single_event;

typedef struct tpc_event_iter {
    int32_t numBuckets;
    int32_t start, stop, stride;
    tpc_single_event* singles;      /* numBuckets entries, heap allocated */
} tpc_event_iter;

struct tpc_event {
    TPC_ATOMIC(int32_t) refCount;
    tpc_event_kind kind;
    union {
        tpc_single_event single;
        tpc_event_iter iter;
    } u;
};

/** One dependency of a multi-dependency task.
 *
 *  event: the awaited event.
 *  index: TPC_NO_ITER for a single event, otherwise the iteration index.
 */
typedef struct tpc_flow_event_dep {
    tpc_flow_event event;
    int32_t index;
} tpc_flow_event_dep;

/** Callback that schedules a task freed from a triggered event's channel.
 *  queue is the opaque target passed to trigger: a worker context, a test stack,
 *  or the runtime ready queue.
 */
typedef void (*tpc_flow_event_enqueue)(void* queue, tpc_task* task);

/* Lifecycle
 * --------------------------------------------------------------- */

/** Initializes a single event, reference count 1.
 *  Aborts when the event is already initialized.
 */
void tpc_flow_event_initSingle(tpc_flow_event* ev);

/** Initializes an iteration event over [start, stop) with the given stride,
 *  one channel bucket per iteration, reference count 1.
 *  Aborts unless stop > start, stride > 0 and the event is uninitialized.
 */
void tpc_flow_event_initIter(tpc_flow_event* ev, int32_t start, int32_t stop, int32_t stride);

/** Adds one reference to the event behind the handle. */
void tpc_flow_event_ref(tpc_flow_event ev);

/** Drops one reference and frees the event on the last one, e set to NULL.
 *  Every dependent task must already be drained or re-deferred, so every
 *  channel is empty at unref time. The caller must not race a trigger
 *  still in flight.
 */
void tpc_flow_event_unref(tpc_flow_event* ev);

/* Deferring
 * --------------------------------------------------------------- */

/** Defers `task` until `ev` triggers.
 *
 *  Returns true when the task was delayed: the node moves to the event's
 *  channel and the caller must not touch the task again.
 *  The triggerer schedules it. Returns false when the event already fired:
 *  the caller schedules the task itself.
 */
bool tpc_flow_event_delayedUntil(tpc_task* task, tpc_flow_event ev);

/** Defers `task` until iteration `index` of the iteration event `ev` fires.
 *  Same contract as tpc_flow_event_delayedUntil. Aborts when index is
 *  outside [start, stop).
 */
bool tpc_flow_event_delayedUntilIter(tpc_task* task, tpc_flow_event ev, int32_t index);

/** Defers `task` until every event in `deps` (n entries) has triggered.
 *
 *  Builds one TaskNode per dependency and links the nodes through nextDep
 *  in dependency order. The task defers on the first still-untriggered dependency.
 *  An already-triggered dependency frees its node during the walk.
 *
 *  Returns true when the task was delayed on some still-untriggered event
 *  (the caller must not touch the task). Returns false when every
 *  dependency already fired: the caller schedules the task itself.
 */
bool tpc_flow_event_delayedUntilMulti(tpc_task* task, const tpc_flow_event_dep* deps, int32_t n);

/* Triggering
 * --------------------------------------------------------------- */

/** Fires a single event. Schedules every drained dependent task through
 *  enqueue(queue, task), re-deferring tasks that still wait on other events.
 *  An event triggers once, a re-trigger aborts.
 */
void tpc_flow_event_trigger(tpc_flow_event ev, void* queue, tpc_flow_event_enqueue enqueue);

/** Fires iteration `index` of an iteration event with the same contract
 *  as tpc_flow_event_trigger. Aborts when index is outside [start, stop).
 */
void tpc_flow_event_triggerIter(tpc_flow_event ev, int32_t index, void* queue, tpc_flow_event_enqueue enqueue);

#ifdef __cplusplus
}
#endif

#endif /* TPC_FLOW_EVENTS_H */
