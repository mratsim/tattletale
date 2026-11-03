/* threadpool_compute: scoped barriers.
 *
 * Scoped barrier
 * ----------------------------------------------------------------------------------
 *
 * A scoped barrier allows detecting if there are child, grandchild or descendant tasks
 * still running so that a thread can wait for nested spawns.
 *
 * This solves the following problem:
 *
 * parallelFor i in 0 ..< M:
 *   awaitable: loopI
 *   parallelFor j in 0 ..< M:
 *     awaitable: loopJ
 *     discard
 *
 * Here, the loop i will add all the parallel tasks for loop j to the task queue
 * and continue on. Even if we await loopI, the task was really to just create the loopJ,
 * but the loop itself might still be pending in the task queue
 *
 * We could instead await loopJ before going to the next i iteration but that
 * restrict the parallelism exposed.
 *
 * Alternatively we could use `syncRoot` after those statements but that means
 * that it cannot be called from parallel code.
 *
 * In short:
 * - the scoped barrier should prevent a thread from continuing while any descendant task
 *   is still running.
 * - it should be composable and parallel constructs need not to worry about its presence
 *   in spawn functions.
 * - the enclosed scope should be able to expose all parallelism opportunities
 *   in particular nested parallel for regions.
 *
 * TODO: transform into a Futex/eventcount and allow sleeping
 */
#ifndef TPC_SCOPED_BARRIERS_H
#define TPC_SCOPED_BARRIERS_H

#include "../primitives/atomics_compat.h"
#include <stdbool.h>
#include <stdint.h>

/** A scoped barrier allows detecting if there are child, grandchild or descendant tasks
 *  still running so that a thread can wait for nested spawns.
 *
 *  Scoped barriers can be nested and work like a stack.
 *  Only one can be active for a given thread for a given code section.
 *
 *  They can be allocated on the stack given that a scoped barrier can not be exited
 *  before all the descendant tasks exited and so the descendants cannot escape,
 *  i.e. they have a pointer to their scope which is always valid.
 *
 *  This means that in case of nested scopes, only the inner scope needs to track its descendants.
 */
typedef struct tpc_scoped_barrier {
    TPC_ATOMIC(int32_t) descendants;
} tpc_scoped_barrier;

/* Note: If one is defined, the other destructors proc are not implictly create inline
 *       even if trivial
 *
 * A scoped barrier cannot be copied.
 */

/** Initialize a scoped barrier */
static inline void tpc_scoped_barrier_initialize(tpc_scoped_barrier* sb) {
    atomic_store_explicit(&sb->descendants, 0, memory_order_relaxed);
}

/** Register a descendant task to the scoped barrier
 *  Important: the thread creating the task must register the descendant
 *  before handing them over to the runtime.
 *  This way, if the end of scope is reached and we have 0 descendant it means that
 *  - either no task was created in the scope
 *  - tasks were created, but descendants increment/decrement cannot reach 0 before all descendants actually exited
 */
static inline void tpc_scoped_barrier_registerDescendant(tpc_scoped_barrier* sb) {
    if (sb == NULL) {
        return;
    }
    atomic_fetch_add_explicit(&sb->descendants, 1, memory_order_release);
}

/** Unlist a descendant task from the scoped barrier.
 *  Important: if that task spawned new tasks, it is fine even if those grandchild tasks
 *  are still running, however they must have been registered to the scoped barrier to avoid race conditions.
 */
static inline void tpc_scoped_barrier_unlistDescendant(tpc_scoped_barrier* sb) {
    if (sb == NULL) {
        return;
    }
    atomic_thread_fence(memory_order_release);
    atomic_fetch_sub_explicit(&sb->descendants, 1, memory_order_release);
}

/** Returns true if a scoped barrier has at least a descendant task.
 *  This should only be called from the thread that created the scoped barrier.
 */
static inline bool tpc_scoped_barrier_hasDescendantTasks(const tpc_scoped_barrier* sb) {
    return atomic_load_explicit(&sb->descendants, memory_order_acquire) != 0;
}

#endif /* TPC_SCOPED_BARRIERS_H */
