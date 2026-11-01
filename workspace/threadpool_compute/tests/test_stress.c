/* threadpool_compute: pool stress through the public macro API.
 *
 * Spawn storms push the per-worker rings past their capacity.
 * Parfor chunks spawn leaf tasks mid-loop. Rounds repeat to shake races.
 * Checks count leaves, every task must run exactly once.
 */
#include "threadpool_compute.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

static void leaf(_Atomic int32_t* count);

TPC_DEFINE_SPAWN(leaf, (_Atomic int32_t*))
static void leaf(_Atomic int32_t* count) {
    atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
}

static void stormRound(int32_t leaves) {
    _Atomic int32_t count = 0;
    TPC_SYNC_SCOPE({
        for (int32_t i = 0; i < leaves; i++) {
            TPC_SPAWN(leaf, (&count));
        }
    });
    if (atomic_load_explicit(&count, memory_order_relaxed) != leaves) {
        fprintf(stderr, "storm: %d leaves, %d ran\n", leaves,
                atomic_load_explicit(&count, memory_order_relaxed));
        exit(1);
    }
}

TPC_DEFINE_PARFOR(mix, _Atomic int32_t* count; int32_t leaves;)
TPC_PARFOR_BODY(mix, i, (count, leaves), {
    (void)i;
    for (int32_t j = 0; j < leaves; j++) {
        TPC_SPAWN(leaf, (count));
    }
})

static void mixRound(int32_t iters, int32_t leaves) {
    _Atomic int32_t count = 0;
    TPC_SYNC_SCOPE({
        TPC_PARFOR(mix, 0, iters, (&count, leaves));
    });
    if (atomic_load_explicit(&count, memory_order_relaxed) !=
        iters * leaves) {
        fprintf(stderr, "mix: want %d, got %d\n", iters * leaves,
                atomic_load_explicit(&count, memory_order_relaxed));
        exit(1);
    }
}

int main(void) {
    int32_t workers[] = {2, 8, 20};
    for (int32_t w = 0; w < 3; w++) {
        tpc_threadpool* tp = TPC_THREADPOOL_INIT(workers[w]);
        for (int32_t r = 0; r < 60; r++) {
            stormRound(511 + (r % 3));
            mixRound(64, 8);
        }
        TPC_THREADPOOL_SHUTDOWN(tp);
        printf("stress @%d workers OK\n", workers[w]);
    }
    printf("test_stress: OK\n");
    return 0;
}
