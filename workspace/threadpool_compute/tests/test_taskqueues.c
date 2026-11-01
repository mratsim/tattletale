/* threadpool_compute: Chase-Lev deque invariants, single thread.
 *
 * Owner push/pop runs LIFO over the ring, growth re-copies the live
 * window when the ring fills, peek reads the occupancy.
 * Steal takes the front item FIFO and records the thief.
 * Empty-deque pop and steal regression-guard the signed front/back underflow.
 */
#include "threadpool_compute.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static tpc_task* fake(int32_t i) {
    return (tpc_task*)(uintptr_t)(0x1000 + (uintptr_t)i);
}

static void expect(int cond, const char* what) {
    if (!cond) {
        fprintf(stderr, "taskqueues: %s\n", what);
        exit(1);
    }
}

static void lifoAndGrowth(void) {
    tpc_taskqueue tq;
    tpc_taskqueue_init(&tq, 32);
    const int32_t n = 5000;
    for (int32_t i = 0; i < n; i++) {
        tpc_taskqueue_push(&tq, fake(i));
    }
    expect(tpc_taskqueue_peek(&tq) == n, "size after 5000 pushes");
    for (int32_t i = n - 1; i >= 0; i--) {
        tpc_task* t = tpc_taskqueue_pop(&tq);
        expect(t == fake(i), "pop order is LIFO");
    }
    expect(tpc_taskqueue_pop(&tq) == NULL, "pop on empty is NULL");
    expect(tpc_taskqueue_peek(&tq) == 0, "size after drain");
    tpc_taskqueue_teardown(&tq);
}

static void stealTakesFront(void) {
    tpc_taskqueue tq;
    tpc_taskqueue_init(&tq, 32);
    struct tpc_task real[3];
    for (int32_t i = 0; i < 3; i++) {
        memset(&real[i], 0, sizeof(real[i]));
        tpc_taskqueue_push(&tq, &real[i]);
    }
    tpc_task* s = tpc_taskqueue_steal(7u, &tq);
    expect(s == &real[0], "steal takes the front item");
    expect(tpc_taskqueue_steal(7u, &tq) == &real[1], "steal order is FIFO");
    expect(tpc_taskqueue_peek(&tq) == 1, "size after two steals");
    expect(tpc_taskqueue_pop(&tq) == &real[2], "pop takes the last item");
    expect(tpc_taskqueue_steal(7u, &tq) == NULL, "steal on empty is NULL");
    expect(tpc_taskqueue_pop(&tq) == NULL, "pop on empty is NULL");
    tpc_taskqueue_teardown(&tq);
}

int main(void) {
    lifoAndGrowth();
    stealTakesFront();
    printf("test_taskqueues: OK\n");
    return 0;
}
