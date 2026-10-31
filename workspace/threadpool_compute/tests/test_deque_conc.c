/* threadpool_compute: concurrent deque traffic, one owner plus thieves.
 *
 * One owner thread pushes the whole task window and drains with pops,
 * thieves steal concurrently, every task must leave the deque
 * exactly once across all consumers.
 */
#include "threadpool_compute.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NTASKS 200000
#define NTHIEVES 4

static struct tpc_task tasks[NTASKS];
static _Atomic int32_t hits[NTASKS];
static _Atomic int32_t taken;
static _Atomic int32_t done;

static void consume(tpc_task* t) {
    int32_t idx = (int32_t)(t - tasks);
    atomic_fetch_add_explicit(&hits[idx], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&taken, 1, memory_order_relaxed);
}

static tpc_taskqueue sharedQueue;

static void* thiefMain(void* arg) {
    (void)arg;
    for (;;) {
        tpc_task* t = tpc_taskqueue_steal(9u, &sharedQueue);
        if (t != NULL) {
            consume(t);
            continue;
        }
        if (atomic_load_explicit(&done, memory_order_acquire)) {
            break;
        }
    }
    return NULL;
}

int main(void) {
    tpc_taskqueue_init(&sharedQueue, 64);
    pthread_t thieves[NTHIEVES];
    for (int32_t i = 0; i < NTHIEVES; i++) {
        pthread_create(&thieves[i], NULL, thiefMain, NULL);
    }
    for (int32_t i = 0; i < NTASKS; i++) {
        memset(&tasks[i], 0, sizeof(tasks[i]));
        tpc_taskqueue_push(&sharedQueue, &tasks[i]);
    }
    tpc_task* t;
    while ((t = tpc_taskqueue_pop(&sharedQueue)) != NULL) {
        consume(t);
    }
    atomic_store_explicit(&done, 1, memory_order_release);
    for (int32_t i = 0; i < NTHIEVES; i++) {
        pthread_join(thieves[i], NULL);
    }
    if (atomic_load_explicit(&taken, memory_order_relaxed) != NTASKS) {
        fprintf(stderr, "deque: took %d of %d\n",
                atomic_load_explicit(&taken, memory_order_relaxed), NTASKS);
        return 1;
    }
    for (int32_t i = 0; i < NTASKS; i++) {
        if (atomic_load_explicit(&hits[i], memory_order_relaxed) != 1) {
            fprintf(stderr, "deque: task %d taken %d times\n", i,
                    atomic_load_explicit(&hits[i], memory_order_relaxed));
            return 1;
        }
    }
    tpc_taskqueue_teardown(&sharedQueue);
    printf("test_deque_conc: OK\n");
    return 0;
}
