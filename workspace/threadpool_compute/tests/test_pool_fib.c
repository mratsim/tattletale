/* threadpool_compute: fib through the macro frontend. */

#include "threadpool_compute.h"

#include <stdio.h>
#include <stdlib.h>

static int64_t fib_expected(int64_t n) {
    int64_t x = 0, y = 1;
    for (int64_t i = 0; i < n; i++) {
        int64_t t = x + y;
        x = y;
        y = t;
    }
    return x;
}

static int64_t fib(int32_t n);

TPC_DEFINE_SPAWN_RET(int64_t, fib, (int32_t))

static int64_t fib(int32_t n) {
    if (n < 2) {
        return n;
    }
    tpc_flowvar a = TPC_SPAWN(fib, (n - 1));
    int64_t b = fib(n - 2);
    int64_t ra = TPC_SYNC(a, int64_t);
    return ra + b;
}

int main(void) {
    int64_t expected[31];
    for (int32_t i = 0; i <= 30; i++) {
        expected[i] = fib_expected(i);
    }

    int32_t counts[] = {2, 8, tpc_topology_getNumThreadsOS()};
    for (unsigned ci = 0; ci < 3; ci++) {
        tpc_threadpool* tp = TPC_THREADPOOL_INIT(counts[ci]);
        tpc_flowvar fv = TPC_SPAWN(fib, (20));
        int64_t result = TPC_SYNC(fv, int64_t);
        if (result != expected[20]) {
            fprintf(stderr, "fib(20) @%d workers: %lld != %lld\n",
                    tpc_threadpool_current()->numThreads,
                    (long long)result, (long long)expected[20]);
            return 1;
        }
        int32_t ran = tpc_threadpool_current()->numThreads;
        TPC_THREADPOOL_SHUTDOWN(tp);
        printf("fib(20) = %lld @%d workers OK\n", (long long)result, ran);
    }
    printf("test_pool_fib: OK\n");
    return 0;
}
