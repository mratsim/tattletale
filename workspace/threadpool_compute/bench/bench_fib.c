/* threadpool_compute: Fibonacci through the spawn path.
 *
 * Pool startup happens before timing. One parfib(n) call runs inside
 * a timed region, its left branch spawns and its right branch runs inline.
 */
#include "threadpool_compute.h"
#include "wtime.h"

#include <stdio.h>
#include <stdlib.h>

static int64_t fib(int32_t n);

TPC_DEFINE_SPAWN_RET(int64_t, fib, (int32_t))

static int64_t fib(int32_t n) {
    if (n < 2) {
        return n;
    }
    tpc_flowvar x = TPC_SPAWN(fib, (n - 1));
    int64_t y = fib(n - 2);
    int64_t rx = TPC_SYNC(x, int64_t);
    return rx + y;
}

int main(int argc, char** argv) {
    int32_t n = argc > 1 ? atoi(argv[1]) : 40;
    int32_t nthreads = tpc_topology_getNumThreadsOS();
    const char* env = getenv("CTT_NUM_THREADS");
    if (env != NULL) {
        nthreads = atoi(env);
    }

    tpc_threadpool* tp = TPC_THREADPOOL_INIT(nthreads);

    double start = tpc_wtime_msec();
    tpc_flowvar f = TPC_SPAWN(fib, (n));
    int64_t result = TPC_SYNC(f, int64_t);
    double stop = tpc_wtime_msec();

    TPC_THREADPOOL_SHUTDOWN(tp);

    printf("--------------------------------------------------------------------------\n");
    printf("Scheduler:                                    threadpool_compute\n");
    printf("Benchmark:                                    Fibonacci\n");
    printf("Threads:                                      %d\n", nthreads);
    printf("Time(ms)                                      %.3f\n", stop - start);
    printf("--------------------------------------------------------------------------\n");
    printf("n requested:                                  %d\n", n);
    printf("result:                                       %lld\n", (long long)result);
    return 0;
}
