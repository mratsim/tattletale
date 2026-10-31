/* threadpool_compute: N-queens through the spawn path.
 *
 * Barcelona OpenMP Tasks Suite shape (Cilk original by Randall):
 * one spawned flowvar per valid column placement, summed at the end.
 * Pool startup happens before timing. One nqueens call runs inside
 * a timed region and the result is verified.
 */
#include "threadpool_compute.h"
#include "wtime.h"

#include <alloca.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** a stores the queen positions, true when no queen conflicts. */
static bool isValid(int32_t n, const unsigned char* a) {
    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = i + 1; j < n; j++) {
            int32_t p = a[i], q = a[j];
            if (q == p || q == p - (j - i) || q == p + (j - i)) {
                return false;
            }
        }
    }
    return true;
}

static int32_t nqueens(int32_t n, int32_t j, const unsigned char* a);

TPC_DEFINE_SPAWN_RET(int32_t, nqueens, (int32_t, int32_t, const unsigned char *))

static int32_t nqueens(int32_t n, int32_t j, const unsigned char* a) {
    if (n == j) {
        return 1;
    }

    tpc_flowvar* localCounts = alloca((size_t)n * sizeof(tpc_flowvar));
    memset(localCounts, 0, (size_t)n * sizeof(tpc_flowvar));

    for (int32_t i = 0; i < n; i++) {
        unsigned char* b = alloca((size_t)j + 1);
        memcpy(b, a, (size_t)j);
        b[j] = (unsigned char)i;
        if (isValid(j + 1, b)) {
            localCounts[i] = TPC_SPAWN(nqueens, (n, j + 1, b));
        }
    }

    int32_t count = 0;
    for (int32_t i = 0; i < n; i++) {
        if (localCounts[i].task != NULL) {
            count += TPC_SYNC(localCounts[i], int32_t);
        }
    }
    return count;
}

static const int32_t solutions[] = {
    1, 0, 0, 2, 10, 4, 10, 92, 352, 724,
    2680, 14200, 73712, 365596, 2279184, 14772512,
};

int main(int argc, char** argv) {
    int32_t n = argc > 1 ? atoi(argv[1]) : 11;
    int32_t nthreads = tpc_topology_getNumThreadsOS();
    const char* env = getenv("CTT_NUM_THREADS");
    if (env != NULL) {
        nthreads = atoi(env);
    }
    if (n < 1 || n > (int32_t)(sizeof(solutions) / sizeof(solutions[0]))) {
        fprintf(stderr, "n must be in [1, %zu]\n", sizeof(solutions) / sizeof(solutions[0]));
        return 1;
    }

    tpc_threadpool* tp = TPC_THREADPOOL_INIT(nthreads);

    unsigned char* a = alloca((size_t)n);
    double start = tpc_wtime_msec();
    int32_t count = nqueens(n, 0, a);
    double stop = tpc_wtime_msec();

    TPC_THREADPOOL_SHUTDOWN(tp);

    if (count != solutions[n - 1]) {
        fprintf(stderr, "N-Queens failure: %d != %d\n", count, solutions[n - 1]);
        return 1;
    }

    printf("--------------------------------------------------------------------------\n");
    printf("Scheduler:                                    threadpool_compute\n");
    printf("Benchmark:                                    N-queens\n");
    printf("Threads:                                      %d\n", nthreads);
    printf("Time(ms)                                      %.3f\n", stop - start);
    printf("--------------------------------------------------------------------------\n");
    printf("Problem size:                                 %dx%d board with %d queens\n", n, n, n);
    printf("Solutions found:                              %d\n", count);
    return 0;
}
