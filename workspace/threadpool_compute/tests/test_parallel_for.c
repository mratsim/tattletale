/* threadpool_compute: parallel-for over the macro frontend. */

#include "threadpool_compute.h"

#include <stdio.h>
#include <stdlib.h>

#define N 4000000

TPC_DEFINE_PARFOR(axpy, float a; float* x; float* y;)

TPC_PARFOR_BODY(axpy, i, (a, x, y), { y[i] += a * x[i]; })

TPC_DEFINE_PARFOR(sum, int32_t n; const int32_t* v; int64_t* out;)

TPC_PARFOR_BODY(sum, i, (n, v, out), {
    int64_t s = 0;
    for (int32_t j = i; j < n; j += 1024 * 1024) {
        s += v[j];
    }
    out[i] = s;
})

int main(void) {
    float* x = malloc(N * sizeof(float));
    float* y = malloc(N * sizeof(float));
    int32_t* v = malloc(N * sizeof(int32_t));
    if (x == NULL || y == NULL || v == NULL) {
        return 1;
    }
    for (int32_t i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = (float)i;
        v[i] = 1;
    }

    int32_t counts[] = {2, 8, tpc_topology_getNumThreadsOS()};
    for (unsigned ci = 0; ci < 3; ci++) {
        tpc_threadpool* tp = TPC_THREADPOOL_INIT(counts[ci]);
        for (int32_t i = 0; i < N; i++) {
            y[i] = (float)i;
            v[i] = 1;
        }

        TPC_SYNC_SCOPE({
            TPC_PARFOR(axpy, 0, N, (2.0f, x, y));
        });

        int64_t* parts = malloc(1024 * 1024 * sizeof(int64_t));
        TPC_SYNC_SCOPE({
            TPC_PARFOR(sum, 0, 1024 * 1024, (N, v, parts));
        });
        int64_t total = 0;
        for (int32_t i = 0; i < 1024 * 1024; i++) {
            total += parts[i];
        }
        free(parts);

        int errors = 0;
        for (int32_t i = 0; i < N; i++) {
            float want = (float)i + 2.0f;
            if (y[i] != want) {
                if (errors < 3) {
                    fprintf(stderr, "y[%d] = %f != %f\n", i, y[i], want);
                }
                errors++;
            }
        }
        if (total != N) {
            fprintf(stderr, "sum = %lld != %d\n", (long long)total, N);
            errors++;
        }
        int32_t ran = tpc_threadpool_current()->numThreads;
        TPC_THREADPOOL_SHUTDOWN(tp);
        if (errors != 0) {
            fprintf(stderr, "test_parallel_for: %d errors @%d workers\n",
                    errors, ran);
            return 1;
        }
        printf("parfor OK @%d workers\n", ran);
    }
    printf("test_parallel_for: OK\n");
    return 0;
}
