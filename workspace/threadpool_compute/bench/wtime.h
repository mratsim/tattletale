/* threadpool_compute: wall-clock timing for the benchmarks. */
#ifndef TPC_WTIME_H
#define TPC_WTIME_H

#include <time.h>

/** Milliseconds on the monotonic clock. */
static inline double tpc_wtime_msec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

#endif /* TPC_WTIME_H */
