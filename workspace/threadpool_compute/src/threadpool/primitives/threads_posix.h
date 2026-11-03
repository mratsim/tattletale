/* Worker spawn and join, posix backend.
 *
 * pthread_create expects a start routine returning void*.
 * tpc_thread_create casts the pool's int-returning worker bodies to that shape.
 * pthread_join stores the void* result in a local the pool never reads.
 */
#ifndef TPC_THREADS_POSIX_H
#define TPC_THREADS_POSIX_H

#include <pthread.h>

typedef pthread_t tpc_thread_t;

/** Spawn a worker thread.
 *
 *  Returns: 0 on success, nonzero on failure.
 *
 *  Parameters:
 *  Out:     thread:  the spawned thread handle.
 *  In:      start:   the worker body, an int return.
 *  In:      arg:     the worker body argument.
 */
static inline int tpc_thread_create(tpc_thread_t* thread, int (*start)(void*), void* arg) {
    return pthread_create(thread, NULL, (void* (*)(void*))start, arg) == 0 ? 0 : 1;
}

/** Block until the thread exits. The join value is discarded. */
static inline void tpc_thread_join(tpc_thread_t thread) {
    void* value;
    pthread_join(thread, &value);
}

#endif /* TPC_THREADS_POSIX_H */
