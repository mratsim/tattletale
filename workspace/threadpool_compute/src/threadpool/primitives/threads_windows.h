/* Worker spawn and join, Windows backend.
 *
 * Status: untested, no Windows machine runs in the development loop.
 * futexes_windows.h has the same standing.
 */
#ifndef TPC_THREADS_WINDOWS_H
#define TPC_THREADS_WINDOWS_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef HANDLE tpc_thread_t;

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
    *thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)start, arg, 0, NULL);
    return *thread != NULL ? 0 : 1;
}

/** Block until the thread exits. The join value is discarded. */
static inline void tpc_thread_join(tpc_thread_t thread) {
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}

#endif /* TPC_THREADS_WINDOWS_H */
