/* Worker spawn and join, Windows backend.
 *
 * Status: untested on this Mac. ABI verified by inspection against Nim threadtypes.nim and the MS x86-64/x86 calling convention docs.
 * futexes_windows.h has the same standing.
 */
#ifndef TPC_THREADS_WINDOWS_H
#define TPC_THREADS_WINDOWS_H

#define WIN32_LEAN_AND_MEAN
#include <stdlib.h>
#include <windows.h>

typedef HANDLE tpc_thread_t;

/* ABI contract for the Windows worker entry (LPTHREAD_START_ROUTINE, stdcall).
 * On 32-bit x86 a stdcall callee pops its own arguments.
 * A cdecl worker body behind that entry leaves the stack unbalanced on worker return.
 * tpc_thread_wrapper below is stdcall itself and calls each body through its declared int (*)(void*) type.
 *
 * CreateThread passes a single pointer to the entry. tpc_thread_create packs the body pointer and its argument into one heap record.
 * Each wrapper invocation frees the record before the body runs. One record per spawned worker, never per task.
 */
typedef struct tpc_thread_start {
    int (*fn)(void*);
    void* arg;
} tpc_thread_start;

static inline DWORD WINAPI tpc_thread_wrapper(void* param) {
    tpc_thread_start* packed = (tpc_thread_start*)param;
    int (*fn)(void*) = packed->fn;
    void* arg = packed->arg;
    free(packed);
    return (DWORD)fn(arg);
}

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
    tpc_thread_start* packed = (tpc_thread_start*)malloc(sizeof(*packed));
    if (packed == NULL) {
        return 1;
    }
    packed->fn = start;
    packed->arg = arg;
    *thread = CreateThread(NULL, 0, tpc_thread_wrapper, packed, 0, NULL);
    if (*thread == NULL) {
        /* CreateThread failed, no thread runs the wrapper, free releases the record */
        free(packed);
        return 1;
    }
    return 0;
}

/** Block until the thread exits. The join value is discarded. */
static inline void tpc_thread_join(tpc_thread_t thread) {
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
}

#endif /* TPC_THREADS_WINDOWS_H */
