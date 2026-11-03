/* Worker spawn and join.
 *
 * tpc_thread_create spawns a worker thread and returns 0 on success.
 * tpc_thread_join blocks until that thread exits and discards the join value.
 * Start routines return an int.
 */
#ifndef TPC_THREADS_H
#define TPC_THREADS_H

#if defined(_WIN32)
#include "threads_windows.h"
#else
#include "threads_posix.h"
#endif

#endif /* TPC_THREADS_H */
