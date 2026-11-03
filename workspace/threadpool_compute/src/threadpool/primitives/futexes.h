/* A wrapper for linux futex.
 * Condition variables do not always wake on signal which can deadlock the runtime
 * so we need to roll up our sleeves and use the low-level futex API.
 */
#ifndef TPC_FUTEXES_H
#define TPC_FUTEXES_H

#include "atomics_compat.h"
#include <stdint.h>

/* OS primitives
 * ------------------------------------------------------------------------
 */

/* Futex API
 * ------------------------------------------------------------------------
 */

/** Suspend a thread if the value of the futex is the same as expected. */

/** Wake one thread (from the same process) */

/** Wake all threads (from the same process) */

/** Contended spin loop primitive: x86 PAUSE, AArch64 YIELD, else a compiler barrier. */
static inline void tpc_cpuRelax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    __atomic_thread_fence(memory_order_seq_cst);
#endif
}

#if defined(_WIN32)
#include "futexes_windows.h"
#elif defined(__APPLE__)
#include "futexes_darwin.h"
#else
#include "futexes_linux.h"
#endif

#endif /* TPC_FUTEXES_H */
