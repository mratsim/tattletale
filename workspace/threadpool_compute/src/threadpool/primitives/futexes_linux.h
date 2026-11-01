/* A wrapper for linux futex.
 */
#ifndef TPC_FUTEXES_LINUX_H
#define TPC_FUTEXES_LINUX_H

#include <errno.h>
#include <limits.h>
#include <linux/futex.h>
#include <sys/syscall.h>

/* glibc declares syscall only under _GNU_SOURCE, defining the macro
 * in a header is too late as system headers may have expanded already
 */
long syscall(long number, ...);

/* OS primitives
 * ------------------------------------------------------------------------
 */

/* Futex API
 * ------------------------------------------------------------------------
 */

/** See https://web.archive.org/web/20230208151430/http://locklessinc.com/articles/futex_cheat_sheet/
 *  and https://www.akkadia.org/drepper/futex.pdf
 */


static inline int tpc_futex_wait(_Atomic uint32_t* addr, uint32_t expected) {
    /* Returns 0 in case of a successful suspend
     * If value are different, it returns EWOULDBLOCK
     * We discard as this is not needed and simplifies compat with Windows futex
     */
    long rc = syscall(SYS_futex, (void*)addr, FUTEX_WAIT_PRIVATE, (uint32_t)expected, NULL, NULL, 0);
    return rc == 0 ? 0 : -(int)errno;
}

static inline int tpc_futex_wake(_Atomic uint32_t* addr, uint32_t nwakes) {
    /* Returns the number of actually woken threads
     * or a Posix error code (if negative)
     * We discard as this is not needed and simplifies compat with Windows futex
     */
    long rc = syscall(SYS_futex, (void*)addr, FUTEX_WAKE_PRIVATE, (int)nwakes, NULL, NULL, 0);
    return rc == 0 ? 0 : -(int)errno;
}

static inline int tpc_futex_wake_all(_Atomic uint32_t* addr) {
    /* Returns the number of actually woken threads
     * or a Posix error code (if negative)
     */
    long rc = syscall(SYS_futex, (void*)addr, FUTEX_WAKE_PRIVATE, INT32_MAX, NULL, NULL, 0);
    return rc == 0 ? 0 : -(int)errno;
}

#endif /* TPC_FUTEXES_LINUX_H */
