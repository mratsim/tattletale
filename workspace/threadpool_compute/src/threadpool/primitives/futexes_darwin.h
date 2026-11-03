/* Futex parking, Darwin backend.
 *
 * Darwin futexes. They are used in libc++ so likely to be very stable.
 * A new API appeared in OSX Big Sur (Jan 2021) ulock_wait2 and macOS pthread_cond_t has been migrated to it
 * - https://github.com/apple/darwin-xnu/commit/d4061fb0260b3ed486147341b72468f836ed6c8f#diff-08f993cc40af475663274687b7c326cc6c3031e0db3ac8de7b24624610616be6
 *
 * The old API is ulock_wait
 * - https://opensource.apple.com/source/xnu/xnu-7195.81.3/bsd/kern/sys_ulock.c.auto.html
 * - https://opensource.apple.com/source/xnu/xnu-7195.81.3/bsd/sys/ulock.h.auto.html
 */
#ifndef TPC_FUTEXES_DARWIN_H
#define TPC_FUTEXES_DARWIN_H

#include <errno.h>
#include <stdint.h>

#define UL_COMPARE_AND_WAIT 1u
#define ULF_NO_ERRNO        0x01000000u
#define ULF_WAKE_ALL        0x00000100u
/* generic flags live in bits [31, 24], the __ulock_wake flags in bits [15, 8]
 *
 * # obsolete names: UL_OSSPINLOCK = UL_COMPARE_AND_WAIT, UL_HANDOFFLOCK = UL_UNFAIR_LOCK
 */

extern int __ulock_wait(uint32_t operation, void* addr, uint64_t value, uint32_t timeout_us);
extern int __ulock_wake(uint32_t operation, void* addr, uint64_t wake_value);

/* Futex API
 * ------------------------------------------------------------------------
 */

static inline int tpc_futex_wait(TPC_ATOMIC(uint32_t)* addr, uint32_t expected) {
    int rc = __ulock_wait(UL_COMPARE_AND_WAIT | ULF_NO_ERRNO, addr, expected, 0);
    if (rc >= 0) {
        return 0;
    }
    return -rc;   /* EAGAIN: the word moved, EINTR: retryable */
}

static inline int tpc_futex_wake(TPC_ATOMIC(uint32_t)* addr, uint32_t nwakes) {
    /* one waiter per call */
    int woken = 0;
    while ((uint32_t)woken < nwakes) {
        int rc = __ulock_wake(UL_COMPARE_AND_WAIT | ULF_NO_ERRNO, addr, 0);
        if (rc == -ENOENT) {
            break;   /* no waiter on the word */
        }
        if (rc < 0) {
            return rc;
        }
        woken++;
    }
    return 0;
}

static inline int tpc_futex_wake_all(TPC_ATOMIC(uint32_t)* addr) {
    int rc = __ulock_wake(UL_COMPARE_AND_WAIT | ULF_NO_ERRNO | ULF_WAKE_ALL, addr, 0);
    return rc < 0 && rc != -ENOENT ? rc : 0;
}

#endif /* TPC_FUTEXES_DARWIN_H */
