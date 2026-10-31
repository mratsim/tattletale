/* Futex parking, Windows backend.
 *
 * Contrary to the documentation, the futex related primitives are NOT in kernel32.dll
 * but in API-MS-Win-Core-Synch-l1-2-0.dll ¯\_(ツ)_/¯
 * Inside the call the kernel re-checks the word against the compare value,
 * a wake racing a not-yet-sleeping waiter is never lost
 */
#ifndef TPC_FUTEXES_WINDOWS_H
#define TPC_FUTEXES_WINDOWS_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static inline int tpc_futex_wait(_Atomic uint32_t* addr, uint32_t expected) {
    uint32_t observed = expected;
    /* The Address should be volatile */
    WaitOnAddress((volatile VOID *)addr, &observed, sizeof(uint32_t), INFINITE);
    return 0;
}

static inline int tpc_futex_wake(_Atomic uint32_t* addr, uint32_t nwakes) {
    for (uint32_t i = 0; i < nwakes; i++) {
        WakeByAddressSingle((PVOID)addr);
    }
    return 0;
}

static inline int tpc_futex_wake_all(_Atomic uint32_t* addr) {
    WakeByAddressAll((PVOID)addr);
    return 0;
}

#endif /* TPC_FUTEXES_WINDOWS_H */
