/* C11 atomics in C and C++ translation units: Nim force-includes the public
 * header into every generated unit, and under `nim cpp` those units are C++,
 * where _Atomic does not exist and <stdatomic.h> ships only from C++23.
 */
#ifndef TPC_ATOMICS_COMPAT_H
#define TPC_ATOMICS_COMPAT_H

#ifdef __cplusplus
    #include <atomic>
    #define TPC_ATOMIC(T) std::atomic<T>
    #define TPC_ALIGNAS(N) alignas(N)
    /* unqualified atomic_*_explicit calls resolve to the std free functions
     * through ADL; enumerators and atomic_thread_fence take no atomic
     * argument, so a using-declaration imports each */
    using std::memory_order_relaxed;
    using std::memory_order_consume;
    using std::memory_order_acquire;
    using std::memory_order_release;
    using std::memory_order_acq_rel;
    using std::memory_order_seq_cst;
    using std::atomic_thread_fence;
#else
    #include <stdatomic.h>
    #define TPC_ATOMIC(T) _Atomic(T)
    #define TPC_ALIGNAS(N) _Alignas(N)
#endif

#endif /* TPC_ATOMICS_COMPAT_H */
