/** C11 atomics in C and C++ translation units.
 *  Nim force-includes the public header into every generated unit.
 *  Under `nim cpp` those units are C++.
 *  `_Atomic` does not exist in C++, `<stdatomic.h>` ships only from C++23.
 */
#ifndef TPC_ATOMICS_COMPAT_H
#define TPC_ATOMICS_COMPAT_H

#ifdef __cplusplus
    #include <atomic>
    #define TPC_ATOMIC(T) std::atomic<T>
    #define TPC_ALIGNAS(N) alignas(N)
    /**
     *  Unqualified `atomic_*_explicit` calls resolve through ADL.
     *  Enumerators and `atomic_thread_fence` take no atomic argument.
     *  A using-declaration imports each.
     */
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
