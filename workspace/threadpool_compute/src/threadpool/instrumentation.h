/* threadpool_compute: instrumentation.
 *
 * A simple design-by-contract API
 * --------------------------------------------------------
 * Everything should be a template that doesn't produce any code
 * when CTT_DEBUG is not defined.
 * Those checks are controlled by a custom flag instead of
 * "--boundsChecks" or "--nilChecks" to decouple them from user code checks.
 * Furthermore, we want them to be very lightweight on performance
 *
 * Debug logging and design-by-contract checks, compiled out unless
 * requested:
 *   -DTPC_THREADPOOL_DEBUG            all debug tracing
 *   -DTPC_THREADPOOL_DEBUG_SPLIT      loop-splitting traces only
 *   -DTPC_THREADPOOL_DEBUG_TERMINATION
 *   -DTPC_THREADPOOL_ASSERTS          pre/post/transient contracts,
 *                                     independent of NDEBUG
 * Contracts print the violated predicate with the worker id.
 *
 * Loggers
 * --------------------------------------------------------
 */
#ifndef TPC_INSTRUMENTATION_H
#define TPC_INSTRUMENTATION_H

#include <stdio.h>

#ifdef TPC_THREADPOOL_DEBUG
    #define tpc_debug_log(...)              \
        do {                                \
            fprintf(stderr, __VA_ARGS__);   \
            fflush(stderr);                 \
        } while (0)
#else
    #define tpc_debug_log(...) ((void)0)
#endif

#ifdef TPC_THREADPOOL_DEBUG_SPLIT
    #define tpc_debug_split(...) tpc_debug_log(__VA_ARGS__)
#else
    #define tpc_debug_split(...) ((void)0)
#endif

#ifdef TPC_THREADPOOL_DEBUG_TERMINATION
    #define tpc_debug_termination(...) tpc_debug_log(__VA_ARGS__)
#else
    #define tpc_debug_termination(...) ((void)0)
#endif

#if defined(TPC_THREADPOOL_ASSERTS) || !defined(NDEBUG)
    #define tpc_preCondition(expr)                                        \
        do {                                                              \
            if (!(expr)) {                                                \
                fprintf(stderr,                                           \
                        "pre-condition violated: %s (%s:%d)\n",           \
                        #expr, __FILE__, __LINE__);                       \
                abort();                                                  \
            }                                                             \
        } while (0)
    #define tpc_postCondition(expr) tpc_preCondition(expr)
    #define tpc_ascertain(expr)     tpc_preCondition(expr)
#else
    #define tpc_preCondition(expr)   ((void)0)
    #define tpc_postCondition(expr)  ((void)0)
    #define tpc_ascertain(expr)      ((void)0)
#endif

#endif /* TPC_INSTRUMENTATION_H */
