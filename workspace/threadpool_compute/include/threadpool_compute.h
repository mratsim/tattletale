/* threadpool_compute: the public API and the C macro frontend over
 * work-stealing scheduling: Chase-Lev deques, futex parking and lazy
 * binary splitting for parallel-for.
 *
 * Parallelism paradigms:
 * - Task parallelism
 * - Data parallelism / parallel for
 *   - parallel-for with thread-local prologue and epilogue
 *   - parallel-reduction without atomics or locks (not ported)
 * - Dataflow parallelism
 *   - also known as:
 *     - Graph parallelism
 *     - Stream parallelism
 *     - Pipeline parallelism
 *     - Data-driven (task) parallelism
 *     with precise input/output dependencies (the flowvar API)
 *
 * Task parallel API (spawn and sync, modeled on the constantine threadpool):
 *   TPC_DEFINE_SPAWN(name, (ty0, ty1, ...))       void fn wrapper
 *   TPC_DEFINE_SPAWN_AWAITABLE(name, (ty0, ...))  void fn, syncable
 *   TPC_DEFINE_SPAWN_RET(rty, name, (ty0, ...))   result fn wrapper
 *   TPC_SPAWN(fn, (v0, v1, ...))                  spawn, the one site
 *   TPC_SPAWN_INTO(fv, fn, (vals))                spawn, handle into fv
 *   TPC_SYNC(fv, rty)                             sync, yields the result
 *
 * Data parallel API:
 *   TPC_DEFINE_PARFOR(name, ty n; ty x; ...)      declare
 *   TPC_PARFOR_BODY(name, i, (n, x, ...), body)   per-iteration body
 *   TPC_PARFOR(name, start, stop, (vals))          launch [start, stop)
 *
 * Scoping and runtime:
 *   TPC_SYNC_SCOPE({ body })     wait for the scope's descendants
 *   TPC_THREADPOOL_INIT(n)       create the pool (worker 0 = this thread)
 *   TPC_THREADPOOL_SHUTDOWN(tp)  syncAll + signal + join
 *
 * Macro contract: DEFINE_ macros take the parameter TYPES only.
 * Each DEFINE generates a static per-task spawn function (name_spawn)
 * with real typed parameters. TPC_SPAWN(fn, vals) is the one spawn
 * site for every task shape. Body fields are named synthetically:
 * tpc_t0, tpc_t1, tpc_t2.
 *
 * Requires GNU-compatible compilers (typeof, statement expressions): GCC and Clang.
 */
#ifndef THREADPOOL_COMPUTE_H
#define THREADPOOL_COMPUTE_H

#include <stddef.h>
#include <stdint.h>
#include "../src/threadpool/primitives/atomics_compat.h"

/* C linkage for the pool symbols: the .c sources compile as C even under
 * `nim cpp`, while C++ translation units calling them must not mangle the names.
 */
#ifdef __cplusplus
extern "C" {
#endif

#include "../src/threadpool/crossthread/backoff.h"
#include "../src/threadpool/crossthread/scoped_barriers.h"
#include "../src/threadpool/crossthread/taskqueues.h"
#include "../src/threadpool/crossthread/tasks_flowvars.h"
#include "../src/threadpool/parallel_offloading.h"
#include "../src/threadpool/primitives/topology.h"
#include "../src/threadpool/threadpool.h"

/* Macro utilities
 * --------------------------------------------------------------- */

#define TPC_VA_NARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define TPC_VA_NARGS(...) \
    TPC_VA_NARGS_IMPL(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

#define TPC_CONCAT_IMPL(x, y) x##y
#define TPC_CONCAT(x, y) TPC_CONCAT_IMPL(x, y)
#define TPC_ARGS(...) __VA_ARGS__

#define TPC_UNPACK_1(x) __typeof__((x)->tpc_t0) tpc_t0 = (x)->tpc_t0
#define TPC_UNPACK_2(x) \
    TPC_UNPACK_1(x);    \
    __typeof__((x)->tpc_t1) tpc_t1 = (x)->tpc_t1
#define TPC_UNPACK_3(x) \
    TPC_UNPACK_2(x);    \
    __typeof__((x)->tpc_t2) tpc_t2 = (x)->tpc_t2
#define TPC_UNPACK_4(x) \
    TPC_UNPACK_3(x);    \
    __typeof__((x)->tpc_t3) tpc_t3 = (x)->tpc_t3
#define TPC_UNPACK_5(x) \
    TPC_UNPACK_4(x);    \
    __typeof__((x)->tpc_t4) tpc_t4 = (x)->tpc_t4
#define TPC_UNPACK_6(x) \
    TPC_UNPACK_5(x);    \
    __typeof__((x)->tpc_t5) tpc_t5 = (x)->tpc_t5
#define TPC_UNPACK_7(x) \
    TPC_UNPACK_6(x);    \
    __typeof__((x)->tpc_t6) tpc_t6 = (x)->tpc_t6
#define TPC_UNPACK_8(x) \
    TPC_UNPACK_7(x);    \
    __typeof__((x)->tpc_t7) tpc_t7 = (x)->tpc_t7
#define TPC_UNPACK_N(x, n) TPC_CONCAT(TPC_UNPACK_, n)(x)

/* Runtime
 * ---------------------------------------------------------------- */

tpc_threadpool* tpc_threadpool_current(void);
void tpc_threadpool_shutdown(tpc_threadpool* tp);

#define TPC_THREADPOOL_INIT(n) tpc_threadpool_new((int32_t)(n))
#define TPC_THREADPOOL_SHUTDOWN(tp) tpc_threadpool_shutdown((tp))

/* Task parallel API
 * ---------------------------------------------------------------- */

/* Generators over the type-only decls list: ty0 through ty7.
 * They build the task data fields plus the spawn-function parameters,
 * and the synthetic name list for the body call.
 */
#define TPC_FIELDS_1(ty0) ty0 tpc_t0;
#define TPC_FIELDS_2(ty0, ty1) ty0 tpc_t0; ty1 tpc_t1;
#define TPC_FIELDS_3(ty0, ty1, ty2) ty0 tpc_t0; ty1 tpc_t1; ty2 tpc_t2;
#define TPC_FIELDS_4(ty0, ty1, ty2, ty3) \
    ty0 tpc_t0; ty1 tpc_t1; ty2 tpc_t2; ty3 tpc_t3;
#define TPC_FIELDS_5(ty0, ty1, ty2, ty3, ty4) \
    ty0 tpc_t0; ty1 tpc_t1; ty2 tpc_t2; ty3 tpc_t3; ty4 tpc_t4;
#define TPC_FIELDS_6(ty0, ty1, ty2, ty3, ty4, ty5) \
    ty0 tpc_t0; ty1 tpc_t1; ty2 tpc_t2; ty3 tpc_t3; ty4 tpc_t4; \
    ty5 tpc_t5;
#define TPC_FIELDS_7(ty0, ty1, ty2, ty3, ty4, ty5, ty6) \
    ty0 tpc_t0; ty1 tpc_t1; ty2 tpc_t2; ty3 tpc_t3; ty4 tpc_t4; \
    ty5 tpc_t5; ty6 tpc_t6;
#define TPC_FIELDS_8(ty0, ty1, ty2, ty3, ty4, ty5, ty6, ty7) \
    ty0 tpc_t0; ty1 tpc_t1; ty2 tpc_t2; ty3 tpc_t3; ty4 tpc_t4; \
    ty5 tpc_t5; ty6 tpc_t6; ty7 tpc_t7;
#define TPC_FIELDS_IMPL(n, ...) TPC_CONCAT(TPC_FIELDS_, n)(__VA_ARGS__)
#define TPC_FIELDS(n, ...) TPC_FIELDS_IMPL(n, __VA_ARGS__)

#define TPC_PARAMS_1(ty0) ty0 tpc_t0
#define TPC_PARAMS_2(ty0, ty1) ty0 tpc_t0, ty1 tpc_t1
#define TPC_PARAMS_3(ty0, ty1, ty2) ty0 tpc_t0, ty1 tpc_t1, ty2 tpc_t2
#define TPC_PARAMS_4(ty0, ty1, ty2, ty3) \
    ty0 tpc_t0, ty1 tpc_t1, ty2 tpc_t2, ty3 tpc_t3
#define TPC_PARAMS_5(ty0, ty1, ty2, ty3, ty4) \
    ty0 tpc_t0, ty1 tpc_t1, ty2 tpc_t2, ty3 tpc_t3, ty4 tpc_t4
#define TPC_PARAMS_6(ty0, ty1, ty2, ty3, ty4, ty5) \
    ty0 tpc_t0, ty1 tpc_t1, ty2 tpc_t2, ty3 tpc_t3, ty4 tpc_t4, \
        ty5 tpc_t5
#define TPC_PARAMS_7(ty0, ty1, ty2, ty3, ty4, ty5, ty6) \
    ty0 tpc_t0, ty1 tpc_t1, ty2 tpc_t2, ty3 tpc_t3, ty4 tpc_t4, \
        ty5 tpc_t5, ty6 tpc_t6
#define TPC_PARAMS_8(ty0, ty1, ty2, ty3, ty4, ty5, ty6, ty7) \
    ty0 tpc_t0, ty1 tpc_t1, ty2 tpc_t2, ty3 tpc_t3, ty4 tpc_t4, \
        ty5 tpc_t5, ty6 tpc_t6, ty7 tpc_t7
#define TPC_PARAMS_IMPL(n, ...) TPC_CONCAT(TPC_PARAMS_, n)(__VA_ARGS__)
#define TPC_PARAMS(n, ...) TPC_PARAMS_IMPL(n, __VA_ARGS__)

#define TPC_NAMES_1 tpc_t0
#define TPC_NAMES_2 tpc_t0, tpc_t1
#define TPC_NAMES_3 tpc_t0, tpc_t1, tpc_t2
#define TPC_NAMES_4 tpc_t0, tpc_t1, tpc_t2, tpc_t3
#define TPC_NAMES_5 tpc_t0, tpc_t1, tpc_t2, tpc_t3, tpc_t4
#define TPC_NAMES_6 tpc_t0, tpc_t1, tpc_t2, tpc_t3, tpc_t4, tpc_t5
#define TPC_NAMES_7 tpc_t0, tpc_t1, tpc_t2, tpc_t3, tpc_t4, tpc_t5, tpc_t6
#define TPC_NAMES_8 tpc_t0, tpc_t1, tpc_t2, tpc_t3, tpc_t4, tpc_t5, \
    tpc_t6, tpc_t7
#define TPC_NAMES(n) TPC_CONCAT(TPC_NAMES_, n)

/* Pack the spawn parameters into the task data, one field per param */
#define TPC_PACKF_1(tpc_d) (tpc_d)->tpc_t0 = tpc_t0
#define TPC_PACKF_2(tpc_d) TPC_PACKF_1(tpc_d); (tpc_d)->tpc_t1 = tpc_t1
#define TPC_PACKF_3(tpc_d) TPC_PACKF_2(tpc_d); (tpc_d)->tpc_t2 = tpc_t2
#define TPC_PACKF_4(tpc_d) TPC_PACKF_3(tpc_d); (tpc_d)->tpc_t3 = tpc_t3
#define TPC_PACKF_5(tpc_d) TPC_PACKF_4(tpc_d); (tpc_d)->tpc_t4 = tpc_t4
#define TPC_PACKF_6(tpc_d) TPC_PACKF_5(tpc_d); (tpc_d)->tpc_t5 = tpc_t5
#define TPC_PACKF_7(tpc_d) TPC_PACKF_6(tpc_d); (tpc_d)->tpc_t6 = tpc_t6
#define TPC_PACKF_8(tpc_d) TPC_PACKF_7(tpc_d); (tpc_d)->tpc_t7 = tpc_t7
#define TPC_PACKF_IMPL(tpc_d, n) TPC_CONCAT(TPC_PACKF_, n)(tpc_d)
#define TPC_PACKF(tpc_d, n) TPC_PACKF_IMPL(tpc_d, n)

/* Void task, fire-and-forget: the runner frees the task on completion.
 * On completion the runner frees the task.
 */
#define TPC_DEFINE_SPAWN_PASS(name, ...)                                 \
    TPC_DEFINE_SPAWN_IMPL(name, TPC_VA_NARGS(__VA_ARGS__), __VA_ARGS__)
#define TPC_DEFINE_SPAWN_IMPL(name, tpc_n, ...)                          \
    struct tpc_##name##_task_data { TPC_FIELDS(tpc_n, __VA_ARGS__) };    \
    static void name##_task_func(void* tpc_env_) {                       \
        struct tpc_##name##_task_data *tpc_d =                           \
            (struct tpc_##name##_task_data *)(                           \
                (unsigned char*)tpc_env_ + TPC_FLOWVAR_HEADER_SIZE);    \
        TPC_UNPACK_N(tpc_d, tpc_n);                                      \
        (void)tpc_d;                                                     \
        name(TPC_NAMES(tpc_n));                                          \
    }                                                                    \
    static void name##_spawn(TPC_PARAMS(tpc_n, __VA_ARGS__)) {           \
        tpc_worker_context* tpc_ctx = tpc_workerContext();               \
        struct tpc_##name##_task_data tpc_d;                             \
        TPC_PACKF(&tpc_d, tpc_n);                                        \
        tpc_task* tpc_task_ = tpc_task_newSpawnEnv(                      \
            tpc_ctx->currentTask, tpc_ctx->currentScope,                 \
            name##_task_func, &tpc_d, (int32_t)sizeof(tpc_d));           \
        tpc_schedule(tpc_ctx, tpc_task_, false);                         \
    }
#define TPC_DEFINE_SPAWN(name, decls)                                    \
    TPC_DEFINE_SPAWN_PASS(name, TPC_ARGS decls)

/* Awaitable void task: the result slot reads true on completion,
 * awaiting blocks until the task completes.
 */
#define TPC_DEFINE_SPAWN_AWAITABLE_PASS(name, ...)                       \
    TPC_DEFINE_SPAWN_AWAITABLE_IMPL(name, TPC_VA_NARGS(__VA_ARGS__),     \
                                    __VA_ARGS__)
#define TPC_DEFINE_SPAWN_AWAITABLE_IMPL(name, tpc_n, ...)                \
    struct tpc_##name##_task_data { TPC_FIELDS(tpc_n, __VA_ARGS__) };    \
    static void name##_task_func(void* tpc_env_) {                       \
        tpc_task* tpc_this = *(tpc_task**)tpc_env_;                     \
        struct tpc_##name##_task_data *tpc_d =                           \
            (struct tpc_##name##_task_data *)(                           \
                (unsigned char*)tpc_env_ + TPC_FLOWVAR_HEADER_SIZE);    \
        TPC_UNPACK_N(tpc_d, tpc_n);                                      \
        name(TPC_NAMES(tpc_n));                                          \
        bool tpc_done = true;                                            \
        tpc_task_readyWith(tpc_this, &tpc_done, sizeof(tpc_done));       \
    }                                                                    \
    static tpc_flowvar name##_spawn(TPC_PARAMS(tpc_n, __VA_ARGS__)) {    \
        tpc_worker_context* tpc_ctx = tpc_workerContext();               \
        struct tpc_##name##_task_data tpc_d;                             \
        TPC_PACKF(&tpc_d, tpc_n);                                        \
        tpc_task* tpc_task_ = tpc_task_newSpawnEnv(                      \
            tpc_ctx->currentTask, tpc_ctx->currentScope,                 \
            name##_task_func, &tpc_d, (int32_t)sizeof(tpc_d));           \
        tpc_flowvar tpc_fv_ = tpc_flowvar_new(tpc_task_);                \
        tpc_schedule(tpc_ctx, tpc_task_, false);                         \
        return tpc_fv_;                                                  \
    }
#define TPC_DEFINE_SPAWN_AWAITABLE(name, decls)                          \
    TPC_DEFINE_SPAWN_AWAITABLE_PASS(name, TPC_ARGS decls)

/* Result task: the payload starts with the result slot.
 * env layout = | ptr Task | result | arg0 | arg1 | ... | argN |
 * Sync reads the slot and the body writes it.
 */
#define TPC_DEFINE_SPAWN_RET_PASS(rty, name, ...)                           \
    TPC_DEFINE_SPAWN_RET_IMPL(rty, name, TPC_VA_NARGS(__VA_ARGS__),         \
                           __VA_ARGS__)
#define TPC_DEFINE_SPAWN_RET_IMPL(rty, name, tpc_n, ...)                    \
    struct tpc_##name##_task_data { rty tpc_result;                      \
                                    TPC_FIELDS(tpc_n, __VA_ARGS__) };    \
    static void name##_task_func(void* tpc_env_) {                       \
        tpc_task* tpc_this = *(tpc_task**)tpc_env_;                     \
        struct tpc_##name##_task_data *tpc_d =                           \
            (struct tpc_##name##_task_data *)(                           \
                (unsigned char*)tpc_env_ + TPC_FLOWVAR_HEADER_SIZE);    \
        TPC_UNPACK_N(tpc_d, tpc_n);                                      \
        rty tpc_r = name(TPC_NAMES(tpc_n));                              \
        tpc_task_readyWith(tpc_this, &tpc_r, sizeof(rty));               \
    }                                                                    \
    static tpc_flowvar name##_spawn(TPC_PARAMS(tpc_n, __VA_ARGS__)) {    \
        tpc_worker_context* tpc_ctx = tpc_workerContext();               \
        struct tpc_##name##_task_data tpc_d;                             \
        TPC_PACKF(&tpc_d, tpc_n);                                        \
        tpc_task* tpc_task_ = tpc_task_newSpawnEnv(                      \
            tpc_ctx->currentTask, tpc_ctx->currentScope,                 \
            name##_task_func, &tpc_d, (int32_t)sizeof(tpc_d));           \
        tpc_flowvar tpc_fv_ = tpc_flowvar_new(tpc_task_);                \
        tpc_schedule(tpc_ctx, tpc_task_, false);                         \
        return tpc_fv_;                                                  \
    }
#define TPC_DEFINE_SPAWN_RET(rty, name, decls)                              \
    TPC_DEFINE_SPAWN_RET_PASS(rty, name, TPC_ARGS decls)

/* Spawn site: TPC_SPAWN(fn, vals) calls the task's generated spawn function,
 * one site covers every task shape.
 */
#define TPC_SPAWN(fn, vals) TPC_CONCAT(fn, _spawn) vals
#define TPC_SPAWN_INTO(fv, fn, vals) ((fv) = TPC_SPAWN(fn, vals))

/* Sync a flowvar into the rty value. */
#define TPC_SYNC(fv, rty)                                                \
    ({                                                                   \
        tpc_flowvar tpc_fv_ = (fv);                                      \
        rty tpc_r_;                                                      \
        tpc_flowvar_sync(&tpc_r_, &tpc_fv_, sizeof(rty));                \
        tpc_r_;                                                          \
    })

/* Data parallel API
 * ---------------------------------------------------------------- */

/* Declare a parallel-for: the decls are NAMED struct members
 * (int n, float* x). The body unpacks them by name.
 */
#define TPC_DEFINE_PARFOR(name, decls)                                   \
    struct tpc_##name##_data { decls; };                                 \
    static void name##_iter(int32_t tpc_i,                               \
                            struct tpc_##name##_data *tpc_d_);           \
    static void name##_wrap(int32_t tpc_i, void* tpc_env_) {             \
        name##_iter(tpc_i, (struct tpc_##name##_data *)tpc_env_);        \
    }                                                                    \
    static void name##_task_func(void* tpc_env_) {                       \
        tpc_task* tpc_this = *(tpc_task**)tpc_env_;                     \
        tpc_worker_context* tpc_ctx = tpc_workerContext();               \
        tpc_parallelForRunner(tpc_ctx, tpc_this, name##_wrap);           \
    }

#define TPC_PARFOR_BODY(name, i, names, ...)                             \
    static void name##_iter(int32_t tpc_i,                               \
                            struct tpc_##name##_data *tpc_d_) {          \
        (void)tpc_d_;                                                    \
        int32_t i = tpc_i;                                               \
        TPC_UNPACK_MEMBER(tpc_d_, TPC_VA_NARGS names, TPC_ARGS names);   \
        __VA_ARGS__;                                                     \
    }

#define TPC_UNPACK_MEMBER_1(x, n0)                                      \
    __typeof__((x)->n0) n0 = (x)->n0
#define TPC_UNPACK_MEMBER_2(x, n0, n1)                                  \
    TPC_UNPACK_MEMBER_1(x, n0);                                         \
    __typeof__((x)->n1) n1 = (x)->n1
#define TPC_UNPACK_MEMBER_3(x, n0, n1, n2)                              \
    TPC_UNPACK_MEMBER_2(x, n0, n1);                                     \
    __typeof__((x)->n2) n2 = (x)->n2
#define TPC_UNPACK_MEMBER_4(x, n0, n1, n2, n3)                          \
    TPC_UNPACK_MEMBER_3(x, n0, n1, n2);                                 \
    __typeof__((x)->n3) n3 = (x)->n3
#define TPC_UNPACK_MEMBER_IMPL(x, n, ...)                               \
    TPC_CONCAT(TPC_UNPACK_MEMBER_, n)(x, __VA_ARGS__)
#define TPC_UNPACK_MEMBER(x, n, ...) TPC_UNPACK_MEMBER_IMPL(x, n, __VA_ARGS__)

/* Pack the launch values into the parfor struct positionally.
 * Member order comes from the decls.
 */
#define TPC_PACK_POS(x, ...) (*(x) = (__typeof__(*(x))){__VA_ARGS__})

#define TPC_PARFOR(name, start_, stop_, vals)                            \
    do {                                                                 \
        tpc_worker_context* tpc_ctx = tpc_workerContext();               \
        struct tpc_##name##_data tpc_d;                                  \
        TPC_PACK_POS(&tpc_d, TPC_ARGS vals);                             \
        tpc_task* tpc_task_ = tpc_task_newLoopEnv(                       \
            tpc_ctx->currentTask, tpc_ctx->currentScope,                 \
            name##_task_func, &tpc_d,                                    \
            (int32_t)sizeof(struct tpc_##name##_data),                   \
            (int32_t)(start_), (int32_t)(stop_), 1, true);               \
        tpc_schedule(tpc_ctx, tpc_task_, false);                         \
    } while (0)

/* Scoping
 * ---------------------------------------------------------------- */

#define TPC_SYNC_SCOPE(body)                                             \
    ({                                                                   \
        tpc_scoped_barrier tpc_sb_;                                      \
        tpc_scoped_barrier_initialize(&tpc_sb_);                         \
        tpc_worker_context* tpc_ctx_ = tpc_workerContext();              \
        tpc_scoped_barrier* tpc_suspended_ = tpc_ctx_->currentScope;     \
        tpc_ctx_->currentScope = &tpc_sb_;                               \
        body;                                                            \
        tpc_syncScopeWait(&tpc_sb_);                                     \
        tpc_ctx_->currentScope = tpc_suspended_;                         \
        0;                                                               \
    })
#ifdef __cplusplus
}
#endif


#endif /* THREADPOOL_COMPUTE_H */
