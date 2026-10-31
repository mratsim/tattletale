# threadpool_compute

A C11 work-stealing threadpool with a macro frontend shaped after
[constantine](https://github.com/mratsim/constantine)'s Nim threadpool:
`spawn` everything, `sync` what you need,
`parallelFor` for data parallel loops. One C file set, no dependencies beyond the platform, GCC and Clang only.

## Goal

GPU kernels will be lowered to C for CPU execution and they need a performant
threadpool as a backend, so Constantine's threadpool was ported from Nim to C.

## Quick start

```c
#include "threadpool_compute.h"

static int64_t parfib(int32_t n);
TPC_DEFINE_SPAWN_RET(int64_t, parfib, (int32_t))

static int64_t parfib(int32_t n) {
    if (n < 2) {
        return n;
    }
    tpc_flowvar a = TPC_SPAWN(parfib, (n - 1));
    int64_t b = parfib(n - 2);
    return TPC_SYNC(a, int64_t) + b;
}

int main(void) {
    tpc_threadpool* tp = TPC_THREADPOOL_INIT(8);
    tpc_flowvar f = TPC_SPAWN(parfib, (30));
    printf("fib(30) = %lld\n", (long long)TPC_SYNC(f, int64_t));
    TPC_THREADPOOL_SHUTDOWN(tp);
}
```

Build and link:

```sh
make                 # build/libthreadpool_compute.a + test binaries
make test            # run the test suite
```

Tests and benches go to `build/`. Link your program against
`build/libthreadpool_compute.a` and add `-lpthread`.

## API

Task parallel:

| Macro | Role |
|---|---|
| `TPC_DEFINE_SPAWN(name, (ty0, ty1, ...))` | declare a void task, fire and forget |
| `TPC_DEFINE_SPAWN_AWAITABLE(name, (ty0, ...))` | declare a void task with a completion flowvar |
| `TPC_DEFINE_SPAWN_RET(rty, name, (ty0, ...))` | declare a task returning `rty` |
| `TPC_SPAWN(fn, (v0, v1, ...))` | spawn; yields a `tpc_flowvar` for awaitable and ret tasks |
| `TPC_SPAWN_INTO(fv, fn, (vals))` | spawn, write the flowvar handle into `fv` |
| `TPC_SYNC(fv, rty)` | wait for the flowvar, yield the result |

Data parallel:

| Macro | Role |
|---|---|
| `TPC_DEFINE_PARFOR(name, ty i; ty x; ...)` | declare a parallel-for and its iteration state |
| `TPC_PARFOR_BODY(name, i, (i, x, ...), { body })` | per-iteration body |
| `TPC_PARFOR(name, start, stop, (vals))` | launch `[start, stop)` |

Scoping and runtime:

| Macro | Role |
|---|---|
| `TPC_SYNC_SCOPE({ body })` | wait for everything spawned inside the block |
| `TPC_THREADPOOL_INIT(n)` | create a pool with `n` workers (worker 0 is the calling thread) |
| `TPC_THREADPOOL_SHUTDOWN(tp)` | finish pending work and join |

Task bodies are plain C functions.
`TPC_DEFINE_*` wraps it, `TPC_SPAWN` calls the generated spawn function.
Spawning from inside a task or a `parallelFor` body is the normal case.
Children complete before their parent's scope leaves.

A `parallelFor` must run inside `TPC_SYNC_SCOPE`, where the loop
waits for its chunks:

```c
TPC_DEFINE_PARFOR(axpy, float a; float* x; float* y;)
TPC_PARFOR_BODY(axpy, i, (a, x, y), {
    y[i] += a * x[i];
})

TPC_SYNC_SCOPE({
    TPC_PARFOR(axpy, 0, n, (2.0f, x, y));
});
```

## Nim

Wrapper source lives at `workspace/threadpool_compute/threadpool_compute.nim`, reexported by the root file.
Import with the `workspace` directory on the Nim path:

```nim
import threadpool_compute

type MulEnv = object
  a, b: int32

proc mulTask(env: pointer) {.cdecl.} =
  let e = cast[ptr MulEnv](cast[int](env) + FlowvarHeaderSize)
  let r = e.a * e.b
  let self = cast[ptr Task](cast[ptr pointer](env)[])
  task_readyWith(self, r.unsafeAddr, csize_t sizeof r)

let tp = threadpool_new(8)
var env = MulEnv(a: 6, b: 7)
let f = spawn(mulTask, env)
doAssert sync[int32](f) == 42
threadpool_shutdown(tp)
```

A task body reads its payload after `FlowvarHeaderSize` where the first env
word is the task self-pointer. `tests/t_nim_api.nim` runs the smoke test:

```sh
nim c --path:../.. -r tests/t_nim_api.nim
```

See also [DESIGN.md](DESIGN.md) and [RANDOM_PERMUTATIONS.md](RANDOM_PERMUTATIONS.md).
