/* threadpool_runtime: every library source compiled as one translation unit.
 *
 * One unit lets the compiler inline across module boundaries and gives
 * embedders a single file to compile. Platform sources select by preprocessor,
 * no build system needs to know the platform.
 * Compile with -Isrc/threadpool -Iinclude, then link -lpthread.
 * Windows needs no extra library.
 */

/* glibc declares the syscall wrappers only under _GNU_SOURCE and system
 * headers expand once per unit, so the macro must precede every include.
 * Per-file builds declared it on the first line of futexes_linux.c,
 * one unit loses that ordering.
 */
#if defined(__linux__) && !defined(_GNU_SOURCE)
  #define _GNU_SOURCE
#endif

#include "threadpool.h"
#include "instrumentation.h"
#include "crossthread/backoff.h"
#include "crossthread/scoped_barriers.h"
#include "crossthread/taskqueues.h"
#include "crossthread/tasks_flowvars.h"
#include "parallel_offloading.h"
#include "primitives/barriers.h"
#include "primitives/futexes.h"
#include "primitives/topology.h"

#include "threadpool.c"
#include "crossthread/taskqueues.c"
#include "flow_events.c"
