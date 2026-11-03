# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU engine types for the Crucible CPU backend.
##
## `CpuEngine` owns the threadpool_compute worker pool.
##
## The engine is exempt from the HwEngine concept:
## arguments are host values, there is no ingest, no artifact and no DeviceMem.

import std/os
import std/cpuinfo
import ./engines_common
when compileOption("threads"):
  import workspace/threadpool_compute
  when defined(macosx):
    # Apple's SDK ships no C11 threads.h, threadpool.h includes it for thrd_create and thrd_join, the shim maps those onto pthreads
    {.passc: "-I\"" & currentSourcePath.parentDir / "c11threads_darwin" & "\"".}

# ═══════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════

type
  CpuEngine* = object
    ## CPU execution engine, the owner of the worker pool.
    deviceName*: string
    deviceVendor*: string
    numThreads*: int
    when compileOption("threads"):
      tp*: ptr Threadpool

proc `=copy`*(dst: var CpuEngine, src: CpuEngine) {.
    error: "CpuEngine owns its worker pool and is non-copyable".}

proc newCpuEngine*(numThreads: int = 0): CpuEngine =
  ## numThreads 0 creates the pool with the OS processor count.
  result.deviceName = "crucible-cpu"
  result.deviceVendor = "cpu"
  when compileOption("threads"):
    let n = if numThreads > 0: numThreads else: countProcessors()
    result.tp = threadpool_new(int32(n))
    result.numThreads = n
  else:
    result.numThreads = 1

proc destroyCpuEngine*(engine: var CpuEngine) =
  ## Pool shutdown, joins the workers and frees the pool memory.
  ## Call once per engine, from the thread that created it.
  when compileOption("threads"):
    doAssert engine.tp != nil,
      "destroyCpuEngine: the engine's pool is already shut down"
    threadpool_shutdown(engine.tp)
    engine.tp = nil
