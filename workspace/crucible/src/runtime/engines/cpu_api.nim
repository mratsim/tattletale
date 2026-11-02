# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU engine types for the Crucible CPU backend.
##
## `CpuEngine` owns the threadpool_compute worker pool.
## `launchCpuFn` executes compile-time-generated kernels on it.
##
## - kernels compile at the launch site into a loop nest with the body inlined,
##   one task per workgroup, threads serial inside the task
## - the engine is exempt from the HwEngine concept
##
##   - arguments are host values, there is no ingest, no artifact and no
##     DeviceMem
##   - the surface is `runCpu` plus `launchCpuFn`
##
## - with `--threads:on` the pool runs one workgroup task per workgroup
##   through the threadpool_compute C runtime, the launch emits C
## - without threads the launch runs serially on the caller thread, the parallel variant must reproduce that output bitwise

import std/osproc
import ./engines_common
when compileOption("threads"):
  import workspace/threadpool_compute

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

proc newCpuEngine*(numThreads: int = 0): CpuEngine =
  ## CPU engine constructor.
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
    threadpool_shutdown(engine.tp)
    engine.tp = nil
