# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Host-side bounded wait for the megakernel launches.
##
## The engine's launch path ends in `waitUntilCompleted`, an unbounded host
## wait. A device grid stuck in a spin never completes and the host spins
## with it until the process is killed.
##
## - `runMegaBounded` enqueues the launch on a worker thread
## - the host polls the command-buffer completion with a wall-clock deadline
## - expiry reads the stage counters directly from the launch's shared
##   counters page and exits nonzero with the per-stage values
##   (the no-copy binding keeps the page host-visible while the grid runs)
##
## A wedged grid cannot be unwound in-process, so the diagnostic is terminal.
## TTT_MegaWaitDeadlineSecMs=<n> gives the default deadline `-d:TTT_MegaWaitDeadlineSecMs=<n>`.

import std/[monotimes, times, os, strformat]

const TTT_MegaWaitDeadlineSecMs* {.intdefine.} = 20_000
  ## Default deadline in milliseconds, one config point.

type LaunchCtx = ref object
  launch: proc() {.gcsafe.}

proc launchWorker(ctx: LaunchCtx) {.thread.} =
  ## Runs the caller's one-dispatch closure to completion, never returning early.
  ##
  ## Precondition:
  ## the wrapper's deadline owns the reporting once a grid wedges, this thread
  ## never unwinds it.
  ctx.launch()

proc runMegaBounded*[C: static int](
    launch: proc() {.gcsafe.};
    counters: ptr UncheckedArray[uint32];
    stageNames: array[C, string];
    deadlineMs: float = TTT_MegaWaitDeadlineSecMs.float;
    onExpiry: proc(msg: string) {.gcsafe.} = nil) =
  ## One megakernel launch under a wall-clock deadline.
  ##
  ## Contract:
  ##
  ## - `launch` performs exactly one engine dispatch and returns
  ## - the wrapper runs each launch on a worker thread and joins it, keeping
  ##   launches serialized exactly as an unwrapped `engine.run` sequence
  ## - `counters` is the launch's host-visible counters page and `stageNames` gives the labels
  ##
  ## Expiry:
  ##
  ## - the counters page is read as the wedged grid left it, the stuck
  ##   stage named per the given labels
  ## - the default expiry is terminal, stderr message then exit nonzero
  ## - `onExpiry` records the expiry and returns, the wedged worker thread
  ##   is then never joined, it holds the engine until the process exits
  ##   and no further launch on that engine is possible
  var ctx = LaunchCtx(launch: launch)
  var th: Thread[LaunchCtx]
  createThread(th, launchWorker, ctx)
  let t0 = getMonoTime()
  while th.running:
    sleep 50
    if (getMonoTime() - t0).inMilliseconds.float / 1000.0 > deadlineMs / 1000.0:
      var msg = &"megakernel launch did not complete within " &
        &"{deadlineMs / 1000.0:.1f}s, stage counters at expiry: "
      for i in 0 ..< C:
        msg.add &"{stageNames[i]}={counters[i]} "
      if onExpiry.isNil:
        stderr.writeLine msg
        quit(1)
      onExpiry(msg)
      return
  joinThread(th)
