# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## threadpool_compute flow events over the Nim wrapper
##
## Test that the Nim wrapper defers tasks on single, iteration and multi-dependency
## events and schedules them on trigger
##
## Run:
##   nim c --path:../.. -r tests/t_nim_flowevents.nim

import std/syncio

import threadpool_compute

type TaskStack = object
  tasks: seq[ptr Task]

proc stackAdd(queue: pointer, task: ptr Task) {.cdecl.} =
  cast[ptr TaskStack](queue).tasks.add(task)

proc newTask(): ptr Task =
  cast[ptr Task](alloc0(sizeof(Task)))

proc main() =
  var stack = TaskStack()

  block: # single event, one deferred task plus one late self-scheduled task
    var ev: FlowEvent
    flowEvent_initSingle(ev)
    let task = newTask()
    doAssert task.flowEvent_delayedUntil(ev)
    doAssert stack.tasks.len == 0
    flowEvent_trigger(ev, stack.addr, stackAdd)
    doAssert stack.tasks.len == 1
    let late = newTask()
    doAssert not late.flowEvent_delayedUntil(ev)
    doAssert stack.tasks.len == 1
    flowEvent_unref(ev)

  block: # iteration event, deferral and trigger on bucket 3
    var ev: FlowEvent
    flowEvent_initIter(ev, 0, 10, 1)
    let task = newTask()
    doAssert task.flowEvent_delayedUntilIter(ev, 3)
    doAssert stack.tasks.len == 1 # includes the task from the block above
    flowEvent_triggerIter(ev, 3, stack.addr, stackAdd)
    doAssert stack.tasks.len == 2
    let late = newTask()
    doAssert not late.flowEvent_delayedUntilIter(ev, 3)
    doAssert stack.tasks.len == 2
    flowEvent_unref(ev)

  block: # multi-dependency: a task schedules only after the second trigger
    var evA, evB: FlowEvent
    flowEvent_initSingle(evA)
    flowEvent_initSingle(evB)
    let task = newTask()
    let deps = [FlowEventDep(event: evA, index: noIterDep),
                FlowEventDep(event: evB, index: noIterDep)]
    doAssert task.flowEvent_delayUntilMulti(deps)
    doAssert stack.tasks.len == 2
    flowEvent_trigger(evA, stack.addr, stackAdd)
    doAssert stack.tasks.len == 2 # re-deferred on evB
    flowEvent_trigger(evB, stack.addr, stackAdd)
    doAssert stack.tasks.len == 3
    flowEvent_unref(evA)
    flowEvent_unref(evB)

  stdout.writeLine "t_nim_flowevents: OK"

main()
