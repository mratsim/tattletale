# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Smoke test for the threadpool_compute Nim wrapper:
# a spawn storm of fire-and-forget tasks and one result task
# synced through a flowvar.

import std/syncio

import threadpool_compute

const numTasks = 1000

type SlotEnv = object
  slot: ptr int32

proc setTask(env: pointer) {.cdecl.} =
  let e = cast[ptr SlotEnv](cast[int](env) + FlowvarHeaderSize)
  e.slot[] = 1

type MulEnv = object
  a, b: int32

proc mulTask(env: pointer) {.cdecl.} =
  let e = cast[ptr MulEnv](cast[int](env) + FlowvarHeaderSize)
  let r = e.a * e.b
  let self = cast[ptr Task](cast[ptr pointer](env)[])
  task_readyWith(self, r.unsafeAddr, csize_t sizeof r)

proc main() =
  let tp = threadpool_new(8)
  var slots: array[numTasks, int32]
  for i in 0 ..< numTasks:
    var env = SlotEnv(slot: slots[i].addr)
    spawnVoid(setTask, env)
  syncAll(tp)
  for i in 0 ..< numTasks:
    doAssert slots[i] == 1, "task " & $i & " did not run"

  var env = MulEnv(a: 6, b: 7)
  let f = spawn(mulTask, env)
  doAssert sync[int32](f) == 42
  stdout.writeLine "t_nim_api: OK"
  threadpool_shutdown(tp)

main()
