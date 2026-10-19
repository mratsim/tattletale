## Metal attribute-param semantics: the five coordinate builtins carry real
## thread-space values and the Apple-documented identity
## thread_position_in_grid == threadgroup_position_in_grid * threads_per_threadgroup
## + thread_position_in_threadgroup holds per axis for all 64 work-items.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_attr_semantics.nim

import std/unittest
import workspace/crucible

const attrSemMsl = metal:
  proc attrSem(C: ptr UncheckedArray[uint32]) {.global, workgroup: (2, 2, 2).} =
    let gx = thread_position_in_grid.x
    let gy = thread_position_in_grid.y
    let gz = thread_position_in_grid.z
    let base = ((gz * 4 + gy) * 4 + gx) * 6
    C[base + 0] = uint32(gx)
    C[base + 1] = uint32(threadgroup_position_in_grid.x)
    C[base + 2] = uint32(threads_per_threadgroup.x)
    C[base + 3] = uint32(threadgroups_per_grid.x)
    C[base + 4] = uint32(thread_position_in_threadgroup.x)
    # per-axis identity: global == tg * tpt + local (0 when true, 1 when broken)
    let tgx = threadgroup_position_in_grid.x
    let tgy = threadgroup_position_in_grid.y
    let tgz = threadgroup_position_in_grid.z
    let tptx = threads_per_threadgroup.x
    let tpty = threads_per_threadgroup.y
    let tptz = threads_per_threadgroup.z
    let tx = thread_position_in_threadgroup.x
    let ty = thread_position_in_threadgroup.y
    let tz = thread_position_in_threadgroup.z
    C[base + 5] = if gx == tgx * tptx + tx and
                     gy == tgy * tpty + ty and
                     gz == tgz * tptz + tz: 0'u32 else: 1'u32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo attrSemMsl

  suite "Metal - attribute-param semantics":

    test "all five attribute params bind to real thread-space values":
      var engine = bkMetal.newEngine()
      engine.ingest(attrSemMsl)
      let oDev = engine.allocDevice(uint32, 64 * 6)
      engine.launchKernel<<(grid: (2, 2, 2), blk: (2, 2, 2))>>("attrSem", oDev, ())
      var res: array[64 * 6, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 64 * 6)
      # global (0,0,0): tg=(0,0,0), tpt=(2,2,2), tpg=(2,2,2), local=(0,0,0)
      check res[0] == 0'u32          # global x
      check res[1] == 0'u32          # tg x
      check res[2] == 2'u32          # tpt x
      check res[3] == 2'u32          # tpg x
      check res[4] == 0'u32          # local x
      # global (1,0,0): local=(1,0,0), tg=(0,0,0)
      check res[6 + 0] == 1'u32      # global x
      check res[6 + 4] == 1'u32      # local x
      # global (2,0,0): tg=(1,0,0), local=(0,0,0)
      check res[12 + 1] == 1'u32     # tg x
      check res[12 + 4] == 0'u32     # local x
      # global (3,0,0): tg=(1,0,0), local=(1,0,0)
      check res[18 + 1] == 1'u32     # tg x
      check res[18 + 4] == 1'u32     # local x
      # global (1,1,1): linear index 21, tg=(0,0,0), local=(1,1,1)
      check res[126 + 0] == 1'u32    # global x
      check res[126 + 1] == 0'u32    # tg x
      check res[126 + 4] == 1'u32    # local x
      # identity holds for every thread
      var base = 0
      while base < 64 * 6:
        check res[base + 5] == 0'u32
        base += 6

when isMainModule:
  runTest()
