## Metal: coordinate builtins thread transitively through device-fn call chains.
## `bump` reads `thread_index_in_threadgroup`; `mid` reads
## `thread_position_in_threadgroup.x` and calls `bump`; the kernels call them.
## The kernels receive the attribute params and forward them through every
## level, so the body identifiers bind at each depth. The second kernel also
## references the builtin itself: one attribute param serves both the kernel
## body and the forwarding args (dedup) — a duplicate param would fail MSL
## compilation at ingest.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_device_fn_builtin_chain.nim

import std/unittest
import workspace/crucible

proc bump(p: ptr UncheckedArray[uint32]) {.device.} =
  let t = thread_index_in_threadgroup
  p[t] = p[t] + 100'u32

proc mid(p: ptr UncheckedArray[uint32]) {.device.} =
  let x = thread_position_in_threadgroup.x
  p[x] = x
  bump(p)

const chainMsl = metal:
  proc chainKernel(p: ptr UncheckedArray[uint32]) {.global.} =
    mid(p)

  proc directKernel(o: ptr UncheckedArray[uint32]) {.global.} =
    let t = thread_index_in_threadgroup
    o[t] = t
    bump(o)

proc runTest() =
  suite "Metal - transitive builtin threading through device-fn chains":
    test "two-level chain forwards both builtins (kernel -> mid -> bump)":
      var engine = bkMetal.init()
      engine.ingest(chainMsl)
      var res: array[32, uint32]
      engine.run<<(grid: (1, 1), blk: (32, 1))>>("chainKernel", res, ())
      for i in 0 ..< 32:
        check res[i] == uint32(i + 100)

    test "kernel own ref and the chain share one attribute param":
      var engine = bkMetal.init()
      engine.ingest(chainMsl)
      var res: array[32, uint32]
      engine.run<<(grid: (1, 1), blk: (32, 1))>>("directKernel", res, ())
      for i in 0 ..< 32:
        check res[i] == uint32(i + 100)

when isMainModule:
  runTest()
