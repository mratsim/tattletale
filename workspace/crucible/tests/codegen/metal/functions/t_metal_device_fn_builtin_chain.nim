## Metal: coordinate builtins thread transitively through device-fn call chains.
## One threadgroup of 32 threads runs per kernel, every slot holds its lane index + 100.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_device_fn_builtin_chain.nim

import std/unittest
import workspace/crucible

proc bump(p: ptr UncheckedArray[uint32]) {.device.} =
  let t = thread_index_in_threadgroup
  p[t] = p[t] + 100'u32

proc mid(p: ptr UncheckedArray[uint32]) {.device.} =
  let x = thread_position_in_threadgroup.x
  p[x] = uint32(x)
  bump(p)

const chainMsl = metal:
  proc chainKernel(p: ptr UncheckedArray[uint32]) {.global.} =
    mid(p)

  proc directKernel(o: ptr UncheckedArray[uint32]) {.global.} =
    let t = thread_index_in_threadgroup
    o[t] = uint32(t)
    bump(o)

proc runTest() =
  suite "Metal - transitive builtin threading through device-fn chains":
    test "two-level chain forwards both builtins (kernel -> mid -> bump)":
      var engine = bkMetal.init()
      engine.ingest(chainMsl)
      let oDev = engine.allocDevice(uint32, 32)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("chainKernel", oDev, ())
      var res: array[32, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 32)
      engine.freeDevice(oDev)
      for i in 0 ..< 32:
        check res[i] == uint32(i + 100)

    test "kernel own ref and the chain share one attribute param":
      var engine = bkMetal.init()
      engine.ingest(chainMsl)
      let oDev = engine.allocDevice(uint32, 32)
      engine.launchKernel<<(grid: (1, 1), blk: (32, 1))>>("directKernel", oDev, ())
      var res: array[32, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 32)
      engine.freeDevice(oDev)
      for i in 0 ..< 32:
        check res[i] == uint32(i + 100)

when isMainModule:
  runTest()
