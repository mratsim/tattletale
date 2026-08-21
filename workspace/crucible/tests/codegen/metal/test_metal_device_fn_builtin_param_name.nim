## Metal: a device fn whose declared param is named like a coordinate builtin.
## `echoParam` declares `thread_index_in_threadgroup: uint32` and writes it
## through. The printer must distinguish the param by symbol (gbkNone) from the
## builtin, so no hidden param is injected and the name binds to the declared
## param. A collision would fail MSL compilation at ingest with a duplicate
## parameter error.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_device_fn_builtin_param_name.nim

import std/unittest
import workspace/crucible

proc echoParam(p: ptr UncheckedArray[uint32];
               thread_index_in_threadgroup: uint32) {.device.} =
  p[0] = thread_index_in_threadgroup

const paramMsl = metal:
  proc paramKernel(p: ptr UncheckedArray[uint32]) {.global.} =
    echoParam(p, 42'u32)

proc runTest() =
  suite "Metal - device fn param named like a builtin":
    test "the declared param binds; no hidden param collides":
      var engine = bkMetal.init()
      engine.ingest(paramMsl)
      var res: array[1, uint32]
      engine.run("paramKernel", res, ())
      check res[0] == 42'u32

when isMainModule:
  runTest()
