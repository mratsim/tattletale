## Metal: a device fn whose declared param is named like a coordinate builtin.
## A declared param named like a builtin binds by symbol: no hidden param is
## injected and the name binds to the declared param.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_device_fn_builtin_param_name.nim

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
      var engine = newEngine(runMetal)
      engine.ingest(paramMsl)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("paramKernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42'u32

when isMainModule:
  runTest()
