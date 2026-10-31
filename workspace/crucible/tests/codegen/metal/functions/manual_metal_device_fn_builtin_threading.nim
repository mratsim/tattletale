## Metal: device-fn builtin threading is deferred and the emitted MSL is invalid.
##
## Test that a `.device.` proc reading a coordinate builtin stays a device
## function and Metal compilation fails on the builtin reference:
## a device function cannot evaluate the attributed kernel parameters.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/manual_metal_device_fn_builtin_threading.nim

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

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runMetal)
  engine.ingest(chainMsl)
  echo "UNREACHABLE — the device fn builtin reads must fail Metal compilation"

when isMainModule:
  runTest()
