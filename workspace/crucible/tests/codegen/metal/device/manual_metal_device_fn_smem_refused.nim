## Metal: a device function's `{.smem.}` local refuses, only a kernel owns threadgroup storage.
##
## Test that a `{.device.}` proc declaring a `{.smem.}` local fails to compile
## at `mslRewriteSmemLocals` with the message naming the proc and the local.
## Kernel-owned `{.smem.}` behavior is covered by `t_metal_shared_memory.nim`.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/manual_metal_device_fn_smem_refused.nim

import workspace/crucible

const smemRefusedMsl = metal:
  proc fold(input: ptr UncheckedArray[float32], output: ptr UncheckedArray[float32]) {.device.} =
    var scratch {.smem.}: array[64, float32]
    let t = thread_position_in_threadgroup.x
    scratch[t] = input[t]

  proc sumSmem(output: ptr UncheckedArray[float32], input: ptr UncheckedArray[float32]) {.global.} =
    fold(input, output)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runMetal)
  engine.ingest(smemRefusedMsl)
  echo "UNREACHABLE — the device fn smem local must refuse"

when isMainModule:
  runTest()
