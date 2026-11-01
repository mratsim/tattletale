## CUDA: an array declaration binding a call result refuses at compile time.
##
## Test that an array declaration whose initializer is a device call
## dies in the array-assign decomposition before any shader is emitted, naming the pass:
##
##   Error: pass decomposeArrayAssign: in proc kern: array declaration binds a call result
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/control_flow/manual_cuda_array_assign_call_init_refused.nim

import workspace/crucible

proc makeArr(): array[8, float32] {.device.} =
  for i in 0 ..< 8:
    result[i] = float32(i)

const callInitCuda = cuda:
  proc kern(output: ptr UncheckedArray[float32]) {.global.} =
    var a: array[8, float32] = makeArr()
    output[0] = a[3]

proc runTest() =
  var engine = newEngine(runCuda)
  engine.ingest(callInitCuda)
  echo "UNREACHABLE — the call-result array declaration must be refused at compile time"

when isMainModule:
  runTest()
