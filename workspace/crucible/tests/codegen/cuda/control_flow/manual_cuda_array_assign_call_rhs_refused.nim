## CUDA: a whole-array assignment binding a call result refuses at compile time.
##
## Test that an array assignment whose right side is a device call
## dies in the array-assign decomposition before any shader is emitted.
## The refusal names the pass:
##
##   Error: pass decomposeArrayAssign: in proc kern: whole-array assignment binds a call result
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/control_flow/manual_cuda_array_assign_call_rhs_refused.nim

import workspace/crucible

proc makeArr(): array[8, float32] {.device.} =
  for i in 0 ..< 8:
    result[i] = float32(i)

const callRhsCuda = cuda:
  proc kern(output: ptr UncheckedArray[float32]) {.global.} =
    var a: array[8, float32]
    a = makeArr()
    output[0] = a[3]

proc runTest() =
  var engine = newEngine(runCuda)
  engine.ingest(callRhsCuda)
  echo "UNREACHABLE — the call-result array assignment must be refused at compile time"

when isMainModule:
  runTest()
