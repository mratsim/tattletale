## CUDA: a whole-array assignment from a constructor refuses at compile time.
##
## Test that an array assignment whose right side is an array constructor
## dies in the array-assign decomposition before any shader is emitted, naming the pass:
##
##   Error: pass decomposeArrayAssign: in proc kern: whole-array assignment right-hand side is not an lvalue
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/control_flow/manual_cuda_array_assign_constr_rhs_refused.nim

import workspace/crucible

const constrRhsCuda = cuda:
  proc kern(output: ptr UncheckedArray[float32]) {.global.} =
    var a: array[8, float32]
    a = [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32, 5.0'f32, 6.0'f32, 7.0'f32, 8.0'f32]
    output[0] = a[3]

proc runTest() =
  var engine = newEngine(runCuda)
  engine.ingest(constrRhsCuda)
  echo "UNREACHABLE — the constructor array assignment must be refused at compile time"

when isMainModule:
  runTest()
