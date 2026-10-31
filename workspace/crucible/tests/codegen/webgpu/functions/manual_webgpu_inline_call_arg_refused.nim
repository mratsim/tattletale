## WebGPU: a call-valued argument to an inlinable pointer callee refuses.
##
## Test that a void device function taking a pointer that cannot cross a WGSL
## function signature refuses when its argument is a call expression:
## inlining would substitute the call at every parameter use,
## re-evaluating it once per use.
##
## Run (must FAIL to compile):
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/functions/manual_webgpu_inline_call_arg_refused.nim

import workspace/crucible

proc getN(): int32 =
  return 3

proc copyInto(p: ptr UncheckedArray[float32], n: int32) =
  for i in 0 ..< n:
    p[i] = float32(i)

const kernel = webgpu:
  proc k(outBuf: ptr UncheckedArray[float32]) {.global.} =
    copyInto(outBuf, getN())

proc runTest() =
  echo "UNREACHABLE: the call-valued argument must refuse in the WGSL lowering"

when isMainModule:
  runTest()
