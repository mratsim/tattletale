## launchCpuFn refuses a launch the runtime argument types cannot discriminate.
##
## Test that a compile-time refusal fires when same-arity overloads exist
## but no candidate accepts the argument types exactly:
##
## - an `int` literal matches neither the `int32` nor the `int64` overload,
##   the launch does not rank implicit conversions
## - the refusal lists every candidate's signature
##
## Run (the compile must fail, both thread modes):
##   nim c --hints:off --warnings:off --threads:off \
##     --path:. --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cpu/manual_cpu_overload_refused.nim
##   nim c --hints:off --warnings:off --threads:on \
##     --path:. --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cpu/manual_cpu_overload_refused.nim

import workspace/crucible

proc kern(outBuf: ptr UncheckedArray[int32], x: int32) =
  outBuf[0] = x

proc kern(outBuf: ptr UncheckedArray[int32], x: int64) =
  outBuf[0] = int32(x)

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var buf: array[4, int32]
  engine.launchCpuFn kern(cast[ptr UncheckedArray[int32]](addr buf[0]), 5) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)

when isMainModule:
  runTest()
