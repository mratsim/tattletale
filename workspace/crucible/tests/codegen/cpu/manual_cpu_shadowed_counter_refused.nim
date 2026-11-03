## launchCpuFn refuses a kernel local shadowing a generated launch counter.
##
## Test that a compile-time refusal fires when a kernel local shares a name
## with a counter the launch generates around the inlined kernel body:
##
## - the iteration function generates the workgroup counters `crucibleWg`,
##   `crucibleWgTotal`, `crucibleWgEnd`
## - and the coordinate counters `crucibleGx`, `crucibleGy`, `crucibleGz`,
##   `crucibleBx`, `crucibleBy`, `crucibleBz`
## - a kernel local of one of those names silently shadows the counter,
##   the coordinate reads evaluate to the local
##
## The refusal names the kernel and the shadowing local.
##
## Run (the compile must fail, both thread modes):
##   nim c --hints:off --warnings:off --threads:off \
##     --path:. --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cpu/manual_cpu_shadowed_counter_refused.nim
##   nim c --hints:off --warnings:off --threads:on \
##     --path:. --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cpu/manual_cpu_shadowed_counter_refused.nim

import workspace/crucible

proc shadowKernel(outBuf: ptr UncheckedArray[int32]) =
  var crucibleGx: int32 = 99
  outBuf[0] = int32(threadgroup_position_in_grid.x) + crucibleGx

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var buf: array[4, int32]
  engine.launchCpuFn shadowKernel(cast[ptr UncheckedArray[int32]](addr buf[0])) do:
    grid = (3, 1, 1)
    blk = (1, 1, 1)
  doAssert buf[0] == 101, "workgroup 2 reads 2 + 99, got " & $buf[0]

when isMainModule:
  runTest()
