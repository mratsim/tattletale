## launchCpuFn refuses pooled launches whose data struct exceeds int32.
##
## Test that a pooled launch refuses at compile time when the runtime parameters exceed the int32 task environment size:
##
## - the pooled tail casts sizeof(crucibleData) to int32_t, a struct at or above 2^31 bytes wraps the cast
## - a wrapped negative size aborts the pool's allocation, a wrapped
##   positive size truncates the payload copy and the task body reads
##   beyond the allocation
##
## Run (the compile must fail, pooled mode only, the serial launch stores no environment size):
##   nim c --hints:off --warnings:off --threads:on \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cpu/manual_cpu_huge_param_refused.nim

import workspace/crucible

type HugeParam = array[3000000000, int32]

proc probe(data: HugeParam, outBuf: ptr UncheckedArray[int32]) =
  outBuf[0] = data[0]

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var huge: HugeParam
  var buf: array[4, int32]
  engine.launchCpuFn probe(huge, cast[ptr UncheckedArray[int32]](addr buf[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)

when isMainModule:
  runTest()
