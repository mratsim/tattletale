## launchCpuFn refuses a non-literal workgroupsPerTask in serial mode.
##
## The serial launch never evaluates the knob, a non-literal expression
## would silently vanish. The compile-time refusal names the construct:
##
##   Error: launchCpuFn: the serial launch has no pool tasks, workgroupsPerTask must be an integer literal (it is ignored), got the expression k
##
## With `--threads:on` the same source compiles and runs and the pooled
## launch validates the knob at runtime.
##
## Run (the serial compile must fail):
##   nim c --hints:off --warnings:off --threads:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cpu/manual_cpu_serial_knob_refused.nim
## Run (the pooled compile runs):
##   nim c -r --hints:off --warnings:off --threads:on \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cpu/manual_cpu_serial_knob_refused.nim

import workspace/crucible

proc probe(outBuf: ptr UncheckedArray[uint32]) =
  outBuf[0] = 1

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var k = 4
  var outBuf: array[1, uint32]
  engine.launchCpuFn probe(cast[ptr UncheckedArray[uint32]](addr outBuf[0])) do:
    grid = (4, 1)
    blk = (1, 1)
    workgroupsPerTask = k
  doAssert outBuf[0] == 1, "the pooled launch runs with a non-literal knob"
  echo "[OK] pooled launch accepts a non-literal workgroupsPerTask"

when isMainModule:
  runTest()
