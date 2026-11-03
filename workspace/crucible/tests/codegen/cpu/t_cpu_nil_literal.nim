## CPU engine: a nil pointer literal in a kernel body.
##
## Test that a kernel comparing a pointer against nil compiles and runs in both launch modes:
##
## - the C printer emits NULL for a nil expression
## - only the serial unit prefix must define NULL itself
## - the pooled units reach stddef.h through the threadpool force-includes
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc probe(flag: bool, outBuf: ptr UncheckedArray[int32]) =
  var p: ptr int32 = nil
  if flag:
    p = addr outBuf[0]
  if p != nil:
    p[] = 42

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var o: array[4, int32]
  engine.launchCpuFn probe(true, cast[ptr UncheckedArray[int32]](addr o[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  doAssert o[0] == 42, "the pointer must be non-nil and store 42, got " & $o[0]
  echo "[OK] launchCpuFn: a nil pointer literal compiles and branches in both launch modes"

runTest()
