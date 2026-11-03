## CPU engine: side-effecting condition of a branch with empty bodies.
##
## Test that an `if` with empty branch bodies still emits the condition call:
##
## - each branch contains only a `const` section, which lowers to no
##   statements
## - the condition call writes into the output buffer, a dropped condition
##   leaves the buffer untouched
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc bump(outBuf: ptr UncheckedArray[int32]): int32 =
  outBuf[1] = 7
  1

proc probe(outBuf: ptr UncheckedArray[int32]) =
  if bump(outBuf) == 1:
    const k = 1
  else:
    const j = 2
  outBuf[0] = outBuf[1]

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var buf: array[4, int32]
  engine.launchCpuFn probe(cast[ptr UncheckedArray[int32]](addr buf[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  doAssert buf[1] == 7,
    "the branch condition call must run, got buf[1] = " & $buf[1]
  doAssert buf[0] == 7,
    "the branch condition call must run, got buf[0] = " & $buf[0]
  echo "[OK] launchCpuFn: a side-effecting condition of a branch with empty bodies still runs"

runTest()
