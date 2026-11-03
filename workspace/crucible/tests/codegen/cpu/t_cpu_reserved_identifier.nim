## CPU engine: kernel local named after a C reserved word.
##
## Test that a kernel local whose name is a C keyword compiles and computes:
##
## - the CPU pipeline renames identifiers colliding with C reserved
##   words before printing
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc probe(outBuf: ptr UncheckedArray[int32]) =
  var register: int32 = 42
  outBuf[0] = register

proc runTest() =
  var engine = newEngine(runCpu)
  var buf: array[4, int32]
  engine.launchCpuFn probe(cast[ptr UncheckedArray[int32]](addr buf[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  doAssert buf[0] == 42, "the local must read 42, got " & $buf[0]
  echo "[OK] launchCpuFn: a kernel local named after a C reserved word renames and computes"

runTest()
