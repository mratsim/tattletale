## CPU engine: field read through a pointer-to-struct parameter.
##
## Test that a `p[].field` read on a `ptr Struct` parameter compiles and computes:
##
## - the dereferenced base of a field access needs parentheses in C
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

type Pair = object
  a: int32
  b: int32

proc probe(p: ptr Pair, outBuf: ptr UncheckedArray[int32]) =
  outBuf[0] = p[].a + p[].b

proc runTest() =
  var engine = newEngine(runCpu)
  var pr = Pair(a: 3, b: 4)
  var buf: array[4, int32]
  engine.launchCpuFn probe(addr pr, cast[ptr UncheckedArray[int32]](addr buf[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  doAssert buf[0] == 7, "p[].a + p[].b must read 3 + 4, got " & $buf[0]
  echo "[OK] launchCpuFn: a field read through a pointer parameter compiles and computes"

runTest()
