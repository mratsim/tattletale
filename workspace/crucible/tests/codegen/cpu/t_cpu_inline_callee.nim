## CPU engine: `{.inline.}` callee in a kernel.
##
## Test that a kernel calling a `{.inline.}` helper compiles, links and computes:
##
## - the C `inline` qualifier without `static` emits no external definition,
##   a call that the optimizer does not inline fails to link
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc helper(x: int32): int32 {.inline.} =
  var acc = x
  for i in 1 ..< 5:
    acc = acc + x
  acc

proc probe(outBuf: ptr UncheckedArray[int32]) =
  outBuf[0] = helper(3)

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var buf: array[4, int32]
  engine.launchCpuFn probe(cast[ptr UncheckedArray[int32]](addr buf[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  doAssert buf[0] == 15, "helper(3) must read 15, got " & $buf[0]
  echo "[OK] launchCpuFn: an {.inline.} callee compiles, links and computes"

runTest()
