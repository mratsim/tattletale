## CPU engine: fixed-size array local in a kernel.
##
## Test that a kernel local of `array[N, T]` declares with its C array
## dimensions and indexes correctly, the declaration site prints
## `T name[N]`.
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc probe(outBuf: ptr UncheckedArray[int32]) =
  var buf: array[4, int32]
  buf[0] = 41
  buf[1] = buf[0] + 1
  outBuf[0] = buf[1]

proc runTest() =
  var engine = newEngine(runCpu)
  var buf: array[4, int32]
  engine.launchCpuFn probe(cast[ptr UncheckedArray[int32]](addr buf[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  doAssert buf[0] == 42, "the array local must read 41 + 1, got " & $buf[0]
  echo "[OK] launchCpuFn: an array local declares with its dimensions and indexes"

runTest()
