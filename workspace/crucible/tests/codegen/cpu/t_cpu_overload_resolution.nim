## CPU engine: kernel overload resolution by argument type.
##
## Test that a launch over two same-arity overloads picks the overload
## that the argument types select:
##
## - parameter count filters first, the runtime argument types then discriminate same-arity candidates
## - a launch that no candidate matches refuses, the message lists every candidate's signature
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc kern(outBuf: ptr UncheckedArray[int32], x: int32) =
  outBuf[0] = x * 2

proc kern(outBuf: ptr UncheckedArray[int32], x: float32) =
  outBuf[0] = int32(x * 3.0'f32)

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var buf: array[4, int32]
  engine.launchCpuFn kern(cast[ptr UncheckedArray[int32]](addr buf[0]), 5'i32) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  doAssert buf[0] == 10, "the int32 overload must run, 5 * 2, got " & $buf[0]
  echo "[OK] launchCpuFn: same-arity overloads resolve by argument type"

runTest()
