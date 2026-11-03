## CPU engine: mutually recursive callees in a kernel.
##
## Test that a kernel calling mutually recursive helpers compiles and computes:
##
## - the generated C emits a prototype before every callee definition,
##   the post-order emission leaves recursion cycles without one
##   otherwise
##
## Run:
##   nim test_crucible_cpu
import workspace/crucible

proc isEven(n: int32): bool
proc isOdd(n: int32): bool

proc isEven(n: int32): bool =
  if n == 0:
    true
  else:
    isOdd(n - 1)

proc isOdd(n: int32): bool =
  if n == 0:
    false
  else:
    isEven(n - 1)

proc probe(outBuf: ptr UncheckedArray[int32]) =
  outBuf[0] = int32(isEven(7))
  outBuf[1] = int32(isOdd(7))
  outBuf[2] = int32(isEven(6))

proc runTest() =
  var engine = newEngine(runCpu)
  var buf: array[4, int32]
  engine.launchCpuFn probe(cast[ptr UncheckedArray[int32]](addr buf[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)
  doAssert buf[0] == 0, "isEven(7) must read false, got " & $buf[0]
  doAssert buf[1] == 1, "isOdd(7) must read true, got " & $buf[1]
  doAssert buf[2] == 1, "isEven(6) must read true, got " & $buf[2]
  echo "[OK] launchCpuFn: mutually recursive callees compile and compute"

runTest()
