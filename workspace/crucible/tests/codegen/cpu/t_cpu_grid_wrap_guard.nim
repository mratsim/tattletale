## CPU engine: the grid workgroup total guard with overflow checks off.
##
## Test that a launch whose grid workgroup total exceeds int64 refuses:
##
## - the guard multiplies the grid dims with wrap-proof math, a product
##   that wraps inside int64 must not pass the int32 bound
## - grid (2^21, 2^21, 2^22) has a workgroup total of 2^64, which wraps
##   to 0 when the guard computes it with overflow checks off
##
## Run:
##   nim test_crucible_cpu
import std/strutils
import workspace/crucible

proc probe(outBuf: ptr UncheckedArray[int32]) =
  outBuf[0] = 7

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var o: array[1, int32]
  var refusal = ""
  try:
    engine.launchCpuFn probe(cast[ptr UncheckedArray[int32]](addr o[0])) do:
      grid = (2097152, 2097152, 4194304)
      blk = (1, 1, 1)
  except AssertionDefect as e:
    refusal = e.msg
  doAssert refusal.len > 0,
    "grid (2097152, 2097152, 4194304) has a workgroup total of 2^64, which wraps to 0 in int64, the guard must refuse loudly and not pass a wrapped product"
  doAssert o[0] == 0, "a refused launch must not run the kernel"
  echo "[OK] launchCpuFn: a grid total beyond int64 refuses with overflow checks off"

runTest()
